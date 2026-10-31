"""Two-agent MACL prototype for the project's reward-ranked SFT proxy.

This is not PPO/GRPO or interactive RL. It tests the central mechanism:
two independent LoRA agents exchange a scalar Fisher-weighted adapter-drift
signal, which controls EWC strength during continual stages.
"""
import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


def read_jsonl(path, limit=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    if not rows:
        raise ValueError(f"No examples in {path}")
    return rows


def choose_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_rows(rows, tokenizer, max_length):
    result = []
    for row in rows:
        messages = [
            {"role": "user", "content": row["prompt"].strip()},
            {"role": "assistant", "content": row["answer"].strip()},
        ]

        full_result = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
        prefix_result = tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )

        full_ids = list(full_result["input_ids"])
        prefix_ids = list(prefix_result["input_ids"])

        if full_ids[:len(prefix_ids)] != prefix_ids:
            raise ValueError("Chat-template prefix mismatch")

        if len(full_ids) > max_length:
            answer_len = len(full_ids) - len(prefix_ids)
            if answer_len >= max_length:
                raise ValueError("Answer exceeds max_length")
            keep_prompt = max_length - answer_len
            full_ids = full_ids[-max_length:]
            prefix_len = keep_prompt
        else:
            prefix_len = len(prefix_ids)

        labels = [-100] * prefix_len + full_ids[prefix_len:]
        if not any(x != -100 for x in labels):
            continue

        result.append((full_ids, labels))

    if not result:
        raise ValueError("No examples remain after tokenization/truncation")
    return result


def make_batch(examples, indices, pad_id, device):
    ids, labels = [], []
    for i in indices:
        x, y = examples[i]
        ids.append(torch.tensor(x, dtype=torch.long))
        labels.append(torch.tensor(y, dtype=torch.long))
    input_ids = pad_sequence(ids, batch_first=True, padding_value=pad_id)
    target = pad_sequence(labels, batch_first=True, padding_value=-100)
    mask = input_ids.ne(pad_id).long()
    return {"input_ids": input_ids.to(device), "attention_mask": mask.to(device),
            "labels": target.to(device)}


def trainable_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def fisher_estimate(model, examples, pad_id, device, n_examples, rng):
    params = trainable_params(model)
    fisher = {n: torch.zeros_like(p, dtype=torch.float32, device="cpu") for n, p in params}
    model.train()
    chosen = [rng.randrange(len(examples)) for _ in range(min(n_examples, len(examples)))]
    count = 0
    for idx in chosen:
        model.zero_grad(set_to_none=True)
        batch = make_batch(examples, [idx], pad_id, device)
        loss = model(**batch).loss
        loss.backward()
        for n, p in params:
            if p.grad is not None:
                fisher[n] += p.grad.detach().float().cpu().square()
        count += 1
    for n in fisher:
        fisher[n] /= max(count, 1)
    # Normalize mean Fisher to 1 so lambda has a usable, reproducible scale.
    total = sum(v.sum().item() for v in fisher.values())
    numel = sum(v.numel() for v in fisher.values())
    mean = total / max(numel, 1)
    if mean > 0:
        for n in fisher:
            fisher[n] /= mean
    return fisher


def snapshot(model):
    return {n: p.detach().float().cpu().clone() for n, p in trainable_params(model)}


def weighted_drift(model, anchor, fisher):
    total = 0.0
    with torch.no_grad():
        for n, p in trainable_params(model):
            delta = p.detach().float().cpu() - anchor[n]
            total += (fisher[n] * delta.square()).sum().item()
    return max(total, 0.0)


def ewc_penalty(model, anchor, fisher, device):
    penalty = torch.zeros((), device=device)
    for n, p in trainable_params(model):
        f = fisher[n].to(device=device, dtype=torch.float32)
        a = anchor[n].to(device=device, dtype=torch.float32)
        penalty = penalty + (f * (p.float() - a).square()).sum()
    return penalty


@torch.no_grad()
def eval_loss(model, examples, pad_id, device, limit):
    model.eval()
    vals = []
    for start in range(0, min(len(examples), limit), 1):
        batch = make_batch(examples, [start], pad_id, device)
        vals.append(float(model(**batch).loss.detach().float().cpu()))
    return sum(vals) / max(len(vals), 1)


def build_model(model_name, tokenizer, device, seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
    )
    model.config.use_cache = False
    lora = LoraConfig(
        r=4, lora_alpha=8, lora_dropout=0.0,
        target_modules=["q_proj", "v_proj"], bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.to(device)
    return model


def train_stage(model, examples, pad_id, device, rng, steps, batch_size,
                method, anchor=None, fisher=None, partner_signal_fn=None,
                fixed_lambda=0.01, min_lambda=0.002, max_lambda=0.05,
                signal_scale=1.0, lr=1e-4, optimizer=None):
    model.train()
    params = [p for _, p in trainable_params(model)]
    if optimizer is None:
        optimizer = torch.optim.AdamW(params, lr=lr)
    trace = []
    for step in range(steps):
        indices = [rng.randrange(len(examples)) for _ in range(batch_size)]
        batch = make_batch(examples, indices, pad_id, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        ewc_value = torch.zeros((), device=device)
        lam = 0.0
        received = 0.0
        if method in ("ewc", "macl") and anchor is not None and fisher is not None:
            ewc_value = ewc_penalty(model, anchor, fisher, device)
            if method == "ewc":
                lam = fixed_lambda
            else:
                received = float(partner_signal_fn()) if partner_signal_fn else 0.0
                # Bounded dynamic lambda: rises as partner's protected drift grows.
                frac = received / (received + signal_scale) if received > 0 else 0.0
                lam = min_lambda + (max_lambda - min_lambda) * frac
        total_loss = loss + lam * ewc_value
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            trace.append({
                "step": step + 1, "task_loss": float(loss.detach().float().cpu()),
                "ewc_penalty": float(ewc_value.detach().float().cpu()),
                "weighted_ewc_contribution": float((lam * ewc_value).detach().float().cpu()),
                "ewc_to_task_loss_ratio": float((lam * ewc_value / loss.detach().clamp_min(1e-8)).detach().float().cpu()),
                "lambda": lam, "received_partner_signal": received,
            })
    return trace, optimizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["naive", "ewc", "macl"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data-dir", default="data/continual_tasks")
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--train-examples", type=int, default=256)
    ap.add_argument("--val-examples", type=int, default=64)
    ap.add_argument("--steps-per-task", type=int, default=30)
    ap.add_argument("--fisher-examples", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--max-length", type=int, default=384)
    ap.add_argument("--fixed-lambda", type=float, default=1.0)
    ap.add_argument("--min-lambda", type=float, default=0.1)
    ap.add_argument("--max-lambda", type=float, default=5.0)
    ap.add_argument("--signal-scale", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"outputs/macl_sft_{args.method}_seed{args.seed}"
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose a new --output")
    out.mkdir(parents=True)

    random.seed(args.seed)
    device = choose_device()
    print(f"Device: {device}; method: {args.method}")
    data_dir = Path(args.data_dir)
    # Load full splits, then shuffle each independently before limiting. The
    # source files are score-ranked, so taking their first N rows creates
    # artificial A/B overlap and makes validation losses look identical.
    raw = {
        "A": read_jsonl(data_dir / "task_a_train.jsonl"),
        "B": read_jsonl(data_dir / "task_b_train.jsonl"),
        "A_val": read_jsonl(data_dir / "task_a_val.jsonl"),
        "B_val": read_jsonl(data_dir / "task_b_val.jsonl"),
    }
    for offset, name in enumerate(("A", "B", "A_val", "B_val")):
        random.Random(args.seed + 500 + offset).shuffle(raw[name])
    raw["A"] = raw["A"][:args.train_examples]
    raw["B"] = raw["B"][:args.train_examples]
    raw["A_val"] = raw["A_val"][:args.val_examples]
    raw["B_val"] = raw["B_val"][:args.val_examples]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id
    data = {k: prepare_rows(v, tokenizer, args.max_length) for k, v in raw.items()}
    # Keep both agents' adapters live: their scalar signals are exchanged each step.
    models = [
        build_model(args.model, tokenizer, device, args.seed + i)
        for i in range(2)
    ]
    histories = {"agent0": [], "agent1": []}
    anchors, fishers = [None, None], [None, None]
    py_rngs = [random.Random(args.seed + 101), random.Random(args.seed + 202)]

    def evaluate(stage):
        return {
            f"agent{i}": {
                "task_a_val_loss": eval_loss(models[i], data["A_val"], pad_id, device, args.val_examples),
                "task_b_val_loss": eval_loss(models[i], data["B_val"], pad_id, device, args.val_examples),
            } for i in range(2)
        }

    metrics = {"method": args.method, "seed": args.seed, "device": str(device),
               "steps_per_task": args.steps_per_task, "train_examples": args.train_examples,
               "fisher_examples": args.fisher_examples, "stages": {}}
    metrics["stages"]["before_A"] = evaluate("before_A")

    # Task A: self-interested reward-ranked SFT proxy.
    for i in range(2):
        trace, _ = train_stage(models[i], data["A"], pad_id, device, py_rngs[i],
                               args.steps_per_task, args.batch_size, "naive", lr=args.lr)
        histories[f"agent{i}"].append({"task": "A", "trace": trace})
    metrics["stages"]["after_A"] = evaluate("after_A")
    for i in range(2):
        anchors[i] = snapshot(models[i])
        fishers[i] = fisher_estimate(
            models[i], data["A"], pad_id, device, args.fisher_examples, py_rngs[i]
        )

    # Task B: fairness-shaped reward-ranked SFT proxy.
    def signal_for(i):
        j = 1 - i
        return weighted_drift(models[j], anchors[j], fishers[j])

    stage_b_optimizers = [
        torch.optim.AdamW([p for _, p in trainable_params(models[i])], lr=args.lr)
        for i in range(2)
    ]
    for step_pair in range(args.steps_per_task):
        for i in range(2):
            method = args.method
            trace, stage_b_optimizers[i] = train_stage(
                models[i], data["B"], pad_id, device, py_rngs[i], 1,
                args.batch_size, method, anchors[i], fishers[i],
                partner_signal_fn=(lambda i=i: signal_for(i)),
                fixed_lambda=args.fixed_lambda, min_lambda=args.min_lambda,
                max_lambda=args.max_lambda, signal_scale=args.signal_scale, lr=args.lr,
                optimizer=stage_b_optimizers[i],
            )
            histories[f"agent{i}"].append({"task": "B", "trace": trace})
    metrics["stages"]["after_B"] = evaluate("after_B")
    metrics["communication"] = {
        "agent0_receives_agent1_signal": signal_for(0),
        "agent1_receives_agent0_signal": signal_for(1),
        "dynamic_lambda_formula": "min_lambda + (max_lambda-min_lambda) * signal/(signal+signal_scale)",
        "min_lambda": args.min_lambda, "max_lambda": args.max_lambda,
        "signal_scale": args.signal_scale,
        "note": "Scalar is partner's normalized-Fisher-weighted LoRA drift; logged as a prototype design choice.",
    }

    # Task A-prime: revisit self-interested task; keep Task-A anchor for retention.
    stage_aprime_optimizers = [
        torch.optim.AdamW([p for _, p in trainable_params(models[i])], lr=args.lr)
        for i in range(2)
    ]
    for step_pair in range(args.steps_per_task):
        for i in range(2):
            trace, stage_aprime_optimizers[i] = train_stage(
                models[i], data["A"], pad_id, device, py_rngs[i], 1,
                args.batch_size, args.method, anchors[i], fishers[i],
                partner_signal_fn=(lambda i=i: signal_for(i)),
                fixed_lambda=args.fixed_lambda, min_lambda=args.min_lambda,
                max_lambda=args.max_lambda, signal_scale=args.signal_scale, lr=args.lr,
                optimizer=stage_aprime_optimizers[i],
            )
            histories[f"agent{i}"].append({"task": "A_prime", "trace": trace})
    metrics["stages"]["after_A_prime"] = evaluate("after_A_prime")
    metrics["caveats"] = [
        "This is continual supervised fine-tuning on reward-ranked historical examples, not PPO/GRPO or online MARL.",
        "Task A/B data are filtered human outcomes; the fairness reward uses the project's illustrative gap penalty.",
        "Two agents share the same task data distribution but have independent LoRA adapters and exchange a scalar drift signal.",
        "Short pilot settings are for implementation validation only; run multiple seeds and matched budgets before claims.",
    ]
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "training_trace.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    for i, model in enumerate(models):
        model.save_pretrained(out / f"agent{i}_adapter")
    tokenizer.save_pretrained(out / "tokenizer")
    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics, traces, and adapters to {out}")


if __name__ == "__main__":
    main()
