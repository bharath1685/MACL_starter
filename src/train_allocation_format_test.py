import json
import re
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = ROOT / "data" / "sft" / "train.jsonl"
VAL_FILE = ROOT / "data" / "sft" / "val.jsonl"

# Separate output directory: the previous adapter is preserved.
OUTPUT_DIR = ROOT / "outputs" / "allocation_format_test"

MAX_LENGTH = 512
TRAIN_LIMIT = 32
VAL_LIMIT = 8
MAX_STEPS = 20

FIELD_RE = re.compile(r"item([0-2])=(-?\d+)")
COUNTS_RE = re.compile(
    r"Your item counts: \[(\d+), (\d+), (\d+)\]"
)
PARTNER_COUNTS_RE = re.compile(
    r"Other agent's item counts: \[(\d+), (\d+), (\d+)\]"
)


def read_valid_allocations(path):
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    valid = []
    rejected = 0

    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            if not line.strip():
                continue

            row = json.loads(line)
            prompt = row.get("prompt", "").strip()
            answer = row.get("answer", "").strip()

            # Require exactly six fields, in the expected order.
            matches = FIELD_RE.findall(answer)
            if len(matches) != 6:
                rejected += 1
                continue

            if [int(x[0]) for x in matches] != [0, 1, 2, 0, 1, 2]:
                rejected += 1
                continue

            values = [int(x[1]) for x in matches]
            if any(value < 0 for value in values):
                rejected += 1
                continue

            own_match = COUNTS_RE.search(prompt)
            partner_match = PARTNER_COUNTS_RE.search(prompt)

            if not own_match or not partner_match:
                rejected += 1
                continue

            own_counts = list(map(int, own_match.groups()))
            partner_counts = list(map(int, partner_match.groups()))

            # The two agents must allocate the same available item supply.
            if own_counts != partner_counts:
                rejected += 1
                continue

            # The two allocations must account for every available item.
            if any(
                values[i] + values[i + 3] != own_counts[i]
                for i in range(3)
            ):
                rejected += 1
                continue

            valid.append({
                "prompt": prompt,
                "answer": answer,
                "source_line": line_number,
            })

    print(f"{path.name}: valid allocation examples={len(valid)}, "
          f"rejected records={rejected}")
    return valid


class AllocationDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer):
        self.items = []

        for row in rows:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are an allocation agent. Output exactly six "
                        "space-separated fields in this order: "
                        "item0=N item1=N item2=N item0=N item1=N item2=N. "
                        "Use non-negative integers only. Output no dialogue, "
                        "explanation, or extra text."
                    ),
                },
                {"role": "user", "content": row["prompt"]},
                {"role": "assistant", "content": row["answer"]},
            ]

            full_result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            prefix_result = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            full_ids = list(full_result["input_ids"])
            prefix_ids = list(prefix_result["input_ids"])

            if full_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    "Chat-template prefix mismatch. "
                    "Training stopped before model updates."
                )

            if len(full_ids) > MAX_LENGTH:
                raise ValueError(
                    f"Example has {len(full_ids)} tokens, exceeding "
                    f"MAX_LENGTH={MAX_LENGTH}. No truncation was applied."
                )

            labels = full_ids.copy()
            labels[:len(prefix_ids)] = [-100] * len(prefix_ids)

            if not any(label != -100 for label in labels):
                raise ValueError("An example has no assistant target tokens.")

            self.items.append({
                "input_ids": torch.tensor(full_ids, dtype=torch.long),
                "labels": torch.tensor(labels, dtype=torch.long),
            })

        if not self.items:
            raise ValueError("No usable allocation examples were found.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


class PadCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)
        input_ids, labels, attention_masks = [], [], []

        for item in batch:
            length = len(item["input_ids"])
            padding = max_len - length

            input_ids.append(torch.cat([
                item["input_ids"],
                torch.full(
                    (padding,), self.pad_token_id, dtype=torch.long
                ),
            ]))

            labels.append(torch.cat([
                item["labels"],
                torch.full((padding,), -100, dtype=torch.long),
            ]))

            attention_masks.append(torch.cat([
                torch.ones(length, dtype=torch.long),
                torch.zeros(padding, dtype=torch.long),
            ]))

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
        }


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS is unavailable. This script expects your "
            "previously verified Mac MPS setup."
        )

    train_rows = read_valid_allocations(TRAIN_FILE)
    val_rows = read_valid_allocations(VAL_FILE)

    train_rows = train_rows[:TRAIN_LIMIT]
    val_rows = val_rows[:VAL_LIMIT]

    if len(train_rows) < TRAIN_LIMIT:
        raise ValueError(
            f"Need {TRAIN_LIMIT} valid training allocations; "
            f"found {len(train_rows)}."
        )

    if len(val_rows) < VAL_LIMIT:
        raise ValueError(
            f"Need {VAL_LIMIT} valid validation allocations; "
            f"found {len(val_rows)}."
        )

    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(val_rows)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = AllocationDataset(train_rows, tokenizer)
    val_dataset = AllocationDataset(val_rows, tokenizer)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    model = get_peft_model(
        model,
        LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        ),
    )
    model.to("mps")
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        max_steps=MAX_STEPS,
        learning_rate=1e-4,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=PadCollator(tokenizer.pad_token_id),
    )

    print("\nStarting allocation-format training test...")
    result = trainer.train()
    print("Training loss:", result.training_loss)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Adapter saved to:", OUTPUT_DIR)

    # Evaluate generation on held-out official validation examples.
    model.config.use_cache = True
    model.eval()
    valid_format = 0
    valid_counts = 0

    print("\nHELD-OUT VALIDATION RESULTS")

    for index, row in enumerate(val_rows, start=1):
        prompt = tokenizer.apply_chat_template(
            [
                {
                    "role": "system",
                    "content": (
                        "You are an allocation agent. Output exactly six "
                        "space-separated fields in this order: "
                        "item0=N item1=N item2=N item0=N item1=N item2=N. "
                        "Use non-negative integers only. Output no dialogue, "
                        "explanation, or extra text."
                    ),
                },
                {"role": "user", "content": row["prompt"]},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(prompt, return_tensors="pt").to("mps")

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=40,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        matches = FIELD_RE.findall(answer)

        format_ok = (
            len(matches) == 6
            and [int(x[0]) for x in matches] == [0, 1, 2, 0, 1, 2]
        )
        valid_format += int(format_ok)

        own_match = COUNTS_RE.search(row["prompt"])
        own_counts = list(map(int, own_match.groups()))
        values = [int(x[1]) for x in matches] if format_ok else []

        counts_ok = (
            format_ok
            and all(value >= 0 for value in values)
            and all(
                values[i] + values[i + 3] == own_counts[i]
                for i in range(3)
            )
        )
        valid_counts += int(counts_ok)

        print(f"\nExample {index}")
        print("Prompt item counts:", own_counts)
        print("Generated:", repr(answer[:200]))
        print("Format valid:", format_ok)
        if format_ok:
            print("Generated allocation:", values[:3])
            print("Other allocation:", values[3:])
            print(
                "Combined counts:",
                [values[i] + values[i + 3] for i in range(3)],
            )
        print("Counts valid:", counts_ok)
        if format_ok:
            violations = [
                {
                    "item": i,
                    "available": own_counts[i],
                    "allocated_total": values[i] + values[i + 3],
                }
                for i in range(3)
                if values[i] + values[i + 3] != own_counts[i]
            ]
            print("Supply constraint violations:", violations)

    print("\nSUMMARY")
    print(f"Validation examples: {len(val_rows)}")
    print(f"Valid output format: {valid_format}/{len(val_rows)}")
    print(f"Count-consistent outputs: {valid_counts}/{len(val_rows)}")
    print("This is a supervised format test, not a MACL result.")


if __name__ == "__main__":
    main()
