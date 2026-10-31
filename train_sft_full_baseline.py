
import json
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

TRAIN_FILE = ROOT / "data" / "sft" / "train.jsonl"
VAL_FILE = ROOT / "data" / "sft" / "val.jsonl"
OUTPUT_DIR = ROOT / "outputs" / "sft_full_baseline"

MAX_LENGTH = 512
MAX_TRAIN_EXAMPLES = 10095
MAX_VAL_EXAMPLES = 8


# --------------------------------------------------
# Read prepared JSONL data
# --------------------------------------------------

def read_examples(path, limit):
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    examples = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                examples.append(json.loads(line))
                if len(examples) >= limit:
                    break

    if not examples:
        raise ValueError(f"No examples found in {path}")

    return examples


# --------------------------------------------------
# Supervised fine-tuning dataset
# --------------------------------------------------

class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, rows, tokenizer):
        self.items = []

        for index, row in enumerate(rows):
            messages = [
                {
                    "role": "user",
                    "content": row["prompt"].strip(),
                },
                {
                    "role": "assistant",
                    "content": row["answer"].strip(),
                },
            ]

            # Use Qwen's official template for the complete conversation.
            full_result = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
            )
            token_ids = list(full_result["input_ids"])

            # Get the exact prefix at which assistant generation begins.
            prompt_result = tokenizer.apply_chat_template(
                messages[:1],
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
            )
            prefix_ids = list(prompt_result["input_ids"])

            # Refuse to train if the template boundary is inconsistent.
            if token_ids[:len(prefix_ids)] != prefix_ids:
                raise ValueError(
                    f"Chat-template prefix mismatch at example {index}. "
                    "No training was started."
                )

            if len(token_ids) > MAX_LENGTH:
                raise ValueError(
                    f"Example {index} has {len(token_ids)} tokens, "
                    f"exceeding MAX_LENGTH={MAX_LENGTH}. "
                    "Increase MAX_LENGTH or handle truncation explicitly."
                )

            # Ignore user/prompt tokens; calculate loss on assistant tokens.
            labels = token_ids.copy()
            labels[:len(prefix_ids)] = [-100] * len(prefix_ids)

            if not any(token != -100 for token in labels):
                raise ValueError(
                    f"Example {index} has no assistant answer tokens."
                )

            self.items.append(
                {
                    "input_ids": torch.tensor(token_ids, dtype=torch.long),
                    "labels": torch.tensor(labels, dtype=torch.long),
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


# --------------------------------------------------
# Pad examples in each batch
# --------------------------------------------------

class PadCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        max_len = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        labels = []
        attention_masks = []

        for item in batch:
            length = len(item["input_ids"])
            padding = max_len - length

            input_ids.append(
                torch.cat(
                    [
                        item["input_ids"],
                        torch.full(
                            (padding,),
                            self.pad_id,
                            dtype=torch.long,
                        ),
                    ]
                )
            )

            labels.append(
                torch.cat(
                    [
                        item["labels"],
                        torch.full(
                            (padding,),
                            -100,
                            dtype=torch.long,
                        ),
                    ]
                )
            )

            attention_masks.append(
                torch.cat(
                    [
                        torch.ones(length, dtype=torch.long),
                        torch.zeros(padding, dtype=torch.long),
                    ]
                )
            )

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attention_masks),
        }


# --------------------------------------------------
# Train a small LoRA smoke test
# --------------------------------------------------

def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS is unavailable. Check your PyTorch installation."
        )

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    model.to("mps")

    print("Preparing datasets...")
    train_rows = read_examples(TRAIN_FILE, MAX_TRAIN_EXAMPLES)
    filtered_rows = []
    for row in train_rows:
        messages = [
            {"role": "user", "content": row["prompt"].strip()},
            {"role": "assistant", "content": row["answer"].strip()},
        ]
        full = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False, return_dict=True
        )["input_ids"]
        prefix = tokenizer.apply_chat_template(
            messages[:1], tokenize=True, add_generation_prompt=True, return_dict=True
        )["input_ids"]
        if len(full) <= MAX_LENGTH and list(full[:len(prefix)]) == list(prefix):
            filtered_rows.append(row)

    print(f"Training rows retained: {len(filtered_rows)} / {len(train_rows)}")
    train_rows = filtered_rows
    val_rows = read_examples(VAL_FILE, MAX_VAL_EXAMPLES)

    train_dataset = SFTDataset(train_rows, tokenizer)
    val_dataset = SFTDataset(val_rows, tokenizer)

    print(f"Training examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=1,
        learning_rate=1e-4,
        logging_steps=1,
        eval_strategy="steps",
        eval_steps=2,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=PadCollator(tokenizer),
    )

    print("Starting smoke-test training...")
    trainer.train()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print(f"Adapter saved to: {OUTPUT_DIR}")
    print("Smoke-test training completed.")


if __name__ == "__main__":
    main()
