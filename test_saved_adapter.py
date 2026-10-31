
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
ADAPTER_DIR = ROOT / "outputs" / "sft_smoke_test"

def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR)

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.float16 if device == "mps" else torch.float32,
    )

    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.to(device)
    model.eval()

    prompt = (
        "<|im_start|>user\n"
        "You are negotiating over books, hats, and balls.\n"
        "Your item counts: [1, 1, 1]\n"
        "Your private values: [5, 1, 2]\n"
        "Other agent's item counts: [1, 1, 1]\n"
        "Dialogue so far:\n"
        "THEM: I would like the book.\n"
        "Predict a possible allocation in the original format.\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=60,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = output[0][inputs["input_ids"].shape[1]:]
    print("Generated answer:")
    print(tokenizer.decode(generated, skip_special_tokens=True))

if __name__ == "__main__":
    main()
