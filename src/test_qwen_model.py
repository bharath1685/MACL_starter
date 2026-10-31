
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading model on Apple GPU (MPS)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model = model.to("mps")
    model.eval()

    prompt = (
        "You are an agent negotiating over hats, books, and balls. "
        "Respond with one short, polite opening offer."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")

    print("Generating a short response...")
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
        )

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    print("\nModel response:")
    print(tokenizer.decode(new_tokens, skip_special_tokens=True))
    print("\nModel loading and generation succeeded.")

if __name__ == "__main__":
    main()
