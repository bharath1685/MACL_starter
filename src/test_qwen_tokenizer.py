
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("Loading tokenizer:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    sample = "Hello! Let's negotiate a fair deal."
    encoded = tokenizer(sample)

    print("Tokenizer loaded successfully.")
    print("Token count:", len(encoded["input_ids"]))
    print("Decoded text:", tokenizer.decode(encoded["input_ids"]))

if __name__ == "__main__":
    main()
