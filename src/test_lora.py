
import torch
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
    ).to("mps")

    config = LoraConfig(
        r=4,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    tokenizer_inputs = {
        "input_ids": torch.tensor(
            [[151644, 8948, 198, 151645]],
            dtype=torch.long,
            device="mps",
        )
    }

    # Use ordinary text tokenization rather than relying on
    # the example token IDs above for the actual training test.
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    batch = tokenizer(
        "You are negotiating a fair deal.",
        return_tensors="pt",
    ).to("mps")
    batch["labels"] = batch["input_ids"].clone()

    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(**batch)
    outputs.loss.backward()

    trainable_with_grad = sum(
        1 for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    )

    print("Loss:", float(outputs.loss.detach().cpu()))
    print("Trainable tensors with gradients:", trainable_with_grad)
    print("LoRA forward/backward test passed.")

if __name__ == "__main__":
    main()
