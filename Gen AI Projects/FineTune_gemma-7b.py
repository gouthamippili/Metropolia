import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, GemmaTokenizer, TrainingArguments
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer

# 1. Configuration
model_id = "google/gemma-7b"
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_id, token=os.environ['HUGGINGFACE_API_KEY'])
lora_config = LoraConfig(r=8, target_modules=["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM")
model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"":0}, token=os.environ['HUGGINGFACE_API_KEY'])

# 2. Dataset loading and mapping
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)

# 3 Before training
def generate_text(text):
    inputs = tokenizer(text, return_tensors="pt").to("cuda:0")
    outputs = model.generate(**inputs, max_new_tokens=20)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("Before training\n")
generate_text("Quote: Imagination is more")

# 4. After training
trainer = SFTTrainer(
    model=model, 
    train_dataset=data["train"],
    max_seq_length = 1024,
    args=TrainingArguments(
        per_device_train_batch_size=1, 
        gradient_accumulation_steps=4, 
        warmup_steps=2, 
        max_steps=10, 
        learning_rate=2e-4, 
        fp16=True, 
        logging_steps=1, 
        output_dir="outputs", 
        optim="paged_adamw_8bit"
    ), 
    peft_config=lora_config, 
    formatting_func=lambda example: [f"Quote: {example['quote'][0]}\nAuthor: {example['author'][0]}"]
)
trainer.train()
print("\n ######## \nAfter training\n")
generate_text("Quote: Imagination is")
model.save_pretrained("outputs")