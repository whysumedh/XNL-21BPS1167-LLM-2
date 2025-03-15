import torch
import deepspeed
import json
import os
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from deepspeed.ops.adam import DeepSpeedCPUAdam

deepspeed.init_distributed()

# Model & Tokenizer
model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Apply LoRA for Efficient Fine-tuning
config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)

# Load Dataset
dataset = load_dataset("parquet", data_files="data.parquet")

def format_example(example):
    instruction = example["instruction"]
    intent = example["intent"]
    response = example["response"]
    return {"input": f"Instruction: {instruction}\n", "output": f"Intent: {intent}\nResponse: {response}"}

dataset = dataset.map(format_example)

def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"]
    return inputs

tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Optimized Training Settings
BATCH_SIZE = 16  # Increase batch size for efficiency
grad_accumulation_steps = 4  # Accumulate gradients
train_dataloader = DataLoader(tokenized_dataset["train"], batch_size=BATCH_SIZE, shuffle=True)

# Load DeepSpeed Config
with open("ds_config.json", "r") as f:
    ds_config = json.load(f)

# Enable bf16/fp16 Precision
if "bf16" in ds_config:
    ds_config["bf16"]["enabled"] = True
elif "fp16" in ds_config:
    ds_config["fp16"]["enabled"] = True

ds_config["gradient_accumulation_steps"] = grad_accumulation_steps

# Define Optimizer
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=2e-5)


# Initialize DeepSpeed
model, optimizer, _, _ = deepspeed.initialize(
    model=model, 
    optimizer=optimizer, 
    config=ds_config
)

# TensorBoard Logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

# Training Loop
epochs = 3
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
        inputs = torch.stack(batch["input_ids"]).to(model.device)
        labels = torch.stack(batch["labels"]).to(model.device)
        
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss / grad_accumulation_steps  # Normalize loss
        
        model.backward(loss)
        
        if (step + 1) % grad_accumulation_steps == 0:
            model.step()
        
        total_loss += loss.item() * grad_accumulation_steps
        
        # Log every 10 steps
        if step % 10 == 0:
            writer.add_scalar("Loss/train", loss.item() * grad_accumulation_steps, epoch * len(train_dataloader) + step)
            print(f"Epoch {epoch + 1}, Step {step}, Loss: {loss.item() * grad_accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1} completed. Avg Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/epoch", avg_loss, epoch)
    
    # Save checkpoint every epoch
    model.save_pretrained(f"checkpoint_epoch_{epoch + 1}")

writer.close()
