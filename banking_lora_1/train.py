import torch
import deepspeed
import json
import os
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW, get_cosine_schedule_with_warmup
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from deepspeed.ops.adam import DeepSpeedCPUAdam

deepspeed.init_distributed()

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(model, config)

dataset = load_dataset("parquet", data_files="data.parquet")

def format_example(example):
    instruction = example["instruction"]
    intent = example["intent"]
    response = example["response"]
    return {
        "input": f"Instruction: {instruction}\n",
        "output": f"Intent: {intent}\nResponse: {response}"
    }

dataset = dataset.map(format_example)

def tokenize_function(examples):
    inputs = tokenizer(examples["input"], truncation=True, padding="max_length", max_length=512)
    outputs = tokenizer(examples["output"], truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = outputs["input_ids"] 
    return inputs

tokenizer.pad_token = tokenizer.eos_token
tokenized_dataset = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_dataset["train"]
sampler = DistributedSampler(train_dataset)
BATCH_SIZE = 16 
grad_accumulation_steps = 4
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)

with open("ds_config.json", "r") as f:
    ds_config = json.load(f)

if "bf16" in ds_config:
    ds_config["bf16"]["enabled"] = True
    if "fp16" in ds_config:
        ds_config["fp16"]["enabled"] = False
else:
    ds_config["fp16"]["enabled"] = True

ds_config["gradient_accumulation_steps"] = grad_accumulation_steps

learning_rate = 2e-5
optimizer = DeepSpeedCPUAdam(model.parameters(), lr=learning_rate)

model, optimizer, _, _ = deepspeed.initialize(
    model=model, 
    optimizer=optimizer, 
    config=ds_config
)

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
writer = SummaryWriter(log_dir=log_dir)

epochs = 3
global_step = 0
for epoch in range(epochs):
    model.train()
    sampler.set_epoch(epoch)
    total_loss = 0.0
    for step, batch in enumerate(train_dataloader):
        inputs = torch.stack(batch["input_ids"]).to(model.device)
        labels = torch.stack(batch["labels"]).to(model.device)
        
        outputs = model(input_ids=inputs, labels=labels)
        loss = outputs.loss / grad_accumulation_steps  
        
        model.backward(loss)
        
        if (step + 1) % grad_accumulation_steps == 0:
            model.step()
            global_step += 1
            
        total_loss += loss.item() * grad_accumulation_steps
        
        if step % 10 == 0:
            writer.add_scalar("Loss/train", loss.item() * grad_accumulation_steps, epoch * len(train_dataloader) + step)
            print(f"Epoch {epoch+1}, Step {step}, Loss: {loss.item() * grad_accumulation_steps:.4f}")
    
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")
    writer.add_scalar("Loss/epoch", avg_loss, epoch)
    
    model.save_pretrained(f"checkpoint_epoch_{epoch+1}")

writer.close()
