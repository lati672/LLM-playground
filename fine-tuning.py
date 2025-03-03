import torch
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling
import os

# Set cache directory for model and tokenizer
cache_dir = "/scratch/mb26/bp0395/cache"

# Load the tokenizer and model from disk (no internet required)
model_dir = "/scratch/mb26/bp0395/llama_3b"
os.makedirs(model_dir, exist_ok=True)  # Create model directory if it doesn't exist
tokenizer = LlamaTokenizer.from_pretrained(model_dir, cache_dir=cache_dir)
model = LlamaForCausalLM.from_pretrained(model_dir, cache_dir=cache_dir)

# Load the dataset from disk (no internet required)
dataset_dir = "/scratch/mb26/bp0395/wikitext_dataset"
os.makedirs(dataset_dir, exist_ok=True)  # Ensure dataset directory exists
dataset = load_from_disk(dataset_dir)

# Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples['text'], return_tensors="pt", padding=True, truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Set up data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # LLaMA is a causal language model, not a masked language model
)

# Define training arguments
output_dir = "/scratch/mb26/bp0395/llama-finetuned"
os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists

logging_dir = '/scratch/mb26/bp0395/logs'
os.makedirs(logging_dir, exist_ok=True)  # Ensure logging directory exists

training_args = TrainingArguments(
    output_dir=output_dir,  # Save model checkpoints
    num_train_epochs=3,  # Number of epochs
    per_device_train_batch_size=1,  # Adjust depending on your GPU memory
    save_steps=10_000,  # Save checkpoint every 10,000 steps
    save_total_limit=2,  # Keep only 2 checkpoints
    logging_steps=500,  # Log every 500 steps
    logging_dir=logging_dir,  # Directory for logs
    report_to="tensorboard",  # Enable tensorboard logging
    fp16=True,  # Enable mixed precision training if available
    gradient_accumulation_steps=16,  # Accumulate gradients to simulate a larger batch size
    cache_dir=cache_dir,  # Set cache directory for training args
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    data_collator=data_collator,
)

# Start the training process
trainer.train()

# Save the fine-tuned model
trainer.save_model(output_dir)
