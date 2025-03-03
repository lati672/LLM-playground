import torch
from transformers import LlamaForCausalLM, LlamaTokenizer
from datasets import load_dataset
import os

# Set cache directory for model and tokenizer
cache_dir = "/scratch/mb26/bp0395/cache"
os.makedirs(cache_dir, exist_ok=True)

# Model and tokenizer name
model_name = "meta-llama/Llama-3B-hf"

# Create directories to save the model and tokenizer
model_save_dir = "/scratch/mb26/bp0395/llama_3b"
os.makedirs(model_save_dir, exist_ok=True)

# Create directory for dataset
dataset_save_dir = "/scratch/mb26/bp0395/wikitext_dataset"
os.makedirs(dataset_save_dir, exist_ok=True)

# Load the tokenizer and model from Hugging Face
print("Loading model and tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
model = LlamaForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)

# Save the model and tokenizer locally
model.save_pretrained(model_save_dir)
tokenizer.save_pretrained(model_save_dir)

# Load the dataset and save it
print("Loading and saving dataset...")
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", cache_dir=cache_dir)

# Save the dataset locally
dataset.save_to_disk(dataset_save_dir)

print("Data and model saved locally.")
