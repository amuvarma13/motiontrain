from transformers import AutoModelForCausalLM, TrainingArguments, Trainer
import torch
from datasets import load_dataset

model_name = "meta-llama/Llama-3.2-3B-Instruct"

model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.resize_token_embeddings(128257)

dataset_name = "UjjD/kafka-qa-dataset"
dataset = load_dataset(dataset_name, split="train")

print(dataset)

training_args = TrainingArguments(
    output_dir="./llama-finetuned-kafka",  # Directory for model checkpoints
    overwrite_output_dir=True,
    per_device_train_batch_size=1,  # Batch size of 1
    logging_steps=1,
    fp16=True,
    remove_unused_columns=True,

    report_to="none",  # Disable Weights & Biases logging
)

trainer = Trainer(
    model = model, 
    args = training_args, 
    train_dataset = dataset
)

trainer.train()