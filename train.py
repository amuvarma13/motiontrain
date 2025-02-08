from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb

dsn = "amuvarma/humanml3d-flat-train-padded-2"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

num_add_tokens = 1000

model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)

wandb.init(project="motiontrainlr", name="5e5")


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=10,
    output_dir="./output",
    per_device_train_batch_size=16,
    logging_steps=10,
    bf16=True,
    # output_dir=f"./{base_repo_id}",
    # fsdp="auto_wrap",
    report_to="wandb",
    save_steps=8374,
    learning_rate=2e-4,
    
    # save_steps=save_steps,
    remove_unused_columns=True,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
)

trainer.train()
