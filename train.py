from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset
import wandb

dsn = "amuvarma/humanml3d-flat-train-padded"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

num_add_tokens = 1000

model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(dsn, split="train")

wandb.init(project="motiontrain", name="r0")


training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=1,
    output_dir="./output",
    per_device_train_batch_size=4,
    logging_steps=1,
    fp16=True,
    # output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb",
    save_steps=10000,
    
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
