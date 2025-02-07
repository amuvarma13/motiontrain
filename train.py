from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
import torch
from datasets import load_dataset

dsn = "amuvarma/humanml3d-flat-train-padded"
model_name = "meta-llama/Llama-3.2-3B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name)

tokenizer = AutoTokenizer.from_pretrained(dsn)

dataset = load_dataset(dsn, split="train")

training_args = TrainingArguments(
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    logging_steps=1,
    fp16=True,
    # output_dir=f"./{base_repo_id}",
    fsdp="auto_wrap",
    report_to="wandb",
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
