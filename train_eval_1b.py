from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

dsn = "amuvarma/humanml3d-flat-train-padded-2"
eval_dsn = "amuvarma/humanml3d-flat-train-padded-dedup-2"
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)

num_add_tokens = 1000
model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

dataset = load_dataset(dsn, split="train")
train_dataset = dataset.shuffle(seed=42)

eval_dataset = split_dataset["test"]

wandb.init(project="motiontrain-full", name="4e2-1b-eval")

training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    logging_steps=10,
    evaluation_strategy="steps",  # Evaluate every `eval_steps` during training
    eval_steps=1000,               # Adjust this value as needed for your use case
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=8374,
    report_to="wandb",
    remove_unused_columns=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()
