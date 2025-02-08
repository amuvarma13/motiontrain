from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import wandb

# Define your dataset and model details
dsn = "amuvarma/humanml3d-flat-train-padded-2"
model_name = "meta-llama/Llama-3.2-1B"
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add extra tokens to the model's vocabulary
num_add_tokens = 1000
model.resize_token_embeddings(model.config.vocab_size + num_add_tokens)

tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load and shuffle your dataset
dataset = load_dataset(dsn, split="train")
dataset = dataset.shuffle(seed=42)

# Split the dataset into training and evaluation sets (e.g., 90% train, 10% eval)
split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Initialize Weights & Biases for logging
wandb.init(project="motiontrain-debug", name="4e2-1b")

# Set up your training arguments
training_args = TrainingArguments(
    output_dir="./output",
    overwrite_output_dir=True,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    logging_steps=10,
    evaluation_strategy="steps",  # Evaluate every `eval_steps` during training
    eval_steps=20,               # Adjust this value as needed for your use case
    bf16=True,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    save_steps=8374,
    report_to="wandb",
    remove_unused_columns=True
)

# Initialize the Trainer with both the training and evaluation datasets
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
)

# Start training
trainer.train()
