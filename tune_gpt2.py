import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments
import wandb
import pandas as pd
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--response_csv",
    default="/workspace/selfcheckgpt/generated_responses.csv",
    type=str,
)

parser.add_argument(
    "--model_save_path",
    default="/workspace/storage/prateek/gpt2",
    type=str,
)

parser.add_argument(
    "--output_dir",
    default="/workspace/output_ft",
    type=str,
)

parser.add_argument(
    "--logging_dir",
    default="/workspace/logs_ft",
    type=str,
)

# Define your custom dataset
class CustomDataset(Dataset):
    def __init__(self, tokenizer, data_df):
        self.tokenizer = tokenizer
        self.data = data_df

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        question = self.data.iloc[idx]["input"]
        output = self.data.iloc[idx]["output"]

        input_text = f"Question: {question}\nAnswer: {output}\n"

        encoding = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            padding="max_length",
            max_length=1024,
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),  # Use input_ids as labels
        }


# Define the training function
def train_function(args, model, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    args = parser.parse_args()

    df = pd.read_csv(args.response_csv)

    df_train = df.head(8000)
    df_eval = df.tail(2000)

    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy="steps",
        eval_steps=500,
        save_steps=500,
        logging_dir=args.logging_dir,
        logging_steps=100,
        num_train_epochs=8,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=5e-5,
        save_total_limit=2,
        report_to="wandb",
        run_name="gpt2_finetuning",
    )

    # Assuming you have df_train defined with 'input' and 'output' columns
    train_dataset = CustomDataset(tokenizer, df_train)
    eval_dataset = CustomDataset(tokenizer, df_eval)

    # Initialize Weights & Biases
    wandb.init(project="gpt2_finetuning")

    # Train the model
    train_function(training_args, model, train_dataset, eval_dataset)

    model.save_pretrained(args.model_saved_path)
