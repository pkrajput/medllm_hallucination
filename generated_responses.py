from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import warnings
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--mode",
    default="train",
    type=str,
)

parser.add_argument(
    "--tuned_model_path",
    default="/workspace/storage/prateek/gpt2",
    type=str,
)

parser.add_argument(
    "--cut",
    default=None,
    type=str,
)

if __name__ == "__main__":
    
    args = parser.parse_args()

    # Check if a GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the GPT-2 model and tokenizer
    model_name = "gpt2"
    if args.mode == "train":
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        output_csv_file = "generated_responses.csv"
    elif args.mode == "eval":
        model_path = args.tuned_model_path
        model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
        output_csv_file = "generated_responses_tuned.csv"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Set the EOS token as the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Load the dataset
    dataset = load_dataset("medalpaca/medical_meadow_wikidoc")
    df = pd.DataFrame(dataset["train"])
    
    if args.cut == "small":
        df = df.head(1000)

    # Define beam search parameters
    num_responses = 4
    max_length = 1024  # Set an appropriate maximum length
    num_beams = 4

    # Generate responses for each question and add to the DataFrame
    response_columns = [f"response{i+1}" for i in range(num_responses)]

    # Ignore warnings
    warnings.filterwarnings("ignore")
    start_time = time.time()

    for i, row in tqdm(df.iterrows(), total=len(df)):
        try:
            question = row["input"]
            context = row["output"]
            responses = []
            
            for _ in range(num_responses):
                input_text = f"Context: {context}\n\nQuestion: {question}\n\nAnswer: "
                input_ids = tokenizer.encode(
                    input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length
                ).to(device)
                attention_mask = torch.ones_like(input_ids).to(device)

                # Generate responses using beam search
                output = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                    num_return_sequences=num_responses,
                    no_repeat_ngram_size=2,
                )
                
                # Decode and add responses
                for j in range(num_responses):
                    response = tokenizer.decode(output[j], skip_special_tokens=True)
                    answer_start = response.find("Answer: ") + len("Answer: ")
                    responses.append(response[answer_start:])

            for j, col_name in enumerate(response_columns):
                df.at[i, col_name] = responses[j]

        except RuntimeError as e:
            print(f"Skipped due to CUDA error: {e}")
            continue

    # Calculate and display estimated time of completion
    end_time = time.time()
    elapsed_time = end_time - start_time
    remaining_time = (elapsed_time / (i + 1)) * (len(df) - i - 1)
    print(f"Estimated time of completion: {remaining_time:.2f} seconds")

    # Reset warning filter
    warnings.resetwarnings()

    df.to_csv(output_csv_file, index=False)
    print(f"Saved generated responses to {output_csv_file}")

