import torch
from selfcheckgpt.modeling_mqag import MQAG
import os
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--output_dir",
    default="/workspace/output",
    type=str,
)

parser.add_argument(
    "--save_json_path",
    default="/workspace/json/",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()
    torch.manual_seed(28)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mqag_model = MQAG(g1_model_type="squad", device=device)

    directory_path = args.output_dir
    json_path = args.save_json_path
    os.makedirs(json_path, exist_ok=True)

    json_dict = {}

    for i, question in tqdm(enumerate(dataset["train"]["input"][:140])):
        file_path = os.path.join(directory_path, f"{i}.txt")

        with open(file_path, "r") as file:
            file_content = file.read()

        json_dict["query"] = question
        json_dict["ChatGPT response"] = file_content
        # print(f"Content of {file_path}:\n{file_content}\n")
        questions = mqag_model.generate(
            context=file_content, do_sample=True, num_questions=3
        )
        for j, question_item in enumerate(questions):
            json_dict[f"question{j}"] = question_item["question"]
            json_dict[f"options{j}"] = [
                question_item["options"][0],
                question_item["options"][1],
                question_item["options"][2],
                question_item["options"][3],
                "None of the above",
            ]

        json_dict["ground_truth_answer"] = dataset["train"]["output"][i]

        with open(json_path + str(i) + ".json", "w") as json_file:
            json.dump(json_dict, json_file)
