import os
import torch
from selfcheckgpt.modeling_mqag import MQAG
import json
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    "--json_dir",
    default="/workspace/json/",
    type=str,
)

parser.add_argument(
    "--mqag_model_type",
    default="squad",
    type=str,
)

if __name__ == "__main__":
    args = parser.parse_args()

    torch.manual_seed(28)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mqag_model = MQAG(g1_model_type=args.mqag_model_type, device=device)
    json_directory = args.json_dir
    processed_data = []
    mcq_probabilities = []

    def is_last_element_minimum(lst):
        return lst[-1] == min(lst)

    uncertain = 0
    num_questions = 0
    uncertain_ground_truth = 0
    for filename in tqdm(os.listdir(json_directory)):
        if filename.endswith(".json"):
            index = filename.split(".")[0]
            json_filepath = os.path.join(json_directory, filename)

            with open(json_filepath, "r") as json_file:
                data = json.load(json_file)
                keys_list = list(data.keys())
                values_list = list(data.values())

                final_storage = []
                for item, value in zip(keys_list, values_list):
                    storage_dict = {}

                    if item.find("question") != -1:
                        num_questions += 1
                        # print(data['options'+item[-1]])
                        remove_least_answer = data["options" + item[-1]].pop(3)
                        questions = [
                            {"question": value, "options": data["options" + item[-1]]}
                        ]
                        # print(value)
                        probs = mqag_model.answer(
                            questions=questions, context=data["ChatGPT response"]
                        )
                        if is_last_element_minimum(probs[0]) == True:
                            uncertain += 1
                        # print(probs[0])
                        storage_dict[value] = probs[0]

                        ground_truth_probs = mqag_model.answer(
                            questions=questions, context=data["ground_truth_answer"]
                        )
                        if is_last_element_minimum(ground_truth_probs[0]) == True:
                            uncertain_ground_truth += 1
                        # print(probs[0])
                        # storage_dict[value] = probs[0]

                    if len(storage_dict) != 0:
                        final_storage.append(storage_dict)
            mcq_probabilities.append({str(index): final_storage})

    print(f"percentage of uncertain questions: {(uncertain/num_questions)*100}%")
    print(
        f"percentage of uncertain questions with known response: {(uncertain_ground_truth/num_questions)*100}%"
    )
