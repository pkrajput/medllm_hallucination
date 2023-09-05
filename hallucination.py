import pandas as pd
from selfcheckgpt.modeling_mqag import MQAG
from tqdm import tqdm
import torch

tqdm.pandas()


def is_last_element_minimum(lst):
    return lst[0][-1] == min(lst[0])


def hallucination(reference, ground_truth):

    try:
        questions = mqag_model.generate(
            context=reference, do_sample=True, num_questions=3, seed=42
        )
        hallucinate = 0
        hallucinate_ground = 0
        for j, question_item in enumerate(questions):
            Q = question_item["question"]
            options = [
                question_item["options"][0],
                question_item["options"][1],
                question_item["options"][2],
                "None of the above",
            ]
            probs = mqag_model.answer(
                questions=[{"question": Q, "options": options}], context=reference
            )
            ground_truth_probs = mqag_model.answer(
                questions=[{"question": Q, "options": options}], context=ground_truth
            )
            if is_last_element_minimum(probs) == True:
                hallucinate += 1
            if is_last_element_minimum(ground_truth_probs) == True:
                hallucinate_ground += 1
        if hallucinate > hallucinate_ground:
            return True
        else:
            return False
    except:
        return None


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mqag_model = MQAG(g1_model_type="squad", device=device)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df = pd.read_csv("/workspace/selfcheckgpt/generated_responses_tuned.csv")

    response_columns = ["response1", "response2", "response3", "response4"]

    # df = df.head(1000)

    for i, response_column in tqdm(
        enumerate(response_columns, start=1), desc="Processing"
    ):

        new_column_name = f"hallucination in {i}"
        df[new_column_name] = df.progress_apply(
            lambda row: hallucination(row[response_column], row["output"]), axis=1
        )

    df.to_csv("generated_responses_tuned_hallucination.csv", index=False)

    print("Finished!")
