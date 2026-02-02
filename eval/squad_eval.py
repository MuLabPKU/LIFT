import json
import numpy as np
import re
import tqdm
from collections import Counter
from lift.synqa import AsyncOpenAIServer


def get_tokens(text):
    """Lowercases, removes punctuation, and splits text into words."""
    lowercased_text = text.lower()
    # Remove punctuation using regex
    clean_text = re.sub(r'[^\w\s]', '', lowercased_text)
    return clean_text.split()


def calculate_f1(reference, candidate):
    ref_tokens = get_tokens(reference)
    cand_tokens = get_tokens(candidate)
    # Count frequencies of each word
    ref_counts = Counter(ref_tokens)
    cand_counts = Counter(cand_tokens)
    # Calculate common words (intersection)
    common_counts = ref_counts & cand_counts
    num_same = sum(common_counts.values())
    # Handle the case of empty tokens
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 1.0 if ref_counts == cand_counts else 0.0
    precision = num_same / len(cand_tokens)
    recall = num_same / len(ref_tokens)
    # Calculate F1 score
    if (precision + recall) == 0:
        return 0.0
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1


def squad_eval(pred_file: str, score_file: str, judge_config: str = "configs/judge/gpt4-judge.yaml"):
    with open(pred_file, "r") as f:
        data = [json.loads(line) for line in f]
    # Compute F1 scores
    print("=== Computing F1 scores ===")
    for entry in tqdm.tqdm(data, desc="Computing F1"):
        entry["f1"] = max(calculate_f1(answer, entry["response"]) for answer in entry["answers"]["text"])
    print("Avg F1:", np.mean([entry["f1"] for entry in data]))
    # Compute LLM as judge scores
    print("=== Computing LLM as judge scores ===")
    server = AsyncOpenAIServer.from_config(judge_config)
    SYSTEM_MSG = """Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False'."""
    all_messages = []
    for entry in data:
        for answer in entry["answers"]["text"]:
            messages = [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": f"Question: {entry['question']}\nGroundtruth: {answer}\nPredict_answer: {entry['response']}"}
            ]
            all_messages.append(messages)
    responses_iter = server.batch_sampling(all_messages)
    for entry in tqdm.tqdm(data, desc="Computing GPT-4 scores"):
        all_scores = []
        for answer in entry["answers"]["text"]:
            response = next(responses_iter)
            all_scores.append("true" in response.lower())
        entry["gpt4-score"] = any(all_scores)
    print("Avg GPT-4 Score:", np.mean([entry["gpt4-score"] for entry in data]))
    # Save results
    with open(score_file, "w") as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--pred_file", type=str, required=True, help="Path to the prediction file.")
    parser.add_argument("-o", "--score_file", type=str, required=True, help="Path to save the score file.")
    parser.add_argument("--judge_config", type=str, default="configs/judge/gpt4-judge.yaml", help="Path to the judge config file.")
    args = parser.parse_args()

    squad_eval(**vars(args))