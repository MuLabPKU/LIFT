import json
import numpy as np
import os
import tqdm
from collections import defaultdict
from lift.synqa import AsyncOpenAIServer
from typing import Optional


USER_MSG_TEMPLATE = """question: {question}
groundtruth = {groundtruth}
predict_answer = {response}
"""
SYSTEM_MSG_TEMPLATE = """Given one question, there is a groundtruth and a predict_answer. Please decide whether they are the same or not in semantic. Please only output 'True' or 'False' ."""


def construct_message(question: str, groundtruth: str, response: str, system_role_name: str = "developer"):
    return [
        {"role": system_role_name, "content": SYSTEM_MSG_TEMPLATE},
        {"role": "user", "content": USER_MSG_TEMPLATE.format(question=question, groundtruth=groundtruth, response=response)},
    ]


def loogle_eval(
    pred_file: str,
    judge_config: str,
    overwrite: bool = False,
    score_file: Optional[str] = None,
    system_role_name: str = "developer",
):
    # Preprocess arguments
    if not pred_file.endswith(".jsonl"):
        raise ValueError(f"Expect `pred_file` to be a JSONL file but receive `{pred_file}`.")
    if score_file is None:
        score_file = pred_file.rsplit(".", maxsplit=1)[0] + ".score.jsonl"
        print(f"The scores will be saved in `{score_file}`.")
    elif not score_file.endswith(".jsonl"):
        raise ValueError(f"Expect `score_file` to be a JSONL file but receive `{score_file}`.")
    # Variables
    num_valid = 0
    error_ids = []
    all_scores = []
    all_type_scores = defaultdict(list)
    # Setup the server
    server = AsyncOpenAIServer.from_config(judge_config)
    # Load the prediction file
    with open(pred_file, "r", encoding="utf-8") as f:
        all_entries = [json.loads(l) for l in f]
    # Handle resuming
    num_resumed = 0
    if os.path.exists(score_file):
        if overwrite:
            os.remove(score_file)
        else:
            with open(score_file, "r") as f:
                for i, l in enumerate(f):
                    num_resumed += 1
                    entry = json.loads(l)
                    if "qa_pairs" in entry:
                        for qa in entry["qa_pairs"]:
                            all_scores.append(qa["score"])
                            if "type" in qa:
                                all_type_scores[qa["type"]].append(qa["score"])
                        num_valid += 1
                    elif "error" in entry:
                        error_ids.append(i)
    # Judge
    for i, entry in enumerate(tqdm.tqdm(all_entries[num_resumed:], desc="LooGLE eval"), start=num_resumed):
        if "qa_pairs" in entry:
            num_valid += 1
            all_messages = [construct_message(qa["Q"], qa["A"], qa.get("P", qa.get("pred", None)), system_role_name=system_role_name) for qa in entry["qa_pairs"]]
            for qa, response in zip(entry["qa_pairs"], server.batch_sampling(all_messages)):
                qa["score"] = "true" in response.lower()
                all_scores.append(qa["score"])
                if "type" in qa:
                    all_type_scores[qa["type"]].append(qa["score"])
        elif "error" in entry:
            error_ids.append(i)
        with open(score_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
    # Average score
    print(f"Find {len(all_entries)} entries in total.")
    print(f"{num_valid} entries are valid.")
    print("The error entries:", *error_ids)
    print("Average score =", np.mean(all_scores))
    if len(all_type_scores) > 0:
        for k, v in all_type_scores.items():
            print(f"\t{k}: {np.mean(v)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--judge_config", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--score_file")
    parser.add_argument("--system_role_name", default="developer")
    args = parser.parse_args()
    loogle_eval(**vars(args))
