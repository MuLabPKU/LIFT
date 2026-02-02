import argparse
import json
import numpy as np
import os
import yaml


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", required=True)
parser.add_argument("-n", "--num_workers", required=True, type=int)
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config, "r") as f:
    niah_config = yaml.safe_load(f)

all_wrong_ids = []
all_scores = {(l, d): [] for l in niah_config["lengths"] for d in niah_config["depths"]}
for i in range(args.num_workers):
    with open(os.path.join(args.input_dir, f"niah_{i}.jsonl"), "r") as f:
        for j, l in enumerate(f):
            entry = json.loads(l)
            response = entry["response"].lower()
            score = all(k in response for k in niah_config["keywords"])
            if not score:
                print(f"Wrong: worker {i} entry {j} (global {j * args.num_workers + i})")
                print("\t" + entry["response"])
                all_wrong_ids.append(j * args.num_workers + i)
            all_scores[(entry["length"], entry["depth"])].append(score)

print("\t\t".join([""] + [f"{d:.2f}" for d in niah_config["depths"]]))
for l in niah_config["lengths"]:
    print("\t\t".join([str(l)] + [f"{np.mean(all_scores[(l, d)]):.3f}" for d in niah_config["depths"]]))


print(all_wrong_ids)