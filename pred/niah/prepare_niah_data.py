"""
Generate NIAH dataset for test.
"""
import argparse
import json
import math
import os
import tqdm
import yaml
from dataclasses import dataclass
from glob import glob
from random import shuffle
from transformers import AutoTokenizer, PreTrainedTokenizer
from typing import List


@dataclass
class NIAHArguments:
    tokenizer_name_or_path: str
    num_samples_per_case: int
    lengths: List[int]
    depths: List[int]
    haystack_dir: str
    needle: str
    prompt: str
    keywords: List[str]  # Used for evaluation


def generate_sample(tokenizer: PreTrainedTokenizer, context: str, context_length: int, needle_depth: int, needle: str, prompt: str):
    num_words = len(context.split())
    if context_length > num_words:
        context = context * math.ceil(context_length / num_words)
    description = "There is an important infomation hidden in the following context. Find the information and memorize it. I will quiz you about the important information there.\n"

    description_input_ids = tokenizer.encode(description, add_special_tokens=False)
    needle_input_ids = tokenizer.encode(needle, add_special_tokens=False)

    description_length = len(description_input_ids)
    needle_length = len(needle_input_ids)

    # must leave room for information and prompt
    minimum_pos = description_length
    maximum_pos = context_length - needle_length - 1
    if minimum_pos > context_length or maximum_pos < 0:
        raise ValueError(f"The length {context_length} is too small. Please increase interval!")

    needle_pos = minimum_pos + round((maximum_pos - minimum_pos) * needle_depth / 100)
    context_input_ids = tokenizer.encode(context, max_length=context_length - description_length - needle_length, truncation=True, add_special_tokens=False)
    context_ids = description_input_ids + context_input_ids[:needle_pos] + needle_input_ids + context_input_ids[needle_pos:]
    context_return = tokenizer.decode(context_ids)

    return context_return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--output_path', required=True)
    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        cfg = NIAHArguments(**yaml.safe_load(f))

    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer_name_or_path)

    all_contexts = []
    haystack_files = [file for file in glob(f"{cfg.haystack_dir}/*.txt")]
    max_length = max(cfg.lengths)
    for i in range(cfg.num_samples_per_case):
        num_tokens = 0
        all_contexts.append("")
        shuffle(haystack_files)
        for file in haystack_files:
            with open(file, 'r') as f:
                content = f.read()
            num_tokens += len(tokenizer(content, add_special_tokens=False))
            all_contexts[i] += content
            if num_tokens > max_length:
                break

    output_dir = os.path.split(args.output_path)[0]
    if len(output_dir) > 0:
        os.makedirs(output_dir, exist_ok=True)
    if os.path.exists(args.output_path):
        os.remove(args.output_path)

    pbar = tqdm.tqdm(total=len(cfg.lengths) * len(cfg.depths) * cfg.num_samples_per_case, desc="Constructing data")
    for i in range(cfg.num_samples_per_case):
        for length in cfg.lengths:
            for depth in cfg.depths:
                context = generate_sample(
                    tokenizer=tokenizer,
                    context=all_contexts[i],
                    context_length=length, 
                    needle_depth=depth,
                    needle=cfg.needle,
                    prompt=cfg.prompt,
                )
                with open(args.output_path, 'a') as f:
                    f.write(json.dumps({'context': context, 'prompt': cfg.prompt, 'answer': cfg.needle, 'length': length, 'depth': depth}) + '\n')
                pbar.update()
    pbar.close()


if __name__ == '__main__':
    main()
