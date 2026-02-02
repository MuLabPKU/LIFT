import json
import os
import torch
import tqdm
from lift.utils import get_pad_token_id, is_instruct
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Literal, Optional


# The original prompt given in the config file of LooGLE
LOOGLE_PROMPT_WITH_ICL = "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: "


def pred_icl_loogle(
    subtask: Literal["shortdep_qa", "longdep_qa"],
    model_name_or_path: str,
    output_path: str,
    model_max_length: Optional[int] = None,
    loogle_dir: str = "datasets/loogle",
    loogle_config_dir: str = "datasets/loogle_config",
    disable_system_prompt: bool = False,
    overwrite: bool = False,
    disable_reasoning: bool = False,
    num_workers: int = 1,
    worker_id: int = 0,
):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(os.path.join(loogle_dir, subtask + ".jsonl"), "r") as f:
        dataset = [json.loads(l) for l in f]
    with open(os.path.join(loogle_config_dir, "task2maxlen.json"), "r") as f:
        max_new_tokens = json.load(f)[subtask]

    dataset = dataset[worker_id::num_workers]  # Handle multi-worker setting
    if os.path.exists(output_path):  # Handle resuming
        if overwrite:
            os.remove(output_path)
        else:
            with open(output_path, "r") as f:
                num_resumed = len(f.readlines())
            dataset = dataset[num_resumed:]
    max_input_length = None if model_max_length is None else model_max_length - max_new_tokens

    for entry in tqdm.tqdm(dataset, desc=subtask):
        qa_pairs = eval(entry["qa_pairs"])
        for qa in qa_pairs:
            if is_instruct(tokenizer):
                if disable_system_prompt:
                    message = [
                        {"role": "user", "content": LOOGLE_PROMPT_WITH_ICL.format(input=entry["input"], Q=qa["Q"])},
                    ]
                else:
                    message = [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": LOOGLE_PROMPT_WITH_ICL.format(input=entry["input"], Q=qa["Q"])},
                    ]
                if disable_reasoning:
                    # for qwen models
                    input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")
                else:
                    input_ids = tokenizer.apply_chat_template(message, add_generation_prompt=True, return_tensors="pt")
            else:
                prompt = LOOGLE_PROMPT_WITH_ICL.format(input=entry["input"], Q=qa["Q"])
                input_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"]
            if max_input_length is not None and input_ids.shape[-1] > max_input_length:
                input_ids = torch.concat((input_ids[..., :max_input_length//2], input_ids[..., -max_input_length//2:]), dim=-1)
            input_ids = input_ids.to(model.device)
            attention_mask = torch.ones_like(input_ids)
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,  # It is recommended in LooGLE to use greedy decoding. `transformers` may raise some warnings (some arguments like `top_p` will be ignored) - just ignore these warnings.
                max_new_tokens=max_new_tokens,
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=get_pad_token_id(tokenizer),
            )
            response = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
            qa["P"] = response
        entry["qa_pairs"] = qa_pairs
        entry["input"] = entry["input"][:200]  # Only used for identification
        with open(output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--subtask", choices=["shortdep_qa", "longdep_qa"], required=True)
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--model_max_length", type=int, default=None)
    parser.add_argument("--loogle_dir", default="datasets/loogle")
    parser.add_argument("--loogle_config_dir", default="datasets/loogle_config")
    parser.add_argument("--disable_system_prompt", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--disable_reasoning", action="store_true")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    args = parser.parse_args()
    pred_icl_loogle(**vars(args))
