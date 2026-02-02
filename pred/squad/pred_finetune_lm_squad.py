"""
NOTE Launch with python (do not use accelerate launch)!
"""
import json
import os
import torch
import tqdm
import yaml
from datasets import load_dataset
from lift.model import load_training_model
from lift.utils import custom_2_hf_training_args, get_pad_token_id
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer
from typing import Any, Dict, Literal, Optional


class SquadTrainingDataset(Dataset):
    def __init__(self, context: str, tokenizer: PreTrainedTokenizer, model_max_length: Optional[int] = None):
        context_ids = tokenizer(context, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        self.data = context_ids
        if model_max_length is not None and len(self.data) > model_max_length:
            raise ValueError(f"Context length ({len(self.data)}) exceeds model max length ({model_max_length}).")

    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        input_ids = torch.tensor(self.data, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids,
        }


def train_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    training_config: Dict[str, Any],
    model_max_length: Optional[int],
):
    training_args = custom_2_hf_training_args(training_config, use_liger_kernel=True)
    train_dataset = SquadTrainingDataset(context, tokenizer, model_max_length=model_max_length)
    trainer = Trainer(model, training_args, train_dataset=train_dataset)
    trainer.train()


@torch.no_grad()
def inference_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    question: str,
):
    messages = [
        {"role": "user", "content": question.strip()},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=128,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer),
    )
    answer = tokenizer.decode(outputs[0, input_ids.shape[-1]:], skip_special_tokens=True)
    return answer


def pred_lift_onlylm_squad(
    model_name_or_path: str,
    output_path: str,
    data_path: str = "datasets/squad",
    training_config: str = "configs/training/lift_lora.yaml",
    adapter: Literal["lora"] = "lora",
    adapter_config: Optional[str] = "configs/adapter/lora_128.yaml",
    model_max_length: Optional[int] = None,
    num_workers: int = 1,
    worker_id: int = 0,
    overwrite: bool = False,
):
    # Load the SQuAD dataset
    # NOTE: we only use the first 1000 examples for faster evaluation
    dataset = load_dataset(data_path, split="train")
    dataset = [dataset[i] for i in range(worker_id, 1000, num_workers)]  # Handle multi-worker
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            with open(output_path, "r") as f:
                num_resumed = len(f.readlines())
            dataset = dataset[num_resumed:]  # Resume from existing output

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    with open(training_config, "r") as f:
        training_config = yaml.safe_load(f)

    for entry in tqdm.tqdm(dataset, desc="Evaluating"):
        model = load_training_model(model_name_or_path, adapter, adapter_config=adapter_config)
        train_field(model, tokenizer, entry["context"], training_config, model_max_length)
        response = inference_field(model, tokenizer, entry["question"])
        entry["response"] = response
        with open(output_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="datasets/squad")
    parser.add_argument("--training_config", type=str, default="configs/training/lift_lora_squad_lm.yaml")
    parser.add_argument("--adapter", type=str, choices=["lora"], default="lora")
    parser.add_argument("--adapter_config", type=str, default="configs/adapter/lora_128.yaml")
    parser.add_argument("--model_max_length", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    pred_lift_onlylm_squad(**vars(args))
