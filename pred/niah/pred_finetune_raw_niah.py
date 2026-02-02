import json
import os
import torch
import tqdm
import yaml
from accelerate import Accelerator
from lift.model import load_training_model
from lift.utils import custom_2_hf_training_args, get_pad_token_id, LIFTDataCollator
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer
from typing import Dict, List, Literal


class LMDataset(Dataset):
    def __init__(self, sentences: List[str], tokenizer: PreTrainedTokenizer, seq_length: int = 1024):
        self.data = []
        for sent in sentences:
            input_ids = tokenizer(sent, add_special_tokens=False)["input_ids"]
            self.data.extend([input_ids[i:i+seq_length] for i in range(0, len(input_ids), seq_length)])
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        input_ids = torch.tensor(self.data[index], dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def train_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sentences: List[str],
    training_config: Dict,
    seq_length: int,
):
    training_args = custom_2_hf_training_args(training_config, use_liger_kernel=True, lr_scheduler_type="cosine")
    train_dataset = LMDataset(sentences, tokenizer, seq_length=seq_length)
    trainer = Trainer(model, training_args, data_collator=LIFTDataCollator(tokenizer), train_dataset=train_dataset)
    trainer.train()


@torch.no_grad()
def inference_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    entry: Dict,
):
    messages = [
        {"role": "user", "content": entry["prompt"]}
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=128,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=get_pad_token_id(tokenizer),
    )
    response = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
    return response


def pred_lift_onlylm_niah(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    seq_length: int = 1024,
    num_workers: int = 1,
    worker_id: int = 0,
    overwrite: bool = False,
    adapter: Literal["lora"] = "lora",
    adapter_config: str = "configs/adapter/lora_128.yaml",
    training_config: str = "configs/training/lift_lora.yaml",
):
    eval_accelerator = Accelerator()

    with open(input_path, "r") as f:
        data = [json.loads(line) for line in f]
        all_indices = list(range(len(data)))
    data = data[worker_id::num_workers]
    all_indices = all_indices[worker_id::num_workers]
    if os.path.exists(output_path):
        if overwrite:
            os.remove(output_path)
        else:
            with open(output_path, "r") as f:
                num_resumed = len(f.readlines())
            print(f"Resuming from {num_resumed} entries.")
            data = data[num_resumed:]
            all_indices = all_indices[num_resumed:]

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    num_entries = len(data)
    with open(training_config, "r") as f:
        training_config = yaml.safe_load(f)
    import spacy
    sent_model = spacy.load("en_core_web_sm")

    for i, entry in tqdm.tqdm(zip(all_indices, data), total=num_entries, desc="Evaluating NIAH"):
        model = load_training_model(model_name_or_path, adapter, adapter_config=adapter_config)

        sentences = [sent.text for sent in sent_model(entry["context"]).sents]
        sentences.append(entry["answer"])
        train_field(model, tokenizer, sentences, training_config, seq_length)

        if eval_accelerator.is_main_process:
            response = inference_field(model, tokenizer, entry)
            entry["response"] = response
            entry["context"] = entry["context"][:512]  # Just for identification
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        eval_accelerator.wait_for_everyone()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the base model.")
    parser.add_argument("--input_path", type=str, default="outputs/niah/input.jsonl", help="Path to the input JSONL file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output JSONL file.")
    parser.add_argument("--seq_length", type=int, default=1024, help="Sequence length for training.")
    parser.add_argument("--num_workers", type=int, default=1, help="Number of parallel workers.")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker ID for parallel processing.")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing output file.")
    parser.add_argument("--adapter", type=str, default="lora", help="Type of adapter to use.")
    parser.add_argument("--adapter_config", type=str, default="configs/adapter/lora_128.yaml", help="Path to the adapter config file.")
    parser.add_argument("--training_config", type=str, default="configs/training/lift_lora.yaml", help="Path to the training config file.")
    args = parser.parse_args()
    pred_lift_onlylm_niah(**vars(args))
