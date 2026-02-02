import json
import os
import torch
import tqdm
import yaml
from accelerate import Accelerator
from lift.model import load_training_model
from lift.synqa import AsyncOpenAIServer, EverySentenceDataset, EverySentenceStopCallback, LLMServerBase
from lift.utils import custom_2_hf_training_args, get_pad_token_id, LIFTDataCollator, WrapperForIterableDataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer
from typing import Dict, List, Literal, Optional


def train_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    cache_path: str,
    dataset_config: str,
    server: LLMServerBase,
    sentences: Optional[List[str]],
    training_config: Dict,
):
    training_args = custom_2_hf_training_args(
        training_config,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the real epoch is controlled by `stop_callback`
        use_liger_kernel=True,
    )
    train_dataset = EverySentenceDataset.from_config(
        dataset_config,
        tokenizer,
        server=server,
        sentences=sentences,
        cache_path=cache_path,
    )
    stop_callback = EverySentenceStopCallback(train_dataset, training_args.num_train_epochs, training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    trainer = Trainer(model, training_args, data_collator=LIFTDataCollator(tokenizer), train_dataset=WrapperForIterableDataset(train_dataset, training_config["per_device_train_batch_size"]), callbacks=[stop_callback])
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


def pred_lift_niah(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    synqa_dir: str = "outputs/niah/synqa",
    server_config: str = "configs/generator/vllm-qwen3-backend.yaml",
    dataset_config: str = "configs/generator/vllm-qwen3-niah.yaml",
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
    server = AsyncOpenAIServer.from_config(server_config)
    import spacy
    sent_model = spacy.load("en_core_web_sm")
    num_entries = len(data)
    with open(training_config, "r") as f:
        training_config = yaml.safe_load(f)
    training_config["per_device_train_batch_size"] *= torch.cuda.device_count()  # Due to split_batch=True

    for i, entry in tqdm.tqdm(zip(all_indices, data), total=num_entries, desc="Evaluating NIAH"):
        cache_path = os.path.join(synqa_dir, f"{i}.jsonl")
        model = load_training_model(model_name_or_path, adapter, adapter_config=adapter_config)

        if os.path.exists(cache_path):
            sentences = None
        else:
            sentences = [sent.text for sent in sent_model(entry["context"]).sents]
        train_field(model, tokenizer, cache_path, dataset_config, server, sentences, training_config)

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
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--synqa_dir", type=str, default="outputs/niah/synqa")
    parser.add_argument("--server_config", type=str, default="configs/generator/vllm-qwen3-backend.yaml")
    parser.add_argument("--dataset_config", type=str, default="configs/generator/vllm-qwen3-niah.yaml")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--worker_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--adapter", type=str, choices=["lora"], default="lora")
    parser.add_argument("--adapter_config", type=str, default="configs/adapter/lora_128.yaml")
    parser.add_argument("--training_config", type=str, default="configs/training/lift_lora.yaml")
    args = parser.parse_args()
    pred_lift_niah(**vars(args))
