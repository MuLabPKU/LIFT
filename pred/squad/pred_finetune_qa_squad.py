import accelerate
import json
import os
import torch
import tqdm
import yaml
from datasets import load_dataset
from lift.model import load_training_model
from lift.synqa import AsyncOpenAIServer, EverySentenceDataset, EverySentenceStopCallback, LLMServerBase
from lift.utils import custom_2_hf_training_args, get_pad_token_id, LIFTDataCollator, WrapperForIterableDataset
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer
from typing import List, Literal, Optional


def train_field(
    model: PreTrainedModel,
    dataset_config: str,
    tokenizer: PreTrainedTokenizer,
    is_main_process: bool,
    server: LLMServerBase,
    sentences: Optional[List[str]],
    cache_path: str,
    training_config: str,
):
    training_dataset = EverySentenceDataset.from_config(
        dataset_config,
        tokenizer,
        is_main_process=is_main_process,
        server=server,
        sentences=sentences,
        cache_path=cache_path,
    )
    training_args = custom_2_hf_training_args(
        training_config,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the real epoch is controlled by `control_callback`
        use_liger_kernel=True,
    )
    control_callback = EverySentenceStopCallback(training_dataset, training_args.num_train_epochs, training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
    trainer = Trainer(model, training_args, data_collator=LIFTDataCollator(tokenizer), train_dataset=WrapperForIterableDataset(training_dataset, training_args.per_device_train_batch_size), callbacks=[control_callback])
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


def pred_lift_onlyqa_squad(
    model_name_or_path: str,
    output_path: str,
    cache_dir: Optional[str] = None,
    server_config: str = "configs/generator/vllm-qwen2.5-backend.yaml",
    dataset_config: str = "configs/generator/vllm-qwen2.5-everysentence-10.yaml",
    dataset_dir: str = "datasets/squad",
    adapter: Literal["lora"] = "lora",
    adapter_config: Optional[str] = "configs/adapter/lora_128.yaml",
    training_config: str = "configs/training/lift_lora.yaml",
):
    eval_accelerator = accelerate.Accelerator()
    with open(training_config, "r") as f:
        training_config = yaml.safe_load(f)
    training_config["per_device_train_batch_size"] *= torch.cuda.device_count()  # due to "split_batch==True"

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    llm_server = AsyncOpenAIServer.from_config(server_config)
    import spacy
    sent_model = spacy.load("en_core_web_sm")

    dataset = load_dataset(dataset_dir, split="train")
    dataset = [dataset[i] for i in range(1000)]
    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            num_resumed = len(f.readlines())
        dataset = [dataset[i] for i in range(num_resumed, len(dataset))]

    for i, entry in enumerate(tqdm.tqdm(dataset, desc="Evaluate")):
        model = load_training_model(model_name_or_path, adapter, adapter_config=adapter_config)
        sentences = [s.text for s in sent_model(entry["context"]).sents]
        cache_path = None if cache_dir is None else os.path.join(cache_dir, f"{i}.jsonl")
        train_field(model, dataset_config, tokenizer, eval_accelerator.is_main_process, llm_server, sentences, cache_path, training_config)
        model.eval()
        if eval_accelerator.is_main_process:
            response = inference_field(model, tokenizer, entry["question"])
            entry["response"] = response
            with open(output_path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        eval_accelerator.wait_for_everyone()
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True, help="Model name or path.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output predictions.")
    parser.add_argument("--cache_dir", type=str, default=None, help="Directory to cache the prepared dataset.")
    parser.add_argument("--server_config", type=str, default="configs/generator/vllm-qwen2.5-backend.yaml", help="LLM server configuration file.")
    parser.add_argument("--dataset_config", type=str, default="configs/generator/vllm-qwen2.5-everysentence-10.yaml", help="Dataset configuration file.")
    parser.add_argument("--dataset_dir", type=str, default="datasets/squad", help="Dataset directory.")
    parser.add_argument("--adapter", type=str, choices=["lora"], default="lora", help="Adapter type.")
    parser.add_argument("--adapter_config", type=str, default="configs/adapter/lora_128.yaml", help="Adapter configuration file.")
    parser.add_argument("--training_config", type=str, default="configs/training/lift_lora.yaml", help="Training configuration file.")
    args = parser.parse_args()
    pred_lift_onlyqa_squad(**vars(args))
