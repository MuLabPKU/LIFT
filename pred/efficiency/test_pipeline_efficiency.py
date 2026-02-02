import accelerate
import json
import os
import torch
import tqdm
import yaml
from lift.model import load_lora_model
from lift.synqa import AsyncOpenAIServer, EverySentenceDataset, EverySentenceStopCallback
from lift.utils import custom_2_hf_training_args, WrapperForIterableDataset, LIFTDataCollator
from transformers import AutoTokenizer, PreTrainedTokenizer, Trainer


def test_with_pipeline(input_length: int, n: int = 0, should_ignore: bool = False):
    accelerator = accelerate.Accelerator()
    # Prepare (this part is not counted in the test time)
    server = AsyncOpenAIServer.from_config("configs/generator/vllm-qwen2.5-backend.yaml")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("models/Meta-Llama-3-8B-Instruct")
    with open("configs/training/lift_lora_fast.yaml", "r") as f:
        training_args = yaml.safe_load(f)
    training_args["per_device_train_batch_size"] *= torch.cuda.device_count()
    training_args = custom_2_hf_training_args(
        training_args,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the real epoch is controlled by `AutoControlStopCallback`
        use_liger_kernel=True,
        logging_strategy="no",
    )
    with open("datasets/loogle/shortdep_qa.jsonl", "r") as f:
        for _ in range(n):
            f.readline()
        context = json.loads(f.readline())["input"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(context_ids) < input_length:
        context_ids *= (input_length // len(context_ids) + 1)
    context_ids = context_ids[:input_length]
    context = tokenizer.decode(context_ids)
    import spacy
    sentence_model = spacy.load("en_core_web_sm")
    sentences = [sent.text for sent in sentence_model(context).sents]
    model = load_lora_model("models/Meta-Llama-3-8B-Instruct", "configs/adapter/lora_128.yaml")
    model.enable_input_require_grads()
    data_collator = LIFTDataCollator(tokenizer)
    # Test
    st_event = torch.cuda.Event(enable_timing=True)
    ed_event = torch.cuda.Event(enable_timing=True)
    st_event.record()
    # NOTE We do not use cache here -- generation is forced to be online.
    dataset = EverySentenceDataset.from_config(
        "configs/generator/vllm-qwen2.5-pipelinetest.yaml",
        tokenizer,
        is_main_process=accelerator.is_main_process,
        server=server,
        sentences=sentences,
    )
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    stop_callback = EverySentenceStopCallback(dataset, training_args.num_train_epochs, total_batch_size)
    dataset = WrapperForIterableDataset(dataset, total_batch_size)
    trainer = Trainer(model, training_args, data_collator=data_collator, train_dataset=dataset, callbacks=[stop_callback])
    trainer.train()
    ed_event.record()
    torch.cuda.synchronize()
    elapsed_time = st_event.elapsed_time(ed_event) / 1000.0
    os.makedirs("outputs/pipeline_efficiency", exist_ok=True)
    if accelerator.is_main_process and not should_ignore:
        with open("outputs/pipeline_efficiency/with_pipeline.jsonl", "a") as f:
            f.write(json.dumps({"input_length": input_length, "time": elapsed_time}) + "\n")


def test_without_pipeline(input_length: int, n: int = 0, should_ignore: bool = False):
    accelerator = accelerate.Accelerator()
    # Prepare (this part is not counted in the test time)
    server = AsyncOpenAIServer.from_config("configs/generator/vllm-qwen2.5-backend.yaml")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained("models/Meta-Llama-3-8B-Instruct")
    with open("configs/training/lift_lora_fast.yaml", "r") as f:
        training_args = yaml.safe_load(f)
    training_args["per_device_train_batch_size"] *= torch.cuda.device_count()
    training_args = custom_2_hf_training_args(
        training_args,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the real epoch is controlled by `AutoControlStopCallback`
        use_liger_kernel=True,
        logging_strategy="no",
    )
    with open("datasets/loogle/shortdep_qa.jsonl", "r") as f:
        for _ in range(n):
            f.readline()
        context = json.loads(f.readline())["input"]
    context_ids = tokenizer(context, add_special_tokens=False)["input_ids"]
    if len(context_ids) < input_length:
        context_ids *= (input_length // len(context_ids) + 1)
    context_ids = context_ids[:input_length]
    context = tokenizer.decode(context_ids)
    import spacy
    sentence_model = spacy.load("en_core_web_sm")
    sentences = [sent.text for sent in sentence_model(context).sents]
    model = load_lora_model("models/Meta-Llama-3-8B-Instruct", "configs/adapter/lora_128.yaml")
    model.enable_input_require_grads()
    data_collator = LIFTDataCollator(tokenizer)
    # Test
    st_event = torch.cuda.Event(enable_timing=True)
    ed_event = torch.cuda.Event(enable_timing=True)
    st_event.record()
    # Remove the cache, so the generation cost is included.
    if os.path.exists("cache_for_pipeline_efficiency.jsonl") and accelerator.is_main_process:
        os.remove("cache_for_pipeline_efficiency.jsonl")
    dataset = EverySentenceDataset.from_config(
        "configs/generator/vllm-qwen2.5-pipelinetest.yaml",
        tokenizer,
        is_main_process=accelerator.is_main_process,
        server=server,
        sentences=sentences,
        cache_path="cache_for_pipeline_efficiency.jsonl",
    )
    if accelerator.is_main_process:
        # Cache all the data, i.e., generate first, and then train
        for _ in dataset:
            pass
    accelerator.wait_for_everyone()

    dataset = EverySentenceDataset.from_config(
        "configs/generator/vllm-qwen2.5-pipelinetest.yaml",
        tokenizer,
        is_main_process=accelerator.is_main_process,
        cache_path="cache_for_pipeline_efficiency.jsonl",
    )
    total_batch_size = training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps
    stop_callback = EverySentenceStopCallback(dataset, training_args.num_train_epochs, total_batch_size)
    dataset = WrapperForIterableDataset(dataset, total_batch_size)
    trainer = Trainer(model, training_args, data_collator=data_collator, train_dataset=dataset, callbacks=[stop_callback])
    trainer.train()
    ed_event.record()
    torch.cuda.synchronize()
    elapsed_time = st_event.elapsed_time(ed_event) / 1000.0
    os.makedirs("outputs/pipeline_efficiency", exist_ok=True)
    if accelerator.is_main_process and not should_ignore:
        with open("outputs/pipeline_efficiency/without_pipeline.jsonl", "a") as f:
            f.write(json.dumps({"input_length": input_length, "time": elapsed_time}) + "\n")


def test_pipeline_efficiency(i: int):
    # Warmup
    test_with_pipeline(1000, should_ignore=True)
    torch.cuda.empty_cache()
    test_without_pipeline(1000, should_ignore=True)
    torch.cuda.empty_cache()

    input_lengths = list(range(1000, 8001, 1000))
    for input_length in tqdm.tqdm(input_lengths):
        test_with_pipeline(input_length)
        torch.cuda.empty_cache()
        test_without_pipeline(input_length)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--i", type=int, default=0)
    args = parser.parse_args()
    test_pipeline_efficiency(args.i)
