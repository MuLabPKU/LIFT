import json
import os
import time
import torch
import tqdm
import yaml
from lift.model import load_training_model
from lift.synqa import AsyncOpenAIServer, EverySentenceDataset, EverySentenceStopCallback
from lift.utils import custom_2_hf_training_args, get_pad_token_id, LIFTDataCollator, WrapperForIterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer
from typing import List, Literal


def test_latency_icl(
    input_lengths: List[int],
    output_path: str,
    model_name_or_path: str = "models/Meta-Llama-3-8B-Instruct",
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    for input_len in tqdm.tqdm(input_lengths):
        input_ids = torch.tensor([[tokenizer.bos_token_id] * input_len], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        all_inf_time = []
        for output_length in [1, 1000, 2000, 3000, 4000]:
            time_st = torch.cuda.Event(enable_timing=True)
            time_ed = torch.cuda.Event(enable_timing=True)
            time_st.record()
            model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                eos_token_id=[],
                pad_token_id=get_pad_token_id(tokenizer),
                max_new_tokens=output_length,
                use_cache=True,
                do_sample=False,
            )
            time_ed.record()
            torch.cuda.synchronize()
            inf_time = time_st.elapsed_time(time_ed) / 1000.0  # Convert to seconds
            all_inf_time.append(inf_time)
        with open(output_path, "a") as f:
            f.write(json.dumps({"input_len": input_len, "inf_time": all_inf_time}) + "\n")
        del input_ids, attention_mask
        torch.cuda.empty_cache()


def test_latency_lift(
    input_lengths: List[int],
    output_path: str,
    input_path: str = "outputs/niah/input.jsonl",
    model_name_or_path: str = "models/Meta-Llama-3-8B-Instruct",
    server_config: str = "configs/generator/vllm-qwen3-backend.yaml",
    dataset_config: str = "configs/generator/vllm-qwen3-efficiency.yaml",
    adapter: Literal["lora"] = "lora",
    adapter_config: str = "configs/adapter/lora_128.yaml",
    training_config: str = "configs/training/lift_lora_fast.yaml",
):
    import accelerate
    accelerator = accelerate.Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    # We use the NIAH input as the base context. Notice that we can't use
    # meaningless context like ICL here, since LIFT's generation cost depends on
    # the context.
    with open(input_path, "r") as f:
        for line in f:
            data = json.loads(line)
            if data["length"] == 128000:
                break
    context = data["context"]
    context_ids = tokenizer(context, add_special_tokens=False).input_ids
    with open(training_config, "r") as f:
        training_args = yaml.safe_load(f)
    training_args["per_device_train_batch_size"] *= torch.cuda.device_count()
    training_args = custom_2_hf_training_args(
        training_args,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the real epoch is controlled by `AutoControlStopCallback`
        use_liger_kernel=True,
    )

    if os.path.exists(output_path):
        with open(output_path, "r") as f:
            num_resumed = len(f.readlines())
    else:
        num_resumed = 0

    server = AsyncOpenAIServer.from_config(server_config)
    for input_len in tqdm.tqdm(input_lengths[num_resumed:]):
        selected_context_ids = context_ids[:input_len]
        sentences = [tokenizer.decode(selected_context_ids[i:i+1000]) for i in range(0, input_len, 1000)]
        model = load_training_model(model_name_or_path, adapter, adapter_config=adapter_config)
        time_st = torch.cuda.Event(enable_timing=True)
        time_ed = torch.cuda.Event(enable_timing=True)
        time_st.record()
        # We do not use cache here -- generation is forced to be online.
        dataset = EverySentenceDataset.from_config(dataset_config, tokenizer, server=server, sentences=sentences)
        control_callback = EverySentenceStopCallback(dataset, training_args.num_train_epochs, training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        dataset = WrapperForIterableDataset(dataset, training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)
        trainer = Trainer(model, training_args, data_collator=LIFTDataCollator(tokenizer), train_dataset=dataset, callbacks=[control_callback])
        trainer.train()
        time_ed.record()
        train_time = time_st.elapsed_time(time_ed) / 1000.0  # Convert to seconds
        all_inf_time = []
        del dataset, trainer, control_callback
        if accelerator.is_main_process:
            input_ids = torch.tensor([[tokenizer.bos_token_id]], dtype=torch.long, device=model.device)
            attention_mask = torch.ones_like(input_ids)
            model.eval()
            for output_length in [1, 1000, 2000, 3000, 4000]:
                st_time = time.time()
                model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    eos_token_id=[],
                    pad_token_id=get_pad_token_id(tokenizer),
                    max_new_tokens=output_length,
                    use_cache=True,
                    do_sample=False,
                )
                ed_time = time.time()
                inf_time = ed_time - st_time
                all_inf_time.append(inf_time)
            del input_ids, attention_mask
        accelerator.wait_for_everyone()
        del model
        if accelerator.is_main_process:
            with open(output_path, "a") as f:
                f.write(json.dumps({"input_len": input_len, "train_time": train_time, "inf_time": all_inf_time}) + "\n")
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_path", type=str, required=True, help="Path to save the output results.")
    parser.add_argument("--input_lengths", type=int, nargs="*", default=[1000, 2000, 4000, 8000], help="List of input lengths to test.")
    parser.add_argument("--model_name_or_path", type=str, default="models/Meta-Llama-3-8B-Instruct", help="Model name or path.")
    parser.add_argument("--type", choices=["icl", "lift"], default="icl", help="Type of test to run.")
    args = parser.parse_args()
    if args.type == "icl":
        test_latency_icl(
            input_lengths=args.input_lengths,
            output_path=args.output_path,
            model_name_or_path=args.model_name_or_path,
        )
    elif args.type == "lift":
        test_latency_lift(
            input_lengths=args.input_lengths,
            output_path=args.output_path,
            model_name_or_path=args.model_name_or_path,
        )
