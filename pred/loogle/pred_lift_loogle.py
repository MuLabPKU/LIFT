"""
Get the output of LIFT on LooGLE (both ShortQA and LongQA).
"""
import json
import os
import time
import torch
import tqdm
import yaml
from accelerate import Accelerator
from lift.model import load_training_model
from lift.synqa import (
    AsyncOpenAIServer,
    EverySentenceDataset,
    EverySentenceStopCallback,
    ServerError,
)
from lift.utils import custom_2_hf_training_args, is_instruct, get_pad_token_id, LIFTDataCollator, WrapperForIterableDataset
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer, Trainer
from transformers import TrainerCallback
from typing import Any, Dict, List, Literal, Optional


class EpochTimeCallback(TrainerCallback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.st_time = None
    
    def on_epoch_begin(self, args, state, control, **kwargs):
        self.st_time = time.time()
    
    def on_epoch_end(self, args, state, control, **kwargs):
        self.epoch_times.append(time.time() - self.st_time)


# The original prompt given in the config file of LooGLE
LOOGLE_PROMPT_WITH_ICL = "Please answer the question based on the long texts below. \n{input}\nQuestion: {Q}\nAnswer: "
# The customized prompt for LIFTed model
CUSTOM_PROMPT_WITHOUT_ICL = "Please answer the question based on \"{title}\".\nQuestion: {Q}\nAnswer: "


class LooGLEEvalDataset(Dataset):
    """
    Arrange it as a torch Dataset for distributed (data-parallel) inference.
    """
    def _construct(self, prompt: str, meta_data: Dict = {}):
        if self.is_instruct:
            msgs = [{"role": "user", "content": prompt}]
            # Support Qwen: disable reasoning
            input_ids = self.tokenizer.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, return_tensors="pt")[0]
        else:
            input_ids = self.tokenizer(prompt, add_special_tokens=True, return_tensors="pt").input_ids[0]
        if self.max_input_len is not None and input_ids.shape[-1] > self.max_input_len:
            front_len = self.max_input_len // 2
            back_len = self.max_input_len - front_len
            input_ids = torch.concat((input_ids[..., :front_len], input_ids[..., -back_len:]), dim=-1)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, **meta_data}
    
    def __init__(self, tokenizer: PreTrainedTokenizer, input: str, title: str, test_qa: List[Dict], max_input_len: Optional[int] = None):
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.data = []
        self.is_instruct = is_instruct(tokenizer)
        for entry in test_qa:
            # W/ ICL
            prompt_w_icl = LOOGLE_PROMPT_WITH_ICL.format(input=input, title=title, Q=entry["Q"])
            self.data.append(self._construct(prompt_w_icl, {"truncated_icl": True}))
            # W/o ICL
            prompt_wo_icl = CUSTOM_PROMPT_WITHOUT_ICL.format(title=title, Q=entry["Q"])
            self.data.append(self._construct(prompt_wo_icl, {"truncated_icl": False}))
    
    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def train_field(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sent_model,
    synqa_type: Literal["every_sentence"],
    context: str,
    title: str,
    cache_path: str,
    dataset_config_path: str,
    is_main_process: bool,
    llm_server: AsyncOpenAIServer,
    training_config: Dict[str, Any],
):
    st_time = time.time()
    # NOTE We use dispatch_batches mode -- only the main process holds the dataset and it distributes the data to other processes.
    # Since we generates the data on-the-fly, this avoids redundant data generation.
    training_args = custom_2_hf_training_args(
        training_config,
        accelerator_config={"split_batches": True, "dispatch_batches": True, "even_batches": True},
        max_steps=int(10000),  # sufficiently large -- the stop of training is controlled by `control_callback`
        use_liger_kernel=True,
    )
    sentences = [s.text for s in sent_model(context).sents]
    training_dataset = EverySentenceDataset.from_config(dataset_config_path, tokenizer, is_main_process=is_main_process, server=llm_server, sentences=sentences, cache_path=cache_path, prompt_template="Please answer the question based on \"" + title + "\".\nQuestion: {question}\nAnswer:")
    mid_time = time.time()

    try:
        control_callback = EverySentenceStopCallback(training_dataset, training_args.num_train_epochs, training_args.train_batch_size * training_args.gradient_accumulation_steps)
        epochtime_callback = EpochTimeCallback()
        trainer = Trainer(model, training_args, data_collator=LIFTDataCollator(tokenizer), train_dataset=WrapperForIterableDataset(training_dataset, training_config["per_device_train_batch_size"]), callbacks=[control_callback, epochtime_callback])
        trainer.train()
    except Exception as e:
        raise e
    finally:
        if is_main_process and training_dataset.len_dataset is None and os.path.exists(cache_path):
            os.remove(cache_path)
    ed_time = time.time()

    return {
        "construction_time": mid_time - st_time,
        "training_time": ed_time - mid_time,
        "generation_time": training_dataset.generate_cost,
        "epoch_times": epochtime_callback.epoch_times,
        "total_training_time": ed_time - st_time,
    }


@torch.no_grad()
def inference_field(
    eval_accelerator: Accelerator,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    context: str,
    title: str,
    all_qa_pairs: List[Dict],
    max_input_len: int,
    max_new_tokens: int,
):
    model.disable_input_require_grads()
    eval_dataloader = DataLoader(LooGLEEvalDataset(tokenizer, context, title, all_qa_pairs, max_input_len), shuffle=False)
    eval_dataloader = eval_accelerator.prepare(eval_dataloader)
    try:
        all_responses = []
        for inputs in eval_dataloader:
            input_ids = inputs.pop("input_ids")
            attention_mask = inputs.pop("attention_mask")
            output_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=True,
                max_new_tokens=max_new_tokens,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=get_pad_token_id(tokenizer),
            )
            response = tokenizer.decode(output_ids[0, input_ids.shape[-1]:], skip_special_tokens=True)
            all_responses.extend(eval_accelerator.gather_for_metrics([response], use_gather_object=True))
            del output_ids, input_ids, attention_mask
    finally:
        eval_accelerator.free_memory(eval_dataloader)
    return all_responses


def main(
    subtask: Literal["shortdep_qa", "longdep_qa"],
    output_path: str,
    synqa_type: Literal["every_sentence"],
    synqa_dir: str,
    server_config_path: str = "configs/generator/vllm-qwen3-backend.yaml",
    dataset_config_path: str = "configs/generator/vllm-qwen3-everysentence-10.yaml",
    loogle_dir: str = "datasets/loogle/",
    loogle_config_dir: str = "datasets/loogle_config/",
    model_name_or_path: str = "models/Meta-Llama-3-8B-Instruct",
    model_max_len: Optional[int] = 8192,
    adapter: Literal["none", "lora"] = "lora",
    adapter_config: Optional[str] = "configs/adapter/lora_128.yaml",
    training_config: str = "configs/training/lift_lora.yaml",
    num_workers: int = 1,
    worker_id: int = 0,
    test_index: Optional[List[int]] = None,
    overwrite: bool = False,
):
    eval_accelerator = Accelerator()

    if not output_path.endswith(".jsonl"):
        raise ValueError(f"Expect `output_path` to be a JSONL file, but receive \"{output_path}\".")
    if eval_accelerator.is_main_process:
        output_dir = os.path.split(output_path)[0]
        if len(output_dir) > 0:
            os.makedirs(output_dir, exist_ok=True)
    os.makedirs(synqa_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    with open(os.path.join(loogle_dir, subtask + ".jsonl"), "r") as f:
        loogle_dataset = [json.loads(l) for l in f]
    with open(os.path.join(loogle_config_dir, "task2maxlen.json"), "r") as f:
        max_new_tokens = json.load(f)[subtask]
    max_input_len = None if model_max_len is None else model_max_len - max_new_tokens

    with open(training_config, "r") as f:
        training_config = yaml.safe_load(f)
    training_config["per_device_train_batch_size"] *= torch.cuda.device_count()  # due to "split_batch==True"

    if test_index is None:
        test_index = list(range(worker_id, len(loogle_dataset), num_workers))
    loogle_dataset = [loogle_dataset[i] for i in test_index]
    if os.path.exists(output_path):
        if overwrite:
            if eval_accelerator.is_main_process:
                os.remove(output_path)
            print(f"Overwrite \"{output_path}\"!")
            num_resumed = 0
        else:
            with open(output_path, "r") as f:
                num_resumed = len(f.readlines())
            print(f"Resume {num_resumed} entries from \"{output_path}\"!")
    else:
        num_resumed = 0

    llm_server = AsyncOpenAIServer.from_config(server_config_path)

    import spacy
    sent_model = spacy.load("en_core_web_sm")

    for entry_id, entry in tqdm.tqdm(zip(test_index[num_resumed:], loogle_dataset[num_resumed:]), total=len(test_index) - num_resumed, desc="LooGLE Eval"):
        cache_path = os.path.join(synqa_dir, str(entry_id) + ".jsonl")
        try:
            model = load_training_model(model_name_or_path, adapter=adapter, adapter_config=adapter_config)
            training_costs = train_field(model, tokenizer, sent_model, synqa_type, entry["input"], entry["title"], cache_path, dataset_config_path, eval_accelerator.is_main_process, llm_server, training_config)
            model.eval()
            torch.cuda.empty_cache()
            all_qa_pairs = eval(entry["qa_pairs"])
            all_responses = inference_field(eval_accelerator, model, tokenizer, entry["input"], entry["title"], all_qa_pairs, max_input_len, max_new_tokens)
            all_test_cases = [{"truncated_icl": truncated_icl, **qa} for qa in all_qa_pairs for truncated_icl in [True, False]]
            entry["qa_pairs"] = [{"P": response, **qa} for qa, response in zip(all_test_cases, all_responses)]
            entry["input"] = entry["input"][:200]  # Only used for identification
            entry["costs"] = training_costs
            if eval_accelerator.is_main_process:
                with open(output_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            eval_accelerator.wait_for_everyone()
            torch.cuda.empty_cache()
        except ServerError as e:  # Record the server error and skip the current entry
            if eval_accelerator.is_main_process:
                with open(output_path, "a") as f:
                    f.write(json.dumps({"error": str(e)}) + "\n")
            eval_accelerator.wait_for_everyone()
        except Exception as e:
            raise e
        finally:
            del model


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("subtask", choices=["shortdep_qa", "longdep_qa"])
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--synqa_type", required=True, choices=["every_sentence"])
    parser.add_argument("--synqa_dir", required=True)
    parser.add_argument("--server_config_path", default="configs/generator/vllm-qwen3-backend.yaml")
    parser.add_argument("--dataset_config_path", required=True)
    parser.add_argument("--loogle_dir", default="datasets/loogle/")
    parser.add_argument("--loogle_config_dir", default="datasets/loogle_config/")
    parser.add_argument("--model_name_or_path", default="models/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--model_max_len", type=int, default=8192)
    parser.add_argument("--adapter", choices=["none", "lora"], default="lora", help="The type of the adapter. `none` means full-parameter training.")
    parser.add_argument("--adapter_config", default="configs/adapter/lora_128.yaml", help="The YAML file storing the hyperparameters of the adapter.")
    parser.add_argument("--training_config", default="configs/training/lift_lora.yaml", help="The YAML file storing the arguments of `LIFTTrainingArguments`.")
    parser.add_argument("--num_workers", type=int, default=1, help="The number of workers.")
    parser.add_argument("--worker_id", type=int, default=0, help="The index of the current worker. Starts from 0.")
    parser.add_argument("--test_index", type=int, nargs="*", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    
    main(**vars(args))
