from __future__ import annotations

import json
import os
import time
import torch
from accelerate.utils import broadcast_object_list
from dataclasses import dataclass, field
from pydantic import BaseModel
from random import shuffle
from torch.utils.data import IterableDataset
from transformers import (
    PreTrainedTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from typing import List, Optional
from .synqa_server import LLMServerBase, ServerError
from ..utils import is_instruct, load_config


class ESQAPair(BaseModel):
    question: str
    answer: str


class ESQAList(BaseModel):
    qa_list: List[ESQAPair]


DEFAULT_TASK_DESCRIPTION = """# Instruction
You are given a paragraph extracted from an article. Please read it and generate at most {num_questions} different questions based on the content of the **last part** of the paragraph. The questions should be diverse in both their form and the content they inquire about.
"""
DEFAULT_JSON_FORMAT = """# Output format
Output in the JSON format. DO NOT output anything else.
{
    "qa_list": [
        {"question": ..., "answer": ...},
        {"question": ..., "answer": ...},
        ...
    ]
}
"""
DEFAULT_USER_PROMPT = """The paragraph:
{paragraph}

The last part of the paragraph:
{sentence}

Generate {num_questions} different questions based on the content of the last part of the paragraph.
"""


@dataclass
class EverySentenceConfig:
    use_pydantic: bool = field(default=True)
    do_shuffle: bool = field(default=True)
    system_role_name: str = field(default="developer")
    num_tasks_per_sentence: int = field(default=10)
    max_sentence_length: int = field(default=-1)
    extra_requirement: str = field(default="")
    disable_training_system_prompt: bool = field(default=False)
    max_training_data_length: Optional[int] = field(default=None)
    
    def __postinit__(self):
        if len(self.extra_requirement) > 0 and not self.extra_requirement.endswith("\n"):
            self.extra_requirement += "\n"


class EverySentenceDataset(IterableDataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        server: Optional[LLMServerBase] = None,
        sentences: Optional[List[str]] = None,
        cache_path: Optional[str] = None,
        use_pydantic: bool = True,
        do_shuffle: bool = True,
        system_role_name: str = "developer",
        is_main_process: bool = True,
        num_tasks_per_sentence: int = 10,
        prompt_template: Optional[str] = None,
        max_sentence_length: int = -1,
        extra_requirement: str = "",
        disable_training_system_prompt: bool = False,
        max_training_data_length: Optional[int] = None,
    ):
        """
        Args:
            tokenizer (`PreTrainedTokenizer`):
                The tokenizer of the model to train.
            server (`LLMServerBase`, *optional*):
                The LLM server used as the generator. If the cache file does not exist, the dataset will generate the QA pairs with the server. Defaults to `None`.
            sentences (`List[str]`, *optional*):
                The raw sentences. It should be provided if the cache file does not exist. Defaults to `None`.
            cache_path (`str`, *optional*):
                The path to the cache file. If it is provided and the cache file exists, the dataset will load the cache without generating again; if it is provided but the cache file does not exist, the generated QA pairs are saved in the cache file. Defaults to `None`.
            use_pydantic (`bool`, *optional*):
                Whether to use the pydantic BaseModel for structured output; otherwise, use JSON format. Defaults to `True`.
            do_shuffle (`bool`, *optional*):
                Whether to shuffle the data for each iteration. Notice that the first iteration is not shuffled if the cache is not ready. Default to `True`.
            system_role_name (`str`, *optional*):
                The name of the system role. For OpenAI API, it should be `"developer"`, while for most other APIs, it should be `"system"`. Defaults to `"developer"`.
            is_main_process (`bool`, *optional*):
                Whether the dataset is materialized on the main process. Only the main process is allowed to generate QA pairs. Defaults to `True`.
            num_tasks_per_sentence (`int`, *optional*):
                The number of tasks to generate for each sentence.
            prompt_template (`str`, *optional*):
                The template of the input prompt. Wrap the question in `"{question}"`. If it is empty, the input will be the synthetic question without any prompt. Defaults to `None`. 
            max_sentence_length (`int`, *optional*):
                Batch sentenses into a single string as long as its length does not exceed max_sentence_length. Defaults to `-1`.
            extra_requirement (`str`, *optional*):
                The extra requirements inserted between DEFAULT_TASK_DESCRIPTION and DEFAULT_JSON_FORMAT (if exists). Defaults to `""`.
            disable_system_prompt (`bool`, *optional*):
                Whether to remove the system prompt in the training data since some models like Gemma do not support the system role. Defaults to `False`.
            max_training_data_length (`int`, *optional*):
                The maximum length of an entry of the training data. Leaving it as `None` means no limitation on the training data length. Defaults to `None`.
        """
        # Config arguments
        self.tokenizer = tokenizer
        self.use_pydantic = use_pydantic
        self.do_shuffle = do_shuffle
        self.cache_path = cache_path
        self.is_main_process = is_main_process
        self.prompt_template = "{question}" if prompt_template is None else prompt_template
        self.disable_training_system_prompt = disable_training_system_prompt
        self.max_training_data_length = max_training_data_length
        # Runtime variables
        self.error_record: Optional[Exception] = None
        self.len_dataset = None
        self.generate_cost = 0  # Record the time cost of generation
        # Preprocess sentences
        if max_sentence_length > 0:
            tmp_sentences = []
            num_sentences = len(sentences)
            i = 0
            while i < num_sentences:
                cur_length = len(sentences[i])
                j = i
                while j + 1 < num_sentences and cur_length + len(sentences[j + 1]) <= max_sentence_length:
                    j += 1
                    cur_length += len(sentences[j])
                tmp_sentences.append(" ".join(sentences[i:j+1]))
                i = j + 1
            sentences = tmp_sentences
        # Prepare data
        if is_main_process:
            if cache_path is not None and os.path.exists(cache_path):
                with open(cache_path, "r") as f:
                    self.data = [self._construct_input(qa["question"], qa["answer"]) for l in f for qa in json.loads(l)["qa_list"]]
                self.len_dataset = len(self.data)
            else:
                self.data = None
                if server is None or sentences is None:
                    if cache_path is None:
                        raise ValueError("server and sentences should be provided if cache_path is None.")
                    else:
                        raise ValueError(f"\"{cache_path}\" does not exist but server and sentences are not provided.")
                batched_messages = self._construct_messages(sentences, use_pydantic, system_role_name, num_tasks_per_sentence, extra_requirement)
                self.initial_iter = server.batch_sampling(batched_messages, structure=ESQAList)
        else:
            self.data = None

    @classmethod
    def from_config(cls, config: str | EverySentenceConfig, tokenizer: PreTrainedTokenizer, is_main_process: bool = True, server: Optional[LLMServerBase] = None, sentences: Optional[List[str]] = None, cache_path: Optional[str] = None, prompt_template: Optional[str] = None) -> EverySentenceDataset:
        if isinstance(config, str):
            config = load_config(EverySentenceConfig, config)
        return cls(
            tokenizer,
            is_main_process=is_main_process,
            server=server,
            sentences=sentences,
            cache_path=cache_path,
            prompt_template=prompt_template,
            **vars(config),
        )

    def _construct_messages(self, sentences: List[str], use_pydantic: bool, system_role_name: str, num_tasks_per_sentence: int, extra_requirement: str):
        batched_messages = []
        j = 0
        for i, sentence in enumerate(sentences):
            while j < i and len(" ".join(sentences[j:i])) > 2048:
                j += 1
            preceding_paragraph = " ".join(sentences[j:i+1])
            if use_pydantic:
                message = [
                    {"role": system_role_name, "content": DEFAULT_TASK_DESCRIPTION.format(num_questions=num_tasks_per_sentence) + extra_requirement},
                    {"role": "user", "content": DEFAULT_USER_PROMPT.format(sentence=sentence, paragraph=preceding_paragraph, num_questions=num_tasks_per_sentence)},
                ]
            else:
                message = [
                    {"role": system_role_name, "content": DEFAULT_TASK_DESCRIPTION.format(num_questions=num_tasks_per_sentence) + extra_requirement + DEFAULT_JSON_FORMAT},
                    {"role": "user", "content": DEFAULT_USER_PROMPT.format(sentence=sentence, paragraph=preceding_paragraph, num_questions=num_tasks_per_sentence)},
                ]
            batched_messages.append(message)
        return batched_messages
    
    def _construct_input(self, question: str, answer: str):
        if is_instruct(self.tokenizer):
            if self.disable_training_system_prompt:
                messages = [
                    {"role": "user", "content": self.prompt_template.format(question=question)},
                    {"role": "assistant", "content": answer},
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": self.prompt_template.format(question=question)},
                    {"role": "assistant", "content": answer},
                ]
            input_ids = self.tokenizer.apply_chat_template(messages, return_tensors="pt").flatten()
            # [Feat] Support Qwen: disable thinking
            input_length = len(self.tokenizer.apply_chat_template(messages[:-1], add_generation_prompt=True, enable_thinking=False))
            labels = torch.clone(input_ids)
            labels[:input_length] = -100
            if self.max_training_data_length is not None and input_ids.shape[0] > self.max_training_data_length:
                input_ids = torch.concat((input_ids[:self.max_training_data_length // 2], input_ids[-self.max_training_data_length // 2:]))
                labels = torch.concat((labels[:self.max_training_data_length // 2], labels[-self.max_training_data_length // 2:]))
            attention_mask = torch.ones_like(input_ids)
        else:
            prompt_ids = self.tokenizer(question + "\nAnswer:\n\n", add_special_tokens=True, return_tensors="pt").input_ids.flatten()  # Add BOS
            answer_ids = self.tokenizer(answer, return_tensors="pt").input_ids.flatten()
            answer_ids = torch.concat((answer_ids, torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)))
            input_ids = torch.concat((prompt_ids, answer_ids))
            labels = torch.concat((torch.full_like(input_ids, -100), answer_ids))
            if self.max_training_data_length is not None and input_ids.shape[0] > self.max_training_data_length:
                input_ids = torch.concat((input_ids[:self.max_training_data_length // 2], input_ids[-self.max_training_data_length // 2:]))
                labels = torch.concat((labels[:self.max_training_data_length // 2], labels[-self.max_training_data_length // 2:]))
            attention_mask = torch.ones_like(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
    
    def _construct_empty_input(self):
        input_ids = torch.tensor([self.tokenizer.eos_token_id], dtype=torch.long)
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
            "labels": input_ids,
        }
    
    def __iter__(self):
        self.error_record = None
        if self.data is None:
            st_time = time.time()
            self.data = []
            is_empty = True
            try:
                for response in self.initial_iter:
                    if self.cache_path is not None:
                        with open(self.cache_path, "a") as f:
                            f.write(response.model_dump_json() + "\n")
                    for qa in response.qa_list:
                        self.data.append(self._construct_input(qa.question, qa.answer))
                        is_empty = False
                        yield self.data[-1]
            except ServerError as e:
                self.error_record = e  # Cache the error and it will be broadcast to all the processes later
                if is_empty:  # Avoid returning an empty dataset, which will lead to error in the Trainer
                    yield self._construct_empty_input()
                return  # Stop the iteration
            self.len_dataset = len(self.data)
            self.generate_cost = time.time() - st_time
        else:
            if self.do_shuffle:
                shuffle(self.data)
            for entry in self.data:
                yield entry


class EverySentenceStopCallback(TrainerCallback):
    def __init__(self, dataset: EverySentenceDataset, num_train_epochs: int, real_batch_size: int):
        """
        Args:
            dataset (`EverySentenceDataset`):
                The instance of the dataset for training.
            num_train_epochs (`int`):
                The number of training epochs as expected.
            real_batch_size (`int`):
                The batch size considering multi-devices and gradient accumulation.
        """
        self.dataset = dataset
        self.num_train_epochs = num_train_epochs
        self.real_batch_size = real_batch_size
        
    def _should_training_stop(self, global_step: int):
        len_dataset = self.dataset.len_dataset
        if len_dataset is None:
            return False
        else:
            max_step = (len_dataset + self.real_batch_size - 1) // self.real_batch_size * self.num_train_epochs
            return global_step >= max_step
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if self._should_training_stop(state.global_step):
            control.should_training_stop = True
        if self.dataset.error_record is not None:
            control.should_training_stop = True
        control.should_training_stop = broadcast_object_list([control.should_training_stop])[0]
        return control

    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        error_message = broadcast_object_list([self.dataset.error_record])[0]
        if error_message is not None:
            raise error_message
