from __future__ import annotations

import asyncio
import openai
import os
import threading
import torch
from abc import abstractmethod, ABC
from dataclasses import dataclass, field
from pydantic import BaseModel
from typing import Any, Dict, Generator, List, Literal, Optional, TYPE_CHECKING
from ..utils import load_config, logger

if TYPE_CHECKING:
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams


class ServerError(Exception):
    def __init__(self, error_message: str):
        super().__init__(error_message)


class LLMServerBase(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    @abstractmethod
    def batch_sampling(self, batched_messages: List[List[Dict]], *args, **kwargs) -> Generator[str | BaseModel, Any, None]:
        """
        Returns an iterator returning the responses to the batched requests (keeping order).
        """
        pass


@dataclass
class AsyncOpenAIConfig:
    model: str
    base_url: Optional[str] = field(default=None)
    api_key: str = field(default="auto")  # using a string starting with "$" to represent environment vars
    max_concurrency: int = field(default=32)
    max_retries: int = field(default=2)
    max_output_tokens: Optional[int] = field(default=None)
    
    def __post_init__(self):
        if self.model.startswith("$"):
            self.model = os.environ[self.model[1:]]
        if self.base_url is not None and self.base_url.startswith("$"):
            self.base_url = os.environ[self.base_url[1:]]
        if self.api_key.startswith("$"):
            self.api_key = os.environ[self.api_key[1:]]


class AsyncOpenAIServer(LLMServerBase):
    def __init__(self, model: str, *args, base_url: Optional[str] = None, api_key: str = os.environ.get("OPENAI_API_KEY", "***"), max_concurrency: int = 32, max_retries: int = 2, max_output_tokens: Optional[int] = None, **kwargs):
        """
        The server based on AsyncOpenAI API.
        Args:
            model (`str`):
                The model name.
            base_url (`str`, *optional*):
                The `base_url` argument in `AsyncOpenAI`. Defaults to `None` (i.e., the OpenAI base URL).
            api_key (`str`, *optional*):
                The API key. Defaults to the environment var `$OPENAI_API_KEY` if it exists; otherwise, defaults to `"***"`.
            max_concurrency (`int`, *optional*):
                The max number of parallel requests. Defaults to `32`.
            max_retries (`int`, *optional*):
                The max number of retries after an error. Defaults to `2` (i.e., try up to 3 times).
            max_output_tokens (`int`, *optional*):
                The max number of tokens to generate for each request. Defaults to `None`.
        """
        super().__init__(*args, **kwargs)
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.max_output_tokens = max_output_tokens
        # Record
        self.usage = {}
        # Check API
        self.if_support_response = self._if_support_response()
    
    @classmethod
    def from_config(cls, config: AsyncOpenAIConfig | str) -> AsyncOpenAIServer:
        if isinstance(config, str):
            config = load_config(AsyncOpenAIConfig, config)
        return cls(**vars(config))
        
    def _if_support_response(self):
        """
        Check whether the endpoint supports the response APIs.
        """
        class _TempBaseModel(BaseModel):
            question: str
            answer: str
        
        client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
        temp_messages = [
            {"role": "user", "content": "Generate a short question-answer pair."}
        ]
        for _ in range(3):  # retry
            try:
                _ = client.responses.parse(
                    model=self.model,
                    input=temp_messages,
                    text_format=_TempBaseModel,
                    max_output_tokens=1024,
                )
            except Exception as e:
                continue
            return True
        return False
    
    def _update_usage(self, usage, src: Dict):
        if usage is None:
            return
        for k, v in usage.items():
            if isinstance(v, int):
                if k in src:
                    assert isinstance(src[k], int)
                    src[k] += v
                else:
                    src[k] = v
            elif v is not None:
                if k in src:
                    assert isinstance(src[k], dict)
                else:
                    src[k] = {}
                self._update_usage(v, src[k])
    
    def batch_sampling(self, batched_messages: List[List[Dict]], structure: Optional[BaseModel] = None):
        cond = threading.Condition()
        cache: List[Optional[str | BaseModel]] = [None for _ in range(len(batched_messages))]
        error: Optional[Exception] = None
        thread_stop = False
        
        async def _processor_all():
            async with openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url) as client:
                sem = asyncio.Semaphore(self.max_concurrency)

                async def _processor_single(idx: int, messages: List[Dict]):
                    delay = 0.5
                    for trying_time in range(self.max_retries + 1):
                        try:
                            async with sem:
                                if structure is None:
                                    response = await client.chat.completions.create(
                                        model=self.model,
                                        messages=messages,
                                        max_completion_tokens=self.max_output_tokens,
                                    )
                                    self._update_usage(response.usage.model_dump(), self.usage)
                                    content = response.choices[0].message.content
                                else:
                                    if self.if_support_response:
                                        response = await client.responses.parse(
                                            model=self.model,
                                            input=messages,
                                            text_format=structure,
                                            max_output_tokens=self.max_output_tokens,
                                        )
                                        content = response.output_parsed
                                    else:
                                        response = await client.chat.completions.create(
                                            model=self.model,
                                            messages=messages,
                                            response_format={"type": "json_object"},
                                            max_completion_tokens=self.max_output_tokens,
                                        )
                                        content = structure.model_validate_json(response.choices[0].message.content)
                                    self._update_usage(response.usage.model_dump(), self.usage)
                            with cond:
                                cache[idx] = content
                                cond.notify_all()
                            break
                        except Exception as e:
                            if trying_time == self.max_retries:
                                error_message = f"Fail to respond to the following message due to {e}:\n{messages[-1]['content'][:100]}..."
                                logger.error(error_message)
                                with cond:
                                    nonlocal error
                                    error = ServerError(error_message)
                                    cond.notify_all()
                            else:
                                await asyncio.sleep(delay)
                                delay *= 2

                tasks = [asyncio.create_task(_processor_single(i, m)) for i, m in enumerate(batched_messages)]
                try:
                    await asyncio.gather(*tasks)
                except Exception as e:
                    for t in tasks:
                        t.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                    with cond:
                        nonlocal error
                        error = e
                        cond.notify_all()
                finally:
                    with cond:
                        nonlocal thread_stop
                        thread_stop = True  # May due to completion or error
                        cond.notify_all()
            
        def _worker_thread_main():
            asyncio.run(_processor_all())
        
        worker_thread = threading.Thread(target=_worker_thread_main, daemon=True)
        worker_thread.start()
        
        def _generator():
            nonlocal error, cache
            for i in range(len(batched_messages)):
                with cond:
                    cond.wait_for(lambda: (cache[i] is not None) or (error is not None) or thread_stop)
                    if error is not None:
                        raise error
                    if cache[i] is None:
                        raise AssertionError(f"The cache #{i} is NULL.")
                    result = cache[i]
                    cond.notify_all()
                yield result
        
        return _generator()


class CustomVLLMServer(LLMServerBase):
    def __init__(
        self,
        model: str,
        sampling_params: Dict[str, Any],
        *args,
        gpu_memory_utilization: float = 0.95,
        tensor_parallel_size: int = torch.cuda.device_count(),
        dtype: str = "bfloat16",
        max_concurrency: int = 32,
        max_retries: int = 2,
        max_model_len: Optional[int] = None,
        batch_timeout: int = 60,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        from vllm import LLM
        # Config
        self.sampling_params = sampling_params
        self.max_concurrency = max_concurrency
        self.max_retries = max_retries
        self.batch_timeout = batch_timeout
        # Load the vLLM LLM
        self.llm = LLM(model, gpu_memory_utilization=gpu_memory_utilization, tensor_parallel_size=tensor_parallel_size, dtype=dtype, max_model_len=max_model_len)
    
    def batch_sampling(self, batched_messages, structure: Optional[BaseModel] = None):
        # Condition
        cond = threading.Condition()
        error: Optional[Exception] = None
        thread_stop = False
        # Init cache
        cache: List[Optional[str | BaseModel]] = [None for _ in range(len(batched_messages))]
        
        def _worker_thread_main():
            from vllm import SamplingParams
            from vllm.sampling_params import GuidedDecodingParams
            if structure is None:
                sampling_params = SamplingParams(**self.sampling_params)
            else:
                sampling_params = SamplingParams(guided_decoding=GuidedDecodingParams(json=structure.model_json_schema()), **self.sampling_params)
            nonlocal error, cache, thread_stop
            for i in range(0, len(batched_messages), self.max_concurrency):
                for try_times in range(self.max_retries + 1):
                    try:
                        minibatch_messages = batched_messages[i:i+self.max_concurrency]
                        minibatch_responses = self.llm.chat(minibatch_messages, sampling_params, use_tqdm=False)
                        for i_offset, r in enumerate(minibatch_responses):  # TODO: It may be optimized ...
                            with cond:
                                if structure is None:
                                    cache[i + i_offset] = r.outputs[0].text
                                else:
                                    cache[i + i_offset] = structure.model_validate_json(r.outputs[0].text)
                                cond.notify_all()
                        break
                    except Exception as e:
                        if try_times == self.max_retries:
                            with cond:
                                error = e
                                cond.notify_all()
            with cond:
                thread_stop = True
                cond.notify_all()
        
        worker_thread = threading.Thread(target=_worker_thread_main, daemon=True)
        worker_thread.start()
        
        def _generator():
            nonlocal error, cache, thread_stop
            try:
                for i in range(len(batched_messages)):
                    with cond:
                        if not cond.wait_for(lambda: (cache[i] is not None) or (error is not None) or thread_stop, timeout=self.batch_timeout):
                            raise TimeoutError(f"Timeout when waiting for entry {i}.")
                        if error is not None:
                            raise error
                        if cache[i] is None:
                            raise AssertionError(f"The cache #{i} is NULL.")
                        result = cache[i]
                        cond.notify_all()
                    yield result
            finally:
                worker_thread.join(timeout=60)

        return _generator()
