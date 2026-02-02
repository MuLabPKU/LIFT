import logging
import torch
import yaml
from torch.nn import Parameter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import IterableDataset
from transformers import PreTrainedModel, PreTrainedTokenizer, TrainingArguments
from typing import Any, Dict, List, Tuple, Type


logging.basicConfig(format="%(asctime)s - %(filename)s - [%(levelname)s] - %(message)s", level=logging.WARNING)
logger = logging.getLogger(__name__)


class LIFTDataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.pad_token_id = tokenizer.eos_token_id if getattr(tokenizer, "pad_token_id", None) is None else tokenizer.pad_token_id

    def __call__(self, batch: List[Dict[str, torch.Tensor]]):
        result = {}
        for k in batch[0].keys():
            if k == "input_ids":  # These should be packed into a tensor
                result[k] = pad_sequence([b[k] for b in batch], batch_first=True, padding_value=self.pad_token_id)
            elif k == "attention_mask":
                result[k] = pad_sequence([b["attention_mask"] for b in batch], batch_first=True, padding_value=0)
            elif k == "labels":
                result[k] = pad_sequence([b["labels"] for b in batch], batch_first=True, padding_value=-100)
            else:  # Metadata
                result[k] = [b[k] for b in batch]
        return result


class WrapperForIterableDataset(IterableDataset):
    def __init__(self, inner_dataset: IterableDataset, total_batch_size: int):
        self.inner_dataset = inner_dataset
        self.total_batch_size = total_batch_size
    
    def __iter__(self):
        cur_iter = iter(self.inner_dataset)
        while True:
            minibatch = []
            stop_iteration = False
            for _ in range(self.total_batch_size):
                try:
                    minibatch.append(next(cur_iter))
                except StopIteration:
                    stop_iteration = True
                    break
            if len(minibatch) == 0:
                break
            if len(minibatch) < self.total_batch_size:
                minibatch = (minibatch * (self.total_batch_size // len(minibatch) + 1))[:self.total_batch_size]
            for entry in minibatch:
                yield entry
            if stop_iteration:
                break


def is_instruct(tokenizer: PreTrainedTokenizer):
    return getattr(tokenizer, "chat_template", None) is not None


def get_pad_token_id(tokenizer: PreTrainedTokenizer):
    return tokenizer.eos_token_id if getattr(tokenizer, "pad_token", None) is None else tokenizer.pad_token_id

def load_config(cls: Type, config_path: str):
    if not config_path.endswith('.yaml'):
        raise AssertionError(f"Expect `config_path` to be a YAML file, but receive \"{config_path}\".")
    with open(config_path, 'r') as f:
        res = cls(**yaml.safe_load(f))
    return res


def get_parameter_groups(model: PreTrainedModel) -> Tuple[List[Parameter], List[Parameter]]:
    """
    Split the parameters of a PreTrainedModel into two groups -- the first requires weight decay while the second not (e.g., bias, layernorm).
    Args:
        model (`PreTrainedModel`):
            The pre-trained model.
    Returns:
        groups (`Tuple[List[Parameter], List[Parameter]]`):
        - decay_params (`List[Parameter]`): The parameters that require weight decay.
        - no_decay_params (`List[Parameter]`): The parameters that do not require weight decay.
    """
    decay_params, no_decay_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if any(nd in name.lower() for nd in ["bias", "ln", "layernorm", "norm"]):
                no_decay_params.append(param)
            else:
                decay_params.append(param)
    return decay_params, no_decay_params


def custom_2_hf_training_args(training_args: Dict[str, Any] | str, **additional_kwargs) -> TrainingArguments:
    """
    Transfer custom training arguments (in YAML) into HF TrainingArguments.
    Args:
        training_args (`Dict[str, Any] | str`):
            The training arguments in dict or the path to the YAML config.
    """
    default_args = dict(
        output_dir="models/temp",
        overwrite_output_dir=True,
        do_train=True,
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=50,
        lr_scheduler_type="constant",
        label_names=["labels"],
    )
    if isinstance(training_args, str):
        with open(training_args, "r") as f:
            training_args = yaml.safe_load(f)
    default_args.update(training_args)
    default_args.update(additional_kwargs)
    return TrainingArguments(**default_args)
