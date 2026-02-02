import torch
from .lora import load_lora_model
from typing import Literal, Optional
from transformers import AutoModelForCausalLM


def load_training_model(
    model_name_or_path: str,
    adapter: Literal["none", "lora"],
    adapter_config: Optional[str] = None,
):
    if adapter == "none":  # Full-parameter fine-tuning
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16)  # Do not use `device_map` for `accelerate`
    elif adapter == "lora":
        model = load_lora_model(model_name_or_path, adapter_config)
    else:
        raise ValueError(f"Unknown adapter \"{adapter}\". `adapter` should be `\"none\"` or `\"lora\"`.")
    model.enable_input_require_grads()
    return model


__all__ = [
    "load_lora_model",
    "load_training_model",
]
