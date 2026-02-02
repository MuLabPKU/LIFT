import torch
import yaml
from transformers import AutoConfig, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig


def load_lora_model(base_model_name_or_path: str, lora_config_path: str):
    """
    Load the LoRA adapter for **accelerate** training.
    Args:
        base_model_name_or_path (`str`):
            The HF name or the local path to the base model.
        lora_config_path (`str`):
            The path to the config file of the LoRA adapter.
    """
    if not lora_config_path.endswith(".yaml"):
        raise AssertionError(f"Expect `lora_config_path` to be a YAML file, but receive \"{lora_config_path}\".")
    base_config = AutoConfig.from_pretrained(base_model_name_or_path)
    # Do not use device_map when launching with `accelerate`
    if "gemma" in base_config.model_type.lower():
        # It's recommended to train Gemma models using eager attention implementation.
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="eager")
    else:
        base_model = AutoModelForCausalLM.from_pretrained(base_model_name_or_path, torch_dtype=torch.bfloat16)
    with open(lora_config_path, "r") as f:
        lora_config = LoraConfig(**yaml.safe_load(f))
    model = get_peft_model(base_model, lora_config)
    return model
