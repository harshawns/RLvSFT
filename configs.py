"""Shared experiment knobs for SFT and GRPO (keep arms from drifting)."""

MODEL_ID = "arcee-ai/AFM-4.5B-Base"

# LoRA — identical for SFT and GRPO-from-base (and for SFT→GRPO init adapters).
LORA_R = 16
LORA_ALPHA = 32
LORA_TARGET_MODULES = "all-linear"
LORA_TASK_TYPE = "CAUSAL_LM"


def build_lora_config():
    from peft import LoraConfig

    return LoraConfig(
        task_type=LORA_TASK_TYPE,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
    )
