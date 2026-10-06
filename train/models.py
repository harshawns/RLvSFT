"""Shared model / tokenizer loading for the feasibility harness."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).resolve().parent / "configs"

MODEL_ALIASES: dict[str, str] = {
    "qwen": "Qwen/Qwen3.5-4B-Base",
    "gemma": "google/gemma-3n-E4B",
    "afm": "arcee-ai/AFM-4.5B-Base",
}


@dataclass
class ModelConfig:
    alias: str
    model_id: str
    revision: str | None = None
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "sdpa"
    trust_remote_code: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Any = "all-linear"
    pad_token_fallback: str = "eos"
    max_seq_length: int = 1024
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_alias(cls, alias: str) -> "ModelConfig":
        key = alias.lower().strip()
        if key not in MODEL_ALIASES:
            raise ValueError(f"Unknown model alias '{alias}'. Choose: {sorted(MODEL_ALIASES)}")
        path = CONFIG_DIR / f"{key}.yaml"
        data: dict[str, Any] = {}
        if path.exists():
            data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in data.items() if k in known and k != "extra"}
        extra = {k: v for k, v in data.items() if k not in known}
        kwargs.setdefault("alias", key)
        kwargs.setdefault("model_id", MODEL_ALIASES[key])
        kwargs["extra"] = {**extra, **dict(kwargs.get("extra") or {})}
        return cls(**kwargs)


def resolve_model_id(model: str) -> str:
    key = model.lower().strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    return model


def _dtype(name: str):
    import torch

    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "auto": "auto",
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype: {name}")
    return mapping[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")


def load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    kwargs: dict[str, Any] = {
        "trust_remote_code": cfg.trust_remote_code,
        "token": hf_token(),
    }
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id, **kwargs)
    if tokenizer.pad_token is None:
        if cfg.pad_token_fallback == "eos" and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def _from_pretrained(model_cls, cfg: ModelConfig, **extra):
    kwargs: dict[str, Any] = {
        "torch_dtype": _dtype(cfg.torch_dtype),
        "trust_remote_code": cfg.trust_remote_code,
        "token": hf_token(),
        "attn_implementation": cfg.attn_implementation,
        **extra,
    }
    if cfg.revision:
        kwargs["revision"] = cfg.revision
    return model_cls.from_pretrained(cfg.model_id, **kwargs)


def load_base_model(cfg: ModelConfig, *, device_map: str | None = None):
    """Load a causal / conditional-generation base model for the alias."""
    from transformers import AutoModelForCausalLM

    extra: dict[str, Any] = {}
    if device_map is not None:
        extra["device_map"] = device_map

    try:
        return _from_pretrained(AutoModelForCausalLM, cfg, **extra)
    except Exception as causal_err:
        # Gemma-3n and similar multimodal cards may need the image-text class.
        try:
            from transformers import AutoModelForImageTextToText

            return _from_pretrained(AutoModelForImageTextToText, cfg, **extra)
        except Exception as mm_err:
            raise RuntimeError(
                f"Failed to load {cfg.model_id}. "
                f"CausalLM error: {causal_err}; ImageTextToText error: {mm_err}. "
                "If this is Gemma, accept the license at "
                "https://huggingface.co/google/gemma-3n-E4B and set HF_TOKEN."
            ) from mm_err


def apply_lora(model, cfg: ModelConfig, *, init_adapter: str | None = None):
    from peft import LoraConfig, PeftModel, get_peft_model

    if init_adapter:
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
        return model

    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.lora_target_modules,
    )
    model = get_peft_model(model, lora)
    return model


def save_adapter(model, output_dir: str | Path) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out))
    return out


def load_adapter(model, adapter_path: str | Path, *, is_trainable: bool = False):
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=is_trainable)


def write_run_metadata(
    output_dir: str | Path,
    *,
    cfg: ModelConfig,
    condition: str,
    seed: int,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write reproducibility metadata next to a checkpoint."""
    import platform

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    versions: dict[str, str] = {"python": platform.python_version()}
    for mod_name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[mod_name] = "not-imported"

    try:
        import torch

        versions["torch_cuda"] = str(torch.version.cuda)
        versions["cuda_available"] = str(torch.cuda.is_available())
        if torch.cuda.is_available():
            versions["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass

    payload = {
        "model_alias": cfg.alias,
        "model_id": cfg.model_id,
        "revision": cfg.revision,
        "condition": condition,
        "seed": seed,
        "model_config": asdict(cfg),
        "library_versions": versions,
        "hf_token_set": bool(hf_token()),
        **(extra or {}),
    }
    path = out / "run_metadata.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
