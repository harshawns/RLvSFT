"""
SFT training entrypoint (WIP).

Imports the same LoRA / model ids as GRPO so SFT→GRPO adapters match.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from configs import MODEL_ID, build_lora_config

__all__ = ["MODEL_ID", "build_lora_config"]
