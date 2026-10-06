# Feasibility harness — go/no-go table

Fill after CARC smokes. Scope: one tiny dataset × three base models × Base / SFT / GRPO / SFT→GRPO × seed `566`.

**Dataset:** `target_expr` (`data/splits/`, seed 566)  
**Revisions:** Qwen `____` / Gemma `____` / AFM `____`  
**GPU:** `____` (SKU TBD) · CUDA `____` · torch `____`

| Model | Load | Base eval | SFT | GRPO | SFT→GRPO | Peak VRAM | GPU-h (smoke) | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `Qwen/Qwen3.5-4B-Base` | | | | | | | | |
| `google/gemma-3n-E4B` | | | | | | | | |
| `arcee-ai/AFM-4.5B-Base` | | | | | | | | |

Mark each cell: `go` / `no-go` / `blocked` (reason in Notes).

## Smoke pass criteria

- Load + 1 forward/generate without OOM
- SFT: 20–50 steps; adapter reload works
- GRPO: 10–20 updates, 4 completions/prompt, short max completion, no OOM
- Full 4-condition smoke for one model fits a single session (target ≲2–4 GPU-hours)
- Peak VRAM logged with ≳10% headroom
