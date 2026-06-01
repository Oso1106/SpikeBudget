"""Small deterministic helpers shared by public reproduction configs."""

from __future__ import annotations

import math
import random


def fixed_eval_starts(num_tokens: int, seq_len: int, batch: int, iters: int, seed: int) -> list[int]:
    """Return deterministic token-window starts for fixed validation."""

    max_start = int(num_tokens) - int(seq_len) - 1
    if max_start <= 0:
        raise ValueError("tokenized split is shorter than seq_len")
    total = max(1, int(batch) * int(iters))
    rng = random.Random(int(seed))
    if total == 1:
        return [rng.randrange(max_start)]
    stride = max(1, max_start // total)
    offset = rng.randrange(max_start)
    return [(offset + i * stride) % max_start for i in range(total)]


def scheduled_lr(
    base_lr: float,
    schedule: str,
    min_lr_ratio: float,
    step: int,
    warmup_steps: int,
    max_steps: int,
) -> float:
    """Return the learning rate for a constant or warmup-cosine schedule."""

    if schedule == "constant":
        return float(base_lr)
    if schedule != "cosine":
        raise ValueError(schedule)

    warmup_steps = max(0, int(warmup_steps))
    if warmup_steps > 0 and step <= warmup_steps:
        return float(base_lr) * (float(step) / warmup_steps)

    denom = max(1, int(max_steps) - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / denom))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return float(base_lr) * (float(min_lr_ratio) + (1.0 - float(min_lr_ratio)) * cosine)
