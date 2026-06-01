import pytest

from spikebudget.training import fixed_eval_starts, scheduled_lr


def test_fixed_eval_starts_are_deterministic_and_in_range():
    starts = fixed_eval_starts(num_tokens=1000, seq_len=128, batch=2, iters=3, seed=7)

    assert starts == fixed_eval_starts(1000, 128, 2, 3, 7)
    assert len(starts) == 6
    assert all(0 <= start < 871 for start in starts)


def test_fixed_eval_starts_reject_short_split():
    with pytest.raises(ValueError, match="shorter than seq_len"):
        fixed_eval_starts(num_tokens=32, seq_len=64, batch=1, iters=1, seed=1)


def test_scheduled_lr_constant_and_cosine_warmup():
    assert scheduled_lr(0.1, "constant", 0.1, 50, 10, 100) == pytest.approx(0.1)
    assert scheduled_lr(0.1, "cosine", 0.1, 5, 10, 100) == pytest.approx(0.05)
    assert scheduled_lr(0.1, "cosine", 0.1, 100, 10, 100) == pytest.approx(0.01)


def test_scheduled_lr_rejects_unknown_schedule():
    with pytest.raises(ValueError, match="linear"):
        scheduled_lr(0.1, "linear", 0.1, 1, 0, 10)
