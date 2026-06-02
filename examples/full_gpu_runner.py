"""Example scratch GPU runner for SpikeBudget public configs.

This is a compact, complete runner example for source checkout users. It is
not the archival research runner. It shows the public contract end to end:
load a checked-in config, build model weights from random initialization, train
on byte tokens, emit logs and summaries, and optionally save generated
checkpoint bytes outside git.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


SAMPLE_TEXT = (
    "SpikeBudget public runner example. "
    "This byte stream is intentionally tiny and only exercises the scratch "
    "training contract. Replace it with a public dataset slice for real runs. "
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dense_low_lr_scratch_screen.yaml"))
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path(os.environ.get("SPIKEBUDGET_ARTIFACT_DIR", "artifacts/runs/example_gpu_runner")),
        help="Output directory for generated run artifacts; default stays under ignored artifacts/runs/.",
    )
    parser.add_argument("--text-file", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--eval-every", type=int, default=None)
    parser.add_argument("--save-checkpoint", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from spikebudget.config import load_repro_config
        from spikebudget.training import fixed_eval_starts, scheduled_lr
    except ModuleNotFoundError as exc:
        raise SystemExit("Install the SpikeBudget scaffold first with python -m pip install -e '.[dev]'.") from exc

    try:
        import torch
        from torch import nn
        import torch.nn.functional as functional
    except ModuleNotFoundError as exc:
        raise SystemExit("This runner example requires PyTorch. Install a CUDA PyTorch build for GPU runs.") from exc

    config = load_repro_config(args.config)
    run = config["run"]
    model_config = config["model"]
    optimizer_config = config["optimizer"]

    device = select_device(args.device, torch)
    seed = int(os.environ.get("SPIKEBUDGET_SEED", "1234"))
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.config, artifact_dir / "config.yaml")
    write_hardware(artifact_dir / "hardware.json", torch, device, config)

    max_steps = int(args.max_steps if args.max_steps is not None else first_positive(run["max_steps"]))
    batch = args.batch if args.batch is not None else int(run.get("batch", 8))
    seq_len = args.seq_len if args.seq_len is not None else int(model_config.get("sequence_length", 128))
    d_model = args.d_model if args.d_model is not None else int(model_config.get("d_model", 128))
    layers = args.layers if args.layers is not None else int(model_config.get("layers", 2))
    timesteps = int(model_config.get("timesteps", 4))
    eval_every = int(args.eval_every if args.eval_every is not None else run.get("eval_every", max(1, max_steps // 5)))
    eval_iters = int(run.get("eval_iters", 1))
    base_lr = float(optimizer_config.get("lr", first_positive(optimizer_config.get("lr_grid", [0.001]))))

    tokens = load_tokens(args.text_file, torch).to(device)
    if tokens.numel() < seq_len + 2:
        raise SystemExit("text source is shorter than requested sequence length")
    eval_starts = fixed_eval_starts(tokens.numel(), seq_len, batch, eval_iters, seed + 1)

    model = ScratchByteTemporalLM(vocab_size=256, d_model=d_model, layers=layers, timesteps=timesteps, nn=nn).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)

    log_path = artifact_dir / "train.jsonl"
    best_metric = float("inf")
    best_checkpoint_path: Path | None = None
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        for step in range(1, max_steps + 1):
            model.train()
            lr = scheduled_lr(base_lr, "cosine", 0.1, step, warmup_steps=max(1, max_steps // 20), max_steps=max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr

            inputs, targets = sample_batch(tokens, batch=batch, seq_len=seq_len, torch=torch)
            logits = model(inputs)
            loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            if step == 1 or step % eval_every == 0 or step == max_steps:
                eval_bpb = evaluate(
                    model,
                    tokens,
                    eval_starts=eval_starts,
                    batch=batch,
                    seq_len=seq_len,
                    torch=torch,
                    functional=functional,
                )
                record = {
                    "step": step,
                    "train_bpb": loss_to_bpb(float(loss.detach().cpu())),
                    "eval_bpb": eval_bpb,
                    "lr": lr,
                    "device": str(device),
                    "tokens_per_step": batch * seq_len,
                    "elapsed_s": round(time.time() - started, 6),
                    "peak_gb": peak_gb(torch, device),
                }
                log.write(json.dumps(record, sort_keys=True) + "\n")
                log.flush()
                if eval_bpb < best_metric:
                    best_metric = eval_bpb
                    if args.save_checkpoint or bool(run.get("save_best_checkpoint", False)):
                        best_checkpoint_path = checkpoint_dir / "best_scratch_checkpoint.pt"
                        torch.save(
                            {
                                "model_state_dict": model.state_dict(),
                                "config": config,
                                "step": step,
                                "eval_bpb": eval_bpb,
                                "training_origin": "random initialization train-from-scratch",
                            },
                            best_checkpoint_path,
                        )

    summary = {
        "status": "complete",
        "config": str(args.config),
        "device": str(device),
        "seed": seed,
        "max_steps": max_steps,
        "best_eval_bpb": best_metric,
        "model": {
            "vocab_size": 256,
            "d_model": d_model,
            "layers": layers,
            "timesteps": timesteps,
            "parameter_count": parameter_count,
            "note": "compact example proxy; actual dimensions are recorded here and may differ from the nominal config parameter target",
        },
        "checkpoint": str(best_checkpoint_path) if best_checkpoint_path else None,
        "checkpoint_policy": "generated scratch output; keep outside git and publish checksum with release bundles",
        "artifact_dir": str(artifact_dir),
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def first_positive(value: Any) -> float:
    if isinstance(value, list):
        return float(value[0])
    return float(value)


def select_device(requested: str, torch: Any) -> Any:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA was requested but torch.cuda.is_available() is false")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_tokens(text_file: Path | None, torch: Any) -> Any:
    if text_file is None:
        text = SAMPLE_TEXT * 2048
    else:
        text = text_file.read_text(encoding="utf-8")
    data = list(text.encode("utf-8"))
    return torch.tensor(data, dtype=torch.long)


def sample_batch(tokens: Any, batch: int, seq_len: int, torch: Any) -> tuple[Any, Any]:
    max_start = tokens.numel() - seq_len - 1
    starts = torch.randint(0, max_start, (batch,), device=tokens.device).tolist()
    return gather_batch(tokens, starts, seq_len, torch)


def gather_batch(tokens: Any, starts: Any, seq_len: int, torch: Any) -> tuple[Any, Any]:
    inputs = torch.stack([tokens[start : start + seq_len] for start in starts])
    targets = torch.stack([tokens[start + 1 : start + seq_len + 1] for start in starts])
    return inputs, targets


def evaluate(model: Any, tokens: Any, eval_starts: list[int], batch: int, seq_len: int, torch: Any, functional: Any) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for offset in range(0, len(eval_starts), batch):
            starts = eval_starts[offset : offset + batch]
            inputs, targets = gather_batch(tokens, starts, seq_len, torch)
            logits = model(inputs)
            loss = functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="sum")
            total_loss += float(loss.detach().cpu())
            total_tokens += int(targets.numel())
    return loss_to_bpb(total_loss / total_tokens)


def loss_to_bpb(loss: float) -> float:
    return loss / 0.6931471805599453


def peak_gb(torch: Any, device: Any) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.max_memory_allocated(device)) / (1024**3)


def write_hardware(path: Path, torch: Any, device: Any, config: dict[str, Any]) -> None:
    payload = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": getattr(torch, "__version__", "unknown"),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "config_name": config["name"],
        "claim_scope": config["claim_scope"],
    }
    if device.type == "cuda":
        payload["gpu_name"] = torch.cuda.get_device_name(device)
        payload["cuda_version"] = getattr(torch.version, "cuda", None)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ScratchByteTemporalLM:
    """Tiny temporal byte LM used by the runner example."""

    def __new__(cls, vocab_size: int, d_model: int, layers: int, timesteps: int, nn: Any) -> Any:
        class _Model(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.blocks = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.LayerNorm(d_model),
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Linear(d_model * 4, d_model),
                        )
                        for _ in range(layers)
                    ]
                )
                self.temporal = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(timesteps)])
                self.norm = nn.LayerNorm(d_model)
                self.head = nn.Linear(d_model, vocab_size)

            def forward(self, inputs: Any) -> Any:
                hidden = self.embedding(inputs)
                for block in self.blocks:
                    hidden = hidden + block(hidden)
                state = hidden.new_zeros(hidden.shape)
                for temporal_layer in self.temporal:
                    state = torch_tanh(temporal_layer(hidden + state))
                return self.head(self.norm(state))

        return _Model()


def torch_tanh(value: Any) -> Any:
    return value.tanh()


if __name__ == "__main__":
    raise SystemExit(main())
