import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

# --- Basic scientific imports ---
import numpy as np
import pandas as pd

# --- Deep learning (uncomment if needed) ---
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, Dataset

# --- Visualization (uncomment if needed) ---
# import matplotlib.pyplot as plt
# import seaborn as sns


def _get_project_root() -> Path:
    project_root = os.environ.get("PROJECT_ROOT")
    if not project_root:
        print("Error: PROJECT_ROOT is not set. Run via run_slurm.sh.", file=sys.stderr)
        sys.exit(1)
    return Path(project_root)


def setup_logger(run_dir: Path, name: str = "experiment") -> logging.Logger:
    """Set up a logger writing to both console and run_dir/experiment.log."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(run_dir / "experiment.log")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


def load_config(exp_dir: Path) -> dict:
    """Load config.yml from the experiment directory."""
    config_path = exp_dir / "config.yml"
    if not config_path.exists():
        return {}
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def parse_args() -> argparse.Namespace:
    """Define CLI args for all variable dimensions (used in GRID_ARGS / RUN_COMMAND)."""
    parser = argparse.ArgumentParser()
    # Add one argument per swept dimension (required=True).
    # These must match GRID_ARGS entries in run_slurm.sh.
    # Example:
    #   parser.add_argument("--model", required=True)
    #   parser.add_argument("--seed",  required=True)
    parser.add_argument("--head", required=True, help="WSM head, e.g. dense:000003")
    parser.add_argument("--size", required=True, type=int, help="inference size (px)")
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    dataset_dir = Path(os.environ.get("DATASET_DIR", str(project_root / "data")))
    output_root = os.environ.get("OUTPUT_ROOT")

    args = parse_args()

    # Build variant_key from all variable dimensions so each config gets its own directory.
    # Include every arg that changes the result (model, seed, prompt, ...).
    # Example:
    #   model_short = args.model.replace("/", "-")
    #   variant_key = f"{model_short}__{args.seed}"
    variant_key = f"{args.head.replace(':', '-')}__{args.size}__demo"

    # Initialize output directory (exits immediately if already completed).
    # Written on OUTPUT_ROOT (scratch) when set; scripts/slurm_entry.sh
    # rsyncs it back to project_root/outputs/ at job end.
    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)

    config = load_config(Path(__file__).parent)
    seed: int = config.get("seed", 42)

    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)

    logger.info(f"Starting: {exp_name} / {variant_key}")
    logger.info(f"run_dir:     {run_dir}")
    logger.info(f"dataset_dir: {dataset_dir}")
    logger.info(f"seed:        {seed}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    import time

    import torch
    from PIL import Image

    wsm_root = project_root / "libraries" / "WSMs"
    sys.path.insert(0, str(wsm_root / "src"))
    from wsms import WSMPipeline

    steps: int = config["num_inference_steps"]
    n_runs: int = config["num_timed_runs"]
    image_path = project_root / config["input_images"][args.head]
    image = Image.open(image_path).convert("RGB")
    logger.info(f"input: {image_path} size={image.size}")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    pipe = WSMPipeline.from_pretrained(
        config["model_repo"],
        head=args.head,
        device="cuda:0",
        runtime_root=wsm_root / "WSM_v1",
    )
    load_sec = time.perf_counter() - t0
    logger.info(f"task: {pipe.task.as_dict()}")
    logger.info(f"load_sec: {load_sec:.1f}")

    run_secs = []
    for i in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = pipe(
            image,
            num_inference_steps=steps,
            seed=seed,
            size=args.size,
            output_dir=run_dir / f"infer_{i}",
        )
        torch.cuda.synchronize()
        run_secs.append(time.perf_counter() - t0)
        logger.info(f"run {i}: {run_secs[-1]:.2f} sec")
    out.images[0].save(run_dir / "result.png")
    image.save(run_dir / "input.png")

    results: dict = {
        "head": args.head,
        "size": args.size,
        "task": pipe.task.as_dict(),
        "gpu": torch.cuda.get_device_name(0),
        "load_sec": load_sec,
        "run_secs": run_secs,
        "sec_per_image_after_warmup": sum(run_secs[1:]) / max(1, len(run_secs) - 1),
        "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_vram_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "output_size": list(out.images[0].size),
    }
    logger.info(json.dumps(results, ensure_ascii=False, default=str))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False, default=str)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
