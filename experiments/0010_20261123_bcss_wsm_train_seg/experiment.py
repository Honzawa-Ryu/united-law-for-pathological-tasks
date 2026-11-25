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
    parser.add_argument("--cache", required=True, help="cache_root の下のキャッシュ名（例: bracs_cls_10x）")
    parser.add_argument("--max-steps", required=True, type=int)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--uni", required=True, choices=["none", "attention", "modulation"],
                        help="UNI の ROI 特徴を条件に足す方法（none 以外はキャッシュ名が *_uni であること）")
    parser.add_argument("--categorical", required=True, choices=["off", "prototype_ce", "balanced_ce", "margin_global", "combined"])
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
    # 既定（バッチ 1、補助損失なし）は従来の名前のままにして、既存の run ディレクトリから再開できるようにする
    variant_key = f"{args.cache}__steps{args.max_steps}"
    if args.batch_size != 1:
        variant_key += f"__bs{args.batch_size}"
    if args.categorical != "off":
        variant_key += f"__cat-{args.categorical}"
    if args.uni != "none":
        variant_key += f"__uni-{args.uni}"

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
    import subprocess
    import threading
    import time

    from huggingface_hub import snapshot_download

    wsm = project_root / "libraries" / "WSMs" / "WSM_v1"
    cache_dir = project_root / config["cache_root"] / args.cache
    wsm_run = project_root / config["run_root"] / exp_name / variant_key
    wsm_run.mkdir(parents=True, exist_ok=True)
    target = config["target_modality"]
    head = int(config["channel_head_index"])

    # custom 設定に目標モダリティと head を足したものを run ディレクトリに置く（libraries/ は git 管理外）
    mod = json.loads((wsm / "configs" / "wsg_v1_modalities_custom.json").read_text())
    mod.setdefault("modality_specs", {})[target] = {
        "storage": "image_like", "channels": 3, "layout": "HWC", "value_range": [0.0, 255.0],
        "encoding": "rgb_0_255_to_minus1_1", "description": "BCSS tissue palette (5 classes + ignore)",
    }
    mod.setdefault("normalization", {}).setdefault("modalities", {})[target] = "linear"
    mod["wsg_v1"][f"Channel_Head_{head:06d}"] = {
        "task_type": "estimation_or_reconstruction", "pair_exists": True, "target_density": "dense",
        "condition": "0-2" if args.uni == "none" else ["Audio_AST", "CurrentFrame_RGB_Condition"],
        "condition_description": "0-2: H&E RGB patch" if args.uni == "none"
        else "Audio_AST: UNI2-h ROI embedding as one global token | CurrentFrame_RGB_Condition: H&E RGB latent",
        "target": target, "target_description": f"{target}: per-pixel tissue class palette",
    }
    mod_path = wsm_run / "modalities_custom.json"
    mod_path.write_text(json.dumps(mod, indent=2))

    base = snapshot_download(config["base_model_repo"], local_files_only=True)
    states = sorted((wsm_run / "state").glob("step_*")) if (wsm_run / "state").exists() else []
    cmd = [
        sys.executable, "scripts/train_TunnelingNet_Flux2_4B_wsg.py",
        "--model-path", base, "--modality-config", str(mod_path),
        "--dense-latent-cache-dir", str(cache_dir), "--channel-head-index", str(head),
        "--target-modality", target,
        "--condition-modalities", "0-2" if args.uni == "none" else "Audio_AST,CurrentFrame_RGB_Condition",
        "--channel-head-rank", str(config["channel_head_rank"]),
        "--channel-head-alpha", str(config["channel_head_alpha"]),
        "--max-steps", str(args.max_steps), "--learning-rate", str(config["learning_rate"]),
        "--batch-size", str(args.batch_size), "--num-workers", str(config["num_workers"]),
        "--eval-every-steps", str(config["eval_every_steps"]), "--viz-every-steps", str(config["viz_every_steps"]),
        "--val-loss-batches", str(config["val_loss_batches"]),
        "--early-stop-seconds", str(config["early_stop_seconds"]),
        "--preview-num-samples", str(config["preview_num_samples"]), "--preview-split", "val",
        "--categorical-loss-mode", args.categorical,
        "--categorical-num-clusters", str(config["categorical_num_clusters"]), "--seed", str(seed),
        "--output-dir", str(wsm_run),
    ]
    if args.uni != "none":
        # 特徴ベクトル条件（著者の音声条件の経路）。val の NPZ を省くと train が検証に使われるので必ず渡す
        cmd += [
            "--audio-dense-condition-train", str(cache_dir / "uni_train.npz"),
            "--audio-dense-condition-val", str(cache_dir / "uni_val.npz"),
            "--audio-condition-dim", str(config["uni_dim"]),
            "--audio-conditioning-mode", args.uni, "--audio-global-condition",
        ]
    if states:
        cmd += ["--resume-from-checkpoint", str(states[-1])]
        logger.info(f"resuming from {states[-1]}")
    env = dict(os.environ, PYTHONPATH=str(wsm), WSG_READER_VISUALIZATION="0")
    logger.info("cmd: " + " ".join(cmd))

    peak = {"mib": 0}
    stop = threading.Event()

    def poll_gpu() -> None:
        while not stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=20,
                ).stdout.split()
                peak["mib"] = max(peak["mib"], *[int(v) for v in out])
            except Exception:
                pass
            stop.wait(config["gpu_poll_seconds"])

    threading.Thread(target=poll_gpu, daemon=True).start()
    t0 = time.perf_counter()
    with open(wsm_run / f"train_{os.environ.get('SLURM_JOB_ID', 'local')}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=wsm, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    wall = time.perf_counter() - t0
    stop.set()

    ckpts = sorted(p.name for p in (wsm_run / "ckpt").glob("*")) if (wsm_run / "ckpt").exists() else []
    results: dict = {
        "cache": args.cache,
        "max_steps": args.max_steps,
        "returncode": rc,
        "resumed_from": str(states[-1]) if states else None,
        "wall_seconds": wall,
        "peak_gpu_memory_mib": peak["mib"],
        "wsm_run_dir": str(wsm_run),
        "checkpoints": ckpts,
    }
    logger.info(json.dumps(results, ensure_ascii=False))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    if rc != 0:
        logger.error(f"trainer exited with {rc}; see {wsm_run}")
        sys.exit(1)
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
