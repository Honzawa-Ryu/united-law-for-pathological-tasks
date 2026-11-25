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
    parser.add_argument("--cache", required=True, help="元のキャッシュ名（例: bracs_cls_10x）")
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
    variant_key = args.cache

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
    sys.path.insert(0, str(project_root / "data" / "pydeps"))  # h5py（0007 と同じ回避）
    import h5py
    import torch

    bracs = Path(config["bracs_dir"])
    src = project_root / config["cache_root"] / args.cache
    dst = project_root / config["cache_root"] / f"{args.cache}_uni"
    samples = pd.read_csv(src / "samples.csv")          # 0004 が書いた一覧（sample_id、split、座標など）
    field = int(json.loads(next((project_root / "outputs").glob(f"0004_*/{args.cache.split('_')[-1]}/results.json")).read_text())["field_px_40x"])
    logger.info(f"cache {args.cache}: {len(samples)} samples, field {field}px@40x")

    # --- UNI 特徴: WSM の入力と同じ範囲（ROI 中央の field 四方。小さい ROI は ROI の範囲）に中心が入るパッチの平均
    h5_by_slide = {f.stem: f for f in (bracs / config["feature_dir"]).glob("*/*/*/features_uni_v2/*.h5")}
    feats, fallback = {}, 0
    for slide, grp in samples.groupby("slide"):
        with h5py.File(h5_by_slide[slide], "r") as h:
            F = np.asarray(h["features"], dtype=np.float32)
            C = np.asarray(h["coords"], dtype=np.float64)
            ps = float(h["coords"].attrs.get("patch_size_level0", 512))
        centers = C + ps / 2.0
        for r in grp.itertuples():
            w, hh = min(r.width, field), min(r.height, field)
            cx, cy = r.x0 + r.width / 2.0, r.y0 + r.height / 2.0
            inside = (np.abs(centers[:, 0] - cx) < w / 2.0) & (np.abs(centers[:, 1] - cy) < hh / 2.0)
            if not inside.any():
                inside = np.zeros(len(C), dtype=bool)
                inside[int(((centers[:, 0] - cx) ** 2 + (centers[:, 1] - cy) ** 2).argmin())] = True
                fallback += 1
            v = F[inside].mean(axis=0)
            feats[r.sample_id] = v / np.linalg.norm(v)
    for split in ("train", "val", "test"):
        ids = samples.sample_id[samples.split == split].tolist()
        dst.mkdir(parents=True, exist_ok=True)
        np.savez(
            dst / f"uni_{split}.npz",
            audio_feat=np.stack([feats[i] for i in ids]).astype(np.float32),
            wsv_path=np.array(ids, dtype=str),
            frame_key=np.array(["0"] * len(ids), dtype=str),
        )

    # --- latent shards: sample_id だけ付け替えてコピー
    n_rows = {}
    for modality in config["modalities"]:
        out_dir = dst / "shards" / modality
        out_dir.mkdir(parents=True, exist_ok=True)
        for f in sorted((src / "shards" / modality).glob("shard_*.pt")):
            d = torch.load(f, map_location="cpu", weights_only=False)
            d["sample_ids"] = [f"{i}::0-0" for i in d["sample_ids"]]
            torch.save(d, out_dir / f.name)
            n_rows[modality] = n_rows.get(modality, 0) + len(d["sample_ids"])
    for name in ("samples.csv", "palette.json"):
        (dst / name).write_bytes((src / name).read_bytes())

    results: dict = {
        "source_cache": str(src), "cache_dir": str(dst), "field_px_40x": field,
        "feature_dim": int(len(next(iter(feats.values())))),
        "n_per_split": samples.split.value_counts().to_dict(),
        "rois_with_no_patch_center_inside": int(fallback),
        "shard_rows": n_rows,
    }
    logger.info(json.dumps(results, ensure_ascii=False))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
