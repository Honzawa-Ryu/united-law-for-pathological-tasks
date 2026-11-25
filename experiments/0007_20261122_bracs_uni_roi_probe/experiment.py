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
    parser.add_argument("--field", required=True, type=int,
                        help="ROI 中央から使う視野（40 倍の px）。0 なら ROI の外接矩形全体")
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
    variant_key = "roi" if args.field == 0 else f"field{args.field}"

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
    sys.path.insert(0, str(project_root / "data" / "pydeps"))  # h5py（run_slurm.sh が入れる）
    import h5py
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.preprocessing import StandardScaler

    bracs = Path(config["bracs_dir"])
    classes = list(config["classes"])
    manifest = pd.concat(
        [pd.read_csv(bracs / "roi_png" / f"manifest_{sp}.csv") for sp in ("train", "val", "test")], ignore_index=True
    )
    h5_by_slide = {f.stem: f for f in (bracs / config["feature_dir"]).glob("*/*/*/features_uni_v2/*.h5")}
    logger.info(f"ROIs {len(manifest)}, slides with features {len(h5_by_slide)}")

    feats, n_patches, fallback = {}, [], 0
    for slide, grp in manifest.groupby("slide"):
        with h5py.File(h5_by_slide[slide], "r") as h:
            F = np.asarray(h["features"], dtype=np.float32)
            C = np.asarray(h["coords"], dtype=np.float64)
            ps = float(h["coords"].attrs.get("patch_size_level0", 512))
        centers = C + ps / 2.0
        for r in grp.itertuples():
            if args.field == 0:
                x0, y0, x1, y1 = r.x0, r.y0, r.x0 + r.width, r.y0 + r.height
            else:  # WSM の入力と同じ: ROI 中央の field 四方（ROI が小さければ ROI の範囲まで）
                w, h_ = min(r.width, args.field), min(r.height, args.field)
                cx, cy = r.x0 + r.width / 2.0, r.y0 + r.height / 2.0
                x0, y0, x1, y1 = cx - w / 2.0, cy - h_ / 2.0, cx + w / 2.0, cy + h_ / 2.0
            inside = (centers[:, 0] >= x0) & (centers[:, 0] < x1) & (centers[:, 1] >= y0) & (centers[:, 1] < y1)
            if not inside.any():   # パッチ中心が 1 つも入らない小さい ROI は、最も近いパッチ 1 枚で代用
                d = (centers[:, 0] - (x0 + x1) / 2) ** 2 + (centers[:, 1] - (y0 + y1) / 2) ** 2
                inside = np.zeros(len(C), dtype=bool)
                inside[int(d.argmin())] = True
                fallback += 1
            feats[r.file] = F[inside].mean(axis=0)
            n_patches.append(int(inside.sum()))
    X = np.stack([feats[f] for f in manifest.file])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = manifest.label.to_numpy()
    idx = {sp: (manifest.split == sp).to_numpy() for sp in ("train", "val", "test")}
    scaler = StandardScaler().fit(X[idx["train"]])
    Xs = scaler.transform(X)

    def metrics(true, pred) -> dict:
        return {
            "accuracy": float(accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=classes, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true, pred, labels=classes, average="weighted", zero_division=0)),
            "per_class_f1": dict(zip(classes, f1_score(true, pred, labels=classes, average=None, zero_division=0).round(4).tolist())),
        }

    sweep, best = [], None
    for c in config["c_grid"]:
        clf = LogisticRegression(C=c, max_iter=config["max_iter"], random_state=seed).fit(Xs[idx["train"]], y[idx["train"]])
        m = metrics(y[idx["val"]], clf.predict(Xs[idx["val"]]))
        sweep.append({"C": c, "val_macro_f1": m["macro_f1"], "val_accuracy": m["accuracy"]})
        if best is None or m["macro_f1"] > best[1]["macro_f1"]:
            best = (c, m, clf)
    c_best, val_m, clf = best
    pred_test = clf.predict(Xs[idx["test"]])
    manifest.assign(pred=clf.predict(Xs))[["file", "split", "slide", "label", "pred"]].to_csv(run_dir / "predictions.csv", index=False)

    results: dict = {
        "field_px_40x": args.field,
        "n": {sp: int(v.sum()) for sp, v in idx.items()},
        "feature_dim": int(X.shape[1]),
        "patches_per_roi": {"mean": float(np.mean(n_patches)), "median": float(np.median(n_patches)), "min": int(np.min(n_patches))},
        "rois_with_no_patch_center_inside": int(fallback),
        "C_sweep": sweep, "C_best": c_best,
        "val": val_m, "test": metrics(y[idx["test"]], pred_test),
        "test_confusion_matrix_rows_true_cols_pred": confusion_matrix(y[idx["test"]], pred_test, labels=classes).tolist(),
        "classes": classes,
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
