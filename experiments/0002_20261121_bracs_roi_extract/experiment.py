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
    parser.add_argument("--split", required=True, choices=["train", "val", "test"])
    return parser.parse_args()


def to_label(name: str, prefix_map: list) -> str:
    key = name.lower().replace("-", " ").strip()
    for prefix, label in prefix_map:
        if key.startswith(prefix):
            return label
    raise ValueError(f"unknown annotation class: {name!r}")


def extract_slide(job: tuple) -> dict:
    """1 枚の WSI から全 ROI（注釈の外接矩形、レベル 0）を PNG に書く。"""
    rel, config, split = job
    import openslide
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    bracs_dir = Path(config["bracs_dir"])
    slide_id = Path(rel).stem
    svs = bracs_dir / config["wsi_dir"] / rel
    geo = (bracs_dir / config["geojson_dir"] / Path(rel).relative_to("BRACS_WSI")).with_suffix(".geojson")
    roi_root = bracs_dir / config["roi_dir"]
    done = roi_root / "_done" / split / f"{slide_id}.json"
    out = {"slide": slide_id, "status": "done", "error": None, "rows": []}
    if done.exists():
        out["rows"] = json.loads(done.read_text())
        return out
    if not svs.exists():
        out["status"] = "wsi_missing"
        return out
    try:
        feats = json.loads(geo.read_text())["features"]
        slide = openslide.OpenSlide(str(svs))
        W, H = slide.dimensions
        mpp = slide.properties.get(openslide.PROPERTY_NAME_MPP_X)
        counter: dict = {}
        rows = []
        for ft in feats:
            name = ft["properties"]["classification"]["name"]
            label = to_label(name, config["class_prefix_map"])
            g = ft["geometry"]
            polys = [g["coordinates"]] if g["type"] == "Polygon" else g["coordinates"]
            pts = [q for poly in polys for q in poly[0]]
            x0 = max(0, int(np.floor(min(q[0] for q in pts))))
            y0 = max(0, int(np.floor(min(q[1] for q in pts))))
            x1 = min(W, int(np.ceil(max(q[0] for q in pts))))
            y1 = min(H, int(np.ceil(max(q[1] for q in pts))))
            counter[label] = counter.get(label, 0) + 1
            fname = f"{slide_id}_{label}_{counter[label]}.png"
            dst = roi_root / split / label / fname
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = slide.read_region((x0, y0), 0, (x1 - x0, y1 - y0)).convert("RGB")
            tmp = dst.with_suffix(".tmp.png")
            img.info.pop("icc_profile", None)  # 約120KBのiCCPが付くとPILで読めなくなる
            img.save(tmp, icc_profile=None)
            tmp.rename(dst)
            rows.append({
                "split": split, "slide": slide_id, "wsi_label": Path(rel).parent.name,
                "label": label, "raw_name": name, "geometry": g["type"], "n_vertices": len(pts),
                "x0": x0, "y0": y0, "width": x1 - x0, "height": y1 - y0, "mpp": mpp,
                "roi_id": ft.get("id"), "file": str(dst.relative_to(roi_root)),
            })
        slide.close()
        done.parent.mkdir(parents=True, exist_ok=True)
        done.write_text(json.dumps(rows))
        out["rows"] = rows
        if config["delete_wsi_after"]:
            svs.unlink()
    except Exception as e:  # 1 枚の失敗で全体を止めない。最後にまとめて失敗扱いにする
        out["status"] = "failed"
        out["error"] = repr(e)
    return out


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
    variant_key = args.split

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
    from multiprocessing import Pool

    bracs_dir = Path(config["bracs_dir"])
    slides = (bracs_dir / "lists" / f"annotated_wsi_{args.split}.txt").read_text().split()
    logger.info(f"slides: {len(slides)}")
    jobs = [(rel, config, args.split) for rel in slides]
    with Pool(config["num_workers"]) as pool:
        per_slide = pool.map(extract_slide, jobs)

    rows = [r for res in per_slide for r in res["rows"]]
    manifest = pd.DataFrame(rows)
    roi_root = bracs_dir / config["roi_dir"]
    manifest.to_csv(roi_root / f"manifest_{args.split}.csv", index=False)
    manifest.to_csv(run_dir / f"manifest_{args.split}.csv", index=False)

    status = pd.Series([res["status"] for res in per_slide]).value_counts().to_dict()
    errors = [f"{res['slide']}: {res['error']}" for res in per_slide if res["error"]]
    for e in errors:
        logger.error(e)
    results: dict = {
        "split": args.split,
        "n_slides": len(slides),
        "slide_status": status,
        "n_rois": int(len(manifest)),
        "rois_per_class": manifest["label"].value_counts().to_dict() if len(manifest) else {},
        "errors": errors,
    }
    logger.info(json.dumps(results, ensure_ascii=False))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    # 取れていない WSI や失敗があるうちは completed にしない（再投入で続きから）。
    if set(status) - {"done"}:
        logger.error(f"incomplete: {status}")
        sys.exit(1)
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
