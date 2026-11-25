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
    parser.add_argument("--magnification", required=True, type=int, choices=[40, 20, 10, 5])
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
    variant_key = f"{args.magnification}x"

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
    import torch
    from diffusers import AutoencoderKLFlux2
    from huggingface_hub import snapshot_download
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    roi_dir = Path(config["bracs_roi_dir"])
    tile = config["tile_px"]
    field = tile * 40 // args.magnification
    palette = {k: np.array(v, dtype=np.uint8) for k, v in config["palette"].items()}
    target_name = config["target_modality"]
    out_root = project_root / config["cache_root"] / f"bracs_cls_{args.magnification}x"

    manifest = pd.concat(
        [pd.read_csv(roi_dir / f"manifest_{s}.csv") for s in ("train", "val", "test")], ignore_index=True
    )
    manifest = manifest.iloc[rng.permutation(len(manifest))].reset_index(drop=True)  # 各 shard に全 split が入るように
    logger.info(f"ROIs: {len(manifest)}  field={field}px@40x -> {tile}px")

    base = snapshot_download(config["base_model_repo"], local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()

    def load_tile(file: str) -> np.ndarray:
        img = Image.open(roi_dir / file).convert("RGB")
        canvas = Image.new("RGB", (field, field), (255, 255, 255))
        w, h = min(img.width, field), min(img.height, field)
        x0, y0 = (img.width - w) // 2, (img.height - h) // 2
        canvas.paste(img.crop((x0, y0, x0 + w, y0 + h)), ((field - w) // 2, (field - h) // 2))
        if field != tile:
            canvas = canvas.resize((tile, tile), Image.LANCZOS)
        return np.asarray(canvas)

    def encode(batch_u8: np.ndarray) -> torch.Tensor:
        # 著者の前処理と同じ: [-1,1] に写し、VAE の事後分布から sample、スケール・パッチ化なしで [N,C,1,h,w]
        x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to("cuda")
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample()
        return z.unsqueeze(2).to(torch.bfloat16).cpu()

    lat_rgb, lat_tgt, coverage = [], [], []
    bs = config["batch_size"]
    for i in range(0, len(manifest), bs):
        rows = manifest.iloc[i:i + bs]
        tiles = np.stack([load_tile(f) for f in rows.file])
        targets = np.stack([np.broadcast_to(palette[lab], (tile, tile, 3)) for lab in rows.label])
        lat_rgb.append(encode(tiles))
        lat_tgt.append(encode(np.ascontiguousarray(targets)))
        coverage.extend(float((t.mean(axis=2) < 220).mean()) for t in tiles)
        if i % (bs * 20) == 0:
            logger.info(f"encoded {i + len(rows)}/{len(manifest)}")
            if i == 0:
                ex = run_dir / "examples"
                ex.mkdir(exist_ok=True)
                for j in range(min(8, len(rows))):
                    Image.fromarray(np.concatenate([tiles[j], targets[j]], axis=1)).save(
                        ex / f"{j:02d}_{rows.label.iloc[j]}.jpg", quality=90
                    )
    lat_rgb, lat_tgt = torch.cat(lat_rgb), torch.cat(lat_tgt)
    logger.info(f"latents: {tuple(lat_rgb.shape)} {lat_rgb.dtype}")

    sample_ids = [f"BRACS/{f}" for f in manifest.file]
    sources = [
        {"dataset_name": "BRACS", "split": r.split, "video_id": Path(r.file).stem, "video_name": r.slide,
         "frame_index": 0, "label": r.label}
        for r in manifest.itertuples()
    ]
    n_shards = config["num_shards"]
    for modality, latents in (("0-2", lat_rgb), (target_name, lat_tgt)):
        d = out_root / "shards" / modality.replace("-", "_")
        d.mkdir(parents=True, exist_ok=True)
        for sid in range(n_shards):
            idx = list(range(sid, len(manifest), n_shards))
            torch.save(
                {
                    "modality": modality,
                    "num_samples": len(idx),
                    "latents": latents[idx].clone(),
                    "sample_ids": [sample_ids[k] for k in idx],
                    "sample_sources": [sources[k] for k in idx],
                    "video_ids": [sources[k]["video_id"] for k in idx],
                    "frame_indices": [0 for _ in idx],
                },
                d / f"shard_{sid:05d}_of_{n_shards:05d}.pt",
            )
    manifest.assign(sample_id=sample_ids, tissue_fraction=coverage).to_csv(out_root / "samples.csv", index=False)
    (out_root / "palette.json").write_text(json.dumps(config["palette"], indent=2))

    results: dict = {
        "magnification": args.magnification,
        "field_px_40x": field,
        "cache_dir": str(out_root),
        "n_samples": int(len(manifest)),
        "per_split": manifest.split.value_counts().to_dict(),
        "latent_shape": list(lat_rgb.shape),
        "latent_dtype": str(lat_rgb.dtype),
        "mean_tissue_fraction": float(np.mean(coverage)),
        "frac_rois_smaller_than_field": float(((manifest.width < field) | (manifest.height < field)).mean()),
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
