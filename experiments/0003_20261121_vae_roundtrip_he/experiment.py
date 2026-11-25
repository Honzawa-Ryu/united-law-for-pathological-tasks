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
    parser.add_argument("--magnification", required=True, type=int, choices=[40, 20, 10])
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
    from skimage.color import rgb2hed
    from skimage.filters import threshold_otsu
    from skimage.measure import label
    from skimage.metrics import peak_signal_noise_ratio, structural_similarity

    Image.MAX_IMAGE_PIXELS = None
    torch.manual_seed(seed)
    roi_dir = Path(config["bracs_roi_dir"])
    tile = config["tile_px"]
    src = tile * 40 // args.magnification  # 40 倍画像から切る一辺（20 倍なら 1024 → 512 に縮小）
    manifest = pd.read_csv(roi_dir / f"manifest_{config['split']}.csv")
    manifest = manifest[(manifest.width >= src) & (manifest.height >= src)].reset_index(drop=True)
    logger.info(f"ROIs large enough for {src}px crop: {len(manifest)}")

    base = snapshot_download(config["base_model_repo"], local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()

    def roundtrip(batch_u8: np.ndarray) -> np.ndarray:
        x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to("cuda")
        with torch.no_grad():
            z = vae.encode(x).latent_dist.mode()
            y = vae.decode(z).sample
        return y.clamp(-1, 1).add(1).mul(127.5).round().permute(0, 2, 3, 1).byte().cpu().numpy()

    def dice(a: np.ndarray, b: np.ndarray) -> float:
        d = a.sum() + b.sum()
        return float(2 * (a & b).sum() / d) if d else 1.0

    rows, tiles, meta = [], [], []
    for r in manifest.itertuples():
        img = Image.open(roi_dir / r.file).convert("RGB")
        x0, y0 = (img.width - src) // 2, (img.height - src) // 2
        t = img.crop((x0, y0, x0 + src, y0 + src))
        if src != tile:
            t = t.resize((tile, tile), Image.LANCZOS)
        a = np.asarray(t)
        if (a.mean(axis=2) < 220).mean() < config["min_tissue_fraction"]:
            continue
        tiles.append(a)
        meta.append((r.file, r.label))
    logger.info(f"tiles after tissue filter: {len(tiles)}")

    ex_dir = run_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    bs = config["batch_size"]
    for i in range(0, len(tiles), bs):
        orig = np.stack(tiles[i:i + bs])
        # 核マスク: ヘマトキシリン成分を大津の方法で二値化（正解注釈ではなく代用指標）
        h_orig = np.stack([rgb2hed(o)[..., 0] for o in orig])
        thr = [threshold_otsu(h) for h in h_orig]
        m_orig = np.stack([h > t for h, t in zip(h_orig, thr)])
        mask_img = np.repeat((m_orig[..., None] * 255).astype(np.uint8), 3, axis=3)
        rec = roundtrip(orig)
        mask_rec = roundtrip(mask_img)
        for j in range(len(orig)):
            h_rec = rgb2hed(rec[j])[..., 0]
            m_rec = h_rec > thr[j]                      # 元画像と同じしきい値
            mm = mask_rec[j].mean(axis=2) > 127.5       # マスク画像の往復後を二値化
            n_o, n_m = int(label(m_orig[j]).max()), int(label(mm).max())
            rows.append({
                "file": meta[i + j][0], "label": meta[i + j][1],
                "psnr": peak_signal_noise_ratio(orig[j], rec[j], data_range=255),
                "ssim": structural_similarity(orig[j], rec[j], channel_axis=2, data_range=255),
                "hematoxylin_corr": float(np.corrcoef(h_orig[j].ravel(), h_rec.ravel())[0, 1]),
                "nuclei_dice_image": dice(m_orig[j], m_rec),          # 入力側: 画像の往復で核領域が保たれるか
                "mask_dice": dice(m_orig[j], mm),                     # 出力側: マスク自体の往復
                "mask_frac_pixels_changed": float((m_orig[j] != mm).mean()),
                "n_components_orig": n_o, "n_components_mask_rec": n_m,
                "nuclei_fraction": float(m_orig[j].mean()),
            })
            k = i + j
            if k < config["num_example_images"]:
                panel = np.concatenate([orig[j], rec[j], mask_img[j], mask_rec[j]], axis=1)
                Image.fromarray(panel).save(ex_dir / f"{k:02d}_{meta[k][1]}.jpg", quality=92)

    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_tile.csv", index=False)
    num = df.drop(columns=["file", "label"])
    results: dict = {
        "magnification": args.magnification,
        "mpp_approx": 0.25 * 40 / args.magnification,
        "n_tiles": int(len(df)),
        "mean": num.mean().round(4).to_dict(),
        "p05": num.quantile(0.05).round(4).to_dict(),
        "median": num.median().round(4).to_dict(),
        "component_count_ratio_median": float((df.n_components_mask_rec / df.n_components_orig.clip(lower=1)).median()),
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
