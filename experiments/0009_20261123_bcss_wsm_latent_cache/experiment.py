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
    import urllib.request

    import torch
    from diffusers import AutoencoderKLFlux2
    from huggingface_hub import snapshot_download
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    bcss = Path(config["bcss_dir"])
    tile = config["tile_px"]
    scale = 40 // args.magnification             # 0.25 MPP（40 倍）からの縮小率
    target_name = config["target_modality"]
    out_root = project_root / config["cache_root"] / f"bcss_seg_{args.magnification}x"
    (out_root / "gt").mkdir(parents=True, exist_ok=True)

    classes = list(config["class_groups"])        # クラス番号は 1 始まり。0 は ignore
    lut = np.zeros(256, dtype=np.uint8)
    for k, name in enumerate(classes, start=1):
        lut[config["class_groups"][name]] = k
    pal = np.array([config["palette"]["ignore"]] + [config["palette"][c] for c in classes], dtype=np.uint8)

    # 取れていないマスクは、meta の表にある figshare のリンクから取る
    bounds = pd.read_csv(bcss / "meta" / "roiBounds_BaseMagnification.csv", index_col=0)
    rgb_files = sorted((bcss / "rgbs_colorNormalized").glob("*.png"))
    fetched = []
    for f in rgb_files:
        m = bcss / "masks" / f.name
        if not m.exists():
            slide = f.name.split("_")[0]
            try:
                urllib.request.urlretrieve(bounds.loc[slide, "mask_link"], m)
                fetched.append(f.name)
            except Exception as e:  # 取れなければその ROI は使わない（結果に記録する）
                logger.error(f"mask fetch failed for {f.name}: {e!r}")
                m.unlink(missing_ok=True)
    n_all = len(rgb_files)
    rgb_files = [f for f in rgb_files if (bcss / "masks" / f.name).exists()]
    logger.info(f"ROIs with mask: {len(rgb_files)}/{n_all}, masks fetched from figshare: {fetched}")

    # 施設単位の分割
    slides = sorted(f.name.split("_")[0] for f in rgb_files)
    inst = {s_: s_.split("-")[1] for s_ in slides}
    rest = [s_ for s_ in slides if inst[s_] not in set(config["test_institutes"])]
    step = max(1, round(1 / config["val_fraction_of_remaining_slides"]))
    val_slides = set(rest[step // 2::step])
    split_of = {s_: ("test" if inst[s_] in set(config["test_institutes"]) else "val" if s_ in val_slides else "train")
                for s_ in slides}

    tiles, targets, rows = [], [], []
    resized, skipped = [], []
    for f in rgb_files:
        slide = f.name.split("_")[0]
        rgb = Image.open(f).convert("RGB")
        mask = Image.open(bcss / "masks" / f.name)
        if rgb.size != mask.size:
            # figshare から取り直したマスクは元の倍率（0.25 MPP より細かい）のことがある。
            # 縦横比が同じなら画像の大きさに最近傍で合わせる。違えばその ROI は使わない。
            ratio = (mask.width / rgb.width, mask.height / rgb.height)
            if abs(ratio[0] - ratio[1]) > 0.005:
                logger.error(f"skip {f.name}: size mismatch rgb={rgb.size} mask={mask.size}")
                skipped.append(f.name)
                continue
            logger.info(f"resize mask of {f.name}: {mask.size} -> {rgb.size}")
            resized.append(f.name)
            mask = mask.resize(rgb.size, Image.NEAREST)
        if scale > 1:
            size = (rgb.width // scale, rgb.height // scale)
            rgb, mask = rgb.resize(size, Image.LANCZOS), mask.resize(size, Image.NEAREST)
        rgb, lab = np.asarray(rgb), lut[np.asarray(mask)]
        for y in range(0, rgb.shape[0] - tile + 1, tile):
            for x in range(0, rgb.shape[1] - tile + 1, tile):
                l = lab[y:y + tile, x:x + tile]
                labeled = float((l > 0).mean())
                if labeled < config["min_labeled_fraction"]:
                    continue
                sid = f"BCSS/{split_of[slide]}/{slide}_y{y}_x{x}.png"
                Image.fromarray(l).save(out_root / "gt" / Path(sid).name)
                tiles.append(rgb[y:y + tile, x:x + tile])
                targets.append(pal[l])
                rows.append({"sample_id": sid, "split": split_of[slide], "slide": slide, "institute": inst[slide],
                             "y": y, "x": x, "labeled_fraction": labeled,
                             **{f"frac_{c}": float((l == k).mean()) for k, c in enumerate(classes, start=1)}})
    samples = pd.DataFrame(rows)
    order = rng.permutation(len(samples))          # 各 shard に全 split が入るように
    samples = samples.iloc[order].reset_index(drop=True)
    tiles = [tiles[i] for i in order]
    targets = [targets[i] for i in order]
    logger.info(f"tiles: {len(samples)}  per split: {samples.split.value_counts().to_dict()}")

    base = snapshot_download(config["base_model_repo"], local_files_only=True)
    vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()

    def encode(batch_u8: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(batch_u8).permute(0, 3, 1, 2).float().div(127.5).sub(1.0).to("cuda")
        with torch.no_grad():
            z = vae.encode(x).latent_dist.sample()
        return z.unsqueeze(2).to(torch.bfloat16).cpu()

    bs = config["batch_size"]
    lat_rgb = torch.cat([encode(np.stack(tiles[i:i + bs])) for i in range(0, len(tiles), bs)])
    lat_tgt = torch.cat([encode(np.stack(targets[i:i + bs])) for i in range(0, len(targets), bs)])
    ex = run_dir / "examples"
    ex.mkdir(exist_ok=True)
    for j in range(min(8, len(tiles))):
        Image.fromarray(np.concatenate([tiles[j], targets[j]], axis=1)).save(ex / f"{j:02d}.jpg", quality=90)

    sources = [{"dataset_name": "BCSS", "split": r.split, "video_id": Path(r.sample_id).stem, "video_name": r.slide,
                "frame_index": 0} for r in samples.itertuples()]
    ids = samples.sample_id.tolist()
    n_shards = config["num_shards"]
    for modality, latents in (("0-2", lat_rgb), (target_name, lat_tgt)):
        d = out_root / "shards" / modality.replace("-", "_")
        d.mkdir(parents=True, exist_ok=True)
        for sid in range(n_shards):
            idx = list(range(sid, len(samples), n_shards))
            torch.save({"modality": modality, "num_samples": len(idx), "latents": latents[idx].clone(),
                        "sample_ids": [ids[k] for k in idx], "sample_sources": [sources[k] for k in idx],
                        "video_ids": [sources[k]["video_id"] for k in idx], "frame_indices": [0 for _ in idx]},
                       d / f"shard_{sid:05d}_of_{n_shards:05d}.pt")
    samples.to_csv(out_root / "samples.csv", index=False)
    (out_root / "palette.json").write_text(json.dumps(
        {"classes": classes, "palette": {c: config["palette"][c] for c in classes}, "ignore": config["palette"]["ignore"]}, indent=2))

    results: dict = {
        "magnification": args.magnification, "mpp": 0.25 * scale, "cache_dir": str(out_root),
        "n_rois": len(rgb_files), "masks_fetched_from_figshare": fetched,
        "masks_resized_to_rgb": resized, "rois_skipped_size_mismatch": skipped,
        "n_tiles": int(len(samples)), "tiles_per_split": samples.split.value_counts().to_dict(),
        "slides_per_split": samples.groupby("split").slide.nunique().to_dict(),
        "class_pixel_fraction": {c: float(samples[f"frac_{c}"].mean()) for c in classes},
        "ignore_pixel_fraction": float(1 - samples.labeled_fraction.mean()),
        "latent_shape": list(lat_rgb.shape),
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
