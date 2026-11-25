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
    parser.add_argument("--encoder", required=True, choices=["dinov2_vitl14", "uni2h"])
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
    variant_key = args.encoder

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

    import timm
    import torch
    import torch.nn.functional as F
    from PIL import Image

    Image.MAX_IMAGE_PIXELS = None
    torch.manual_seed(seed)
    np.random.seed(seed)
    cache = project_root / config["cache_dir"]
    bcss = Path(config["bcss_dir"])
    samples = pd.read_csv(cache / "samples.csv")
    meta = json.loads((cache / "palette.json").read_text())
    classes = list(meta["classes"])
    K = len(classes) + 1                                     # 0 は ignore
    tile, scale, inp = config["tile_px"], config["scale"], config["input_px"]

    enc_cfg = config["encoders"][args.encoder]
    kwargs = dict(enc_cfg["kwargs"]) if isinstance(enc_cfg["kwargs"], dict) else {}
    if enc_cfg["kwargs"] == "uni2h":                           # UNI2-h のモデルカードに記載の設定
        kwargs = {"img_size": 224, "patch_size": 14, "depth": 24, "num_heads": 24, "init_values": 1e-5,
                  "embed_dim": 1536, "mlp_ratio": 2.66667 * 2, "num_classes": 0, "no_embed_class": True,
                  "mlp_layer": timm.layers.SwiGLUPacked, "act_layer": torch.nn.SiLU, "reg_tokens": 8,
                  "dynamic_img_size": True}
    else:
        kwargs = {"num_classes": 0, "img_size": inp, **kwargs}
    try:
        model = timm.create_model(enc_cfg["name"], pretrained=True, **kwargs).cuda().eval()
    except Exception as e:
        # 重みが取れない（承認やトークンが無い）場合は、その事実を結果に残して終わる
        logger.error(f"cannot load {enc_cfg['name']}: {e!r}")
        (run_dir / "results.json").write_text(json.dumps(
            {"encoder": args.encoder, "status": "encoder_unavailable", "error": repr(e)[:500]}, indent=2, ensure_ascii=False))
        sys.exit(1)
    n_prefix = int(getattr(model, "num_prefix_tokens", 1))
    mean = torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1)

    # --- タイルを 0009 と同じ切り方で作り直し、パッチ特徴を取り出す
    feats, gts = [None] * len(samples), [None] * len(samples)
    t0 = time.perf_counter()
    for slide, grp in samples.groupby("slide"):
        files = sorted((bcss / "rgbs_colorNormalized").glob(f"{slide}_*.png"))
        assert len(files) == 1, (slide, files)
        rgb = Image.open(files[0]).convert("RGB")
        rgb = np.asarray(rgb.resize((rgb.width // scale, rgb.height // scale), Image.LANCZOS))
        idx = grp.index.to_list()
        for b in range(0, len(idx), config["feature_batch_size"]):
            part = idx[b:b + config["feature_batch_size"]]
            x = np.stack([rgb[samples.y[i]:samples.y[i] + tile, samples.x[i]:samples.x[i] + tile] for i in part])
            x = torch.from_numpy(x).cuda().permute(0, 3, 1, 2).float().div(255)
            x = F.interpolate(x, size=(inp, inp), mode="bilinear", align_corners=False)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                tok = model.forward_features((x - mean) / std)[:, n_prefix:]
            g = int(tok.shape[1] ** 0.5)
            tok = tok.float().reshape(len(part), g, g, -1).permute(0, 3, 1, 2).half().cpu()
            for j, i in enumerate(part):
                feats[i] = tok[j]
                gts[i] = torch.from_numpy(np.asarray(Image.open(cache / "gt" / Path(samples.sample_id[i]).name)).astype(np.int64))
    feat_sec = time.perf_counter() - t0
    X, Y = torch.stack(feats), torch.stack(gts)
    logger.info(f"features {tuple(X.shape)} in {feat_sec:.0f}s ({feat_sec / len(samples):.3f} s/tile)")
    split = {sp: torch.from_numpy((samples.split == sp).to_numpy()) for sp in ("train", "val", "test")}

    head = torch.nn.Conv2d(X.shape[1], K - 1, kernel_size=1).cuda()
    opt = torch.optim.AdamW(head.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    def logits_for(xb: torch.Tensor) -> torch.Tensor:
        return F.interpolate(head(xb.cuda().float()), size=(tile, tile), mode="bilinear", align_corners=False)

    def evaluate(sp: str) -> dict:
        conf = torch.zeros(K, K, dtype=torch.int64)
        xs, ys = X[split[sp]], Y[split[sp]]
        with torch.no_grad():
            for b in range(0, len(xs), 64):
                pred = logits_for(xs[b:b + 64]).argmax(1).cpu() + 1
                y = ys[b:b + 64]
                v = y > 0
                conf += torch.bincount(y[v] * K + pred[v], minlength=K * K).reshape(K, K)
        conf = conf.numpy()
        tp = np.diag(conf)[1:].astype(float)
        gt_n, pred_n = conf.sum(1)[1:], conf.sum(0)[1:]
        iou = tp / np.maximum(gt_n + pred_n - tp, 1)
        dice = 2 * tp / np.maximum(gt_n + pred_n, 1)
        present = gt_n > 0
        return {"n_tiles": int(len(xs)), "pixel_accuracy": float(tp.sum() / max(conf[1:].sum(), 1)),
                "mIoU": float(iou[present].mean()), "mean_dice": float(dice[present].mean()),
                "per_class_iou": dict(zip(classes, iou.round(4).tolist())),
                "per_class_dice": dict(zip(classes, dice.round(4).tolist()))}

    tr_idx = torch.nonzero(split["train"]).squeeze(1)
    history, best = [], None
    for epoch in range(1, config["epochs"] + 1):
        perm = tr_idx[torch.randperm(len(tr_idx))]
        for b in range(0, len(perm), config["batch_size"]):
            bi = perm[b:b + config["batch_size"]]
            loss = F.cross_entropy(logits_for(X[bi]), (Y[bi] - 1).cuda(), ignore_index=-1)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if epoch % config["eval_every_epochs"] == 0:
            v = evaluate("val")
            history.append({"epoch": epoch, "train_loss_last": float(loss), "val_mIoU": v["mIoU"], "val_pixel_accuracy": v["pixel_accuracy"]})
            logger.info(json.dumps(history[-1]))
            if best is None or v["mIoU"] > best[1]["mIoU"]:
                best = (epoch, v, {k: t.detach().clone() for k, t in head.state_dict().items()})
    head.load_state_dict(best[2])
    torch.save(best[2], run_dir / "linear_head.pt")
    results: dict = {
        "encoder": args.encoder, "timm_name": enc_cfg["name"], "status": "ok",
        "feature_shape": list(X.shape), "feature_seconds_per_tile": feat_sec / len(samples),
        "best_epoch": best[0], "history": history,
        "val": evaluate("val"), "test": evaluate("test"), "classes": classes,
    }
    logger.info(json.dumps({k: results[k] for k in ("encoder", "best_epoch", "val", "test")}, ensure_ascii=False))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
