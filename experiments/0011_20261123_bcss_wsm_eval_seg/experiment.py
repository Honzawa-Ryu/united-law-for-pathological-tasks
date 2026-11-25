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
    parser.add_argument("--run", required=True, help="0005 の variant 名（例: bracs_cls_10x__steps3000）")
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--infer-seed", required=True, type=int,
                        help="生成の初期ノイズのシード。0 なら config.yml の seed（従来どおり）")
    parser.add_argument("--ckpt", required=True,
                        help="best（検証損失が最良）/ final（最終ステップ）/ ckpt ディレクトリ名（例: 0003000）")
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
    # best は従来の名前のまま（既存の結果と同じ場所）
    variant_key = f"{args.run}__{args.split}" + ("" if args.ckpt == "best" else f"__ckpt-{args.ckpt}")
    if args.infer_seed != 0:
        variant_key += f"__seed{args.infer_seed}"

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
    import time

    import torch
    from diffusers import AutoencoderKLFlux2
    from huggingface_hub import snapshot_download
    from PIL import Image

    wsm = project_root / "libraries" / "WSMs" / "WSM_v1"
    wsm_run = project_root / config["run_root"] / config["train_exp"] / args.run
    cache_name = args.run.split("__")[0]
    uni_mode = next((t.split("-", 1)[1] for t in args.run.split("__") if t.startswith("uni-")), None)
    cache_dir = project_root / config["cache_root"] / cache_name
    target = config["target_modality"]

    # 検証損失が最良のチェックポイント（数字だけの名前のうち最新）。無ければ final。
    best = sorted(d for d in (wsm_run / "ckpt").iterdir() if d.name.isdigit())
    finals = sorted((wsm_run / "ckpt").glob("final_step_*"))
    if args.ckpt == "best":
        ckpt_dir = best[-1] if best else finals[-1]
    elif args.ckpt == "final":
        ckpt_dir = finals[-1]
    else:
        ckpt_dir = wsm_run / "ckpt" / args.ckpt
    ckpt = ckpt_dir / "pytorch_lora_weights.safetensors"
    if args.infer_seed != 0:
        seed = args.infer_seed
    label = f"eval_{args.split}_{ckpt_dir.name}" + (f"_seed{args.infer_seed}" if args.infer_seed != 0 else "")
    inf_dir = wsm_run / "pseudo-visualization-inference" / label
    logger.info(f"checkpoint: {ckpt}")

    base = snapshot_download(config["base_model_repo"], local_files_only=True)
    cmd = [
        sys.executable, "scripts/train_TunnelingNet_Flux2_4B_wsg.py",
        "--model-path", base, "--modality-config", str(wsm_run / "modalities_custom.json"),
        "--dense-latent-cache-dir", str(cache_dir), "--channel-head-index", str(config["channel_head_index"]),
        "--target-modality", target,
        "--condition-modalities", "0-2" if uni_mode is None else "Audio_AST,CurrentFrame_RGB_Condition",
        "--channel-head-rank", str(config["channel_head_rank"]),
        "--channel-head-alpha", str(config["channel_head_alpha"]),
        "--categorical-loss-mode", "off", "--seed", str(seed), "--output-dir", str(wsm_run),
        "--inference-only", "--inference-checkpoint", str(ckpt), "--inference-label", label,
        "--inference-steps", str(config["inference_steps"]),
        "--preview-split", args.split, "--preview-num-samples", "100000",
        "--inference-save-raw-latent", "--inference-cache-only",
    ]
    if uni_mode is not None:
        # 特徴条件の経路では、推論対象は --audio-dense-condition-val の NPZ で決まる（--preview-split は効かない）。
        # 変換層の読み込みは tools/patches/apply_wsm_projector_patch.py の修正が必要。
        patched = "[united-law patch]" in (wsm / "scripts" / "train_TunnelingNet_Flux2_4B_wsg.py").read_text()
        if not patched:
            logger.error("run tools/patches/apply_wsm_projector_patch.py first")
            sys.exit(1)
        npz = str(cache_dir / f"uni_{args.split}.npz")
        cmd += [
            "--audio-dense-condition-train", npz, "--audio-dense-condition-val", npz,
            "--val-loss-batches", "24", "--max-val-samples", "0",
            "--audio-condition-dim", str(config["uni_dim"]),
            "--audio-conditioning-mode", uni_mode, "--audio-global-condition",
        ]
    env = dict(os.environ, PYTHONPATH=str(wsm), WSG_READER_VISUALIZATION="0")
    logger.info("cmd: " + " ".join(cmd))
    t0 = time.perf_counter()
    with open(wsm_run / f"{label}_{os.environ.get('SLURM_JOB_ID', 'local')}.log", "w") as log:
        rc = subprocess.run(cmd, cwd=wsm, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    infer_sec = time.perf_counter() - t0
    if rc != 0:
        logger.error(f"inference exited with {rc}; see {wsm_run}/{label}_*.log")
        sys.exit(1)

    files = sorted(inf_dir.glob("**/generated_raw_latent.pt"))
    logger.info(f"generated latents: {len(files)}  ({infer_sec:.0f} sec)")
    vae = AutoencoderKLFlux2.from_pretrained(base, subfolder="vae", torch_dtype=torch.float32).to("cuda").eval()
    meta = json.loads((cache_dir / "palette.json").read_text())
    classes = list(meta["classes"])                               # クラス番号は 1 始まり。0 は ignore
    pal = np.array([meta["ignore"]] + [meta["palette"][c] for c in classes], dtype=np.float32)
    K = len(classes) + 1
    colors = pal.astype(np.uint8)
    ex_dir = run_dir / "examples"
    ex_dir.mkdir(exist_ok=True)
    conf = np.zeros((K, K), dtype=np.int64)                       # 行: 正解、列: 予測
    rows = []
    for k, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        z = d["latent"].float()
        if z.ndim == 5:
            z = z[:, :, 0]
        with torch.no_grad():
            img = vae.decode(z.to("cuda")).sample[0]
        rgb = img.clamp(-1, 1).add(1).mul(127.5).permute(1, 2, 0).cpu().numpy()
        pred = ((rgb[:, :, None, :] - pal[None, None]) ** 2).sum(-1).argmin(-1)   # 最も近い色（ignore の黒も候補）
        sid = d["sample_id"].split("::")[0]
        gt = np.asarray(Image.open(cache_dir / "gt" / Path(sid).name))
        valid = gt > 0
        conf += np.bincount(gt[valid] * K + pred[valid], minlength=K * K).reshape(K, K)
        rows.append({"sample_id": sid, "pixel_acc": float((pred[valid] == gt[valid]).mean()),
                     "pred_ignore_fraction": float((pred[valid] == 0).mean())})
        if k < config["num_example_images"]:
            panel = np.concatenate([colors[gt], colors[pred], rgb.round().astype(np.uint8)], axis=1)
            Image.fromarray(panel).save(ex_dir / f"{k:02d}_{Path(sid).stem}.png")
    df = pd.DataFrame(rows)
    df.to_csv(run_dir / "per_tile.csv", index=False)
    tp = np.diag(conf)[1:].astype(float)
    gt_n, pred_n = conf.sum(1)[1:], conf.sum(0)[1:]
    iou = tp / np.maximum(gt_n + pred_n - tp, 1)
    dice = 2 * tp / np.maximum(gt_n + pred_n, 1)
    present = gt_n > 0
    results: dict = {
        "run": args.run, "split": args.split, "checkpoint": ckpt_dir.name, "infer_seed": int(seed),
        "n_tiles": int(len(df)), "inference_seconds": infer_sec, "sec_per_tile": infer_sec / max(1, len(df)),
        "pixel_accuracy": float(tp.sum() / max(conf[1:].sum(), 1)),
        "mIoU": float(iou[present].mean()), "mean_dice": float(dice[present].mean()),
        "per_class_iou": dict(zip(classes, iou.round(4).tolist())),
        "per_class_dice": dict(zip(classes, dice.round(4).tolist())),
        "gt_pixel_fraction": dict(zip(classes, (gt_n / max(gt_n.sum(), 1)).round(4).tolist())),
        "pred_ignore_fraction_on_labeled_pixels": float(conf[1:, 0].sum() / max(conf[1:].sum(), 1)),
        "confusion_rows_true_cols_pred_incl_ignore": conf.tolist(),
        "classes": ["ignore"] + classes,
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
