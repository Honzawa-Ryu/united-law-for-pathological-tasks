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
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="data/wsm_cache の下の名前（bracs_cls_10x / bracs_cls_20x）")
    parser.add_argument("--seed", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    project_root = _get_project_root()
    sys.path.insert(0, str(project_root))

    from lib.output_utils import complete_run, get_run_dir, write_run_metadata

    exp_name = os.environ["EXP_NAME"]
    output_root = os.environ.get("OUTPUT_ROOT")
    args = parse_args()
    variant_key = f"{args.cache}__seed{args.seed}"

    run_dir = get_run_dir(project_root, __file__, variant_key, output_root=output_root)
    logger = setup_logger(run_dir, exp_name)
    config = load_config(Path(__file__).parent)
    write_run_metadata(run_dir, exp_name=exp_name, variant_key=variant_key)
    logger.info(f"Starting: {exp_name} / {variant_key}")

    # ── Experiment logic ──────────────────────────────────────────────────────
    import subprocess

    import torch
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

    wsm = project_root / "libraries" / "WSMs" / "WSM_v1"
    refiner_py = wsm / "src" / "pipelines" / "categorical_refiner.py"
    assert "[united-law patch] honour explicit split" in refiner_py.read_text(), \
        "tools/patches/apply_wsm_refiner_split_patch.py を先に実行する（公式分割が混ざるのを防ぐ）"
    cache_dir = project_root / config["cache_root"] / args.cache
    shards_root = cache_dir / "shards"
    target = config["target_modality"]
    classes = list(config["classes"])
    palette = [list(config["palette"][c]) for c in classes]
    sidecar_dir = cache_dir / "categorical_targets" / target

    # 1) 分類器の教師（sidecar）を作る。画像全体が 1 クラスなので、64x64 の全セルに「クラス番号 + 1」を入れる。
    target_shards = sorted((shards_root / target).glob("shard_*.pt"))
    assert target_shards, f"no target shards under {shards_root / target}"
    split_of, label_of = {}, {}
    for path in target_shards:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
        ids = [str(s) for s in payload["sample_ids"]]
        labels = []
        for sid, src in zip(ids, payload["sample_sources"]):
            cls = sid.split("/")[2]          # BRACS/<split>/<CLASS>/<file>
            assert cls in classes and sid.split("/")[1] == src["split"], (sid, src)
            labels.append(classes.index(cls))
            split_of[sid], label_of[sid] = src["split"], classes.index(cls)
        out = sidecar_dir / path.name
        if not out.exists():
            h, w = payload["latents"].shape[-2:]
            spatial = (torch.tensor(labels, dtype=torch.uint8) + 1)[:, None, None].expand(-1, h, w).contiguous()
            sidecar_dir.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".tmp")
            torch.save({
                "contract": {"version": 3, "spatial_encoding": "whole_image_single_class"},
                "sample_ids": ids,
                "category_values": torch.arange(len(classes)),
                "category_spatial_index_map": spatial,
                "category_prototypes_rgb": torch.tensor(palette, dtype=torch.uint8),
                "background_rgb": None,
                "failed_sample_ids": [],
            }, tmp)
            tmp.replace(out)
    logger.info(f"sidecars: {len(target_shards)} shards, samples {len(split_of)}")

    # 2) 著者のスクリプトで分類器を学習する（既定値のまま。val の balanced accuracy が最良の重みが best.pt）
    train_dir = project_root / config["run_root"] / exp_name / variant_key / "refiner_train"
    env = dict(os.environ, PYTHONPATH=str(wsm), WSG_READER_VISUALIZATION="0")
    if not (train_dir / "best.pt").exists() or not (train_dir / "DONE").exists():
        cmd = [
            sys.executable, str(refiner_py), "train",
            "--cache-root", str(shards_root), "--categorical-root", str(cache_dir / "categorical_targets"),
            "--condition-modality", config["condition_modality"], "--target-modality", target,
            "--head-index", str(config["channel_head_index"]), "--output-dir", str(train_dir),
            "--device", "cuda:0", "--seed", str(args.seed),
            "--max-steps", str(config["max_steps"]), "--eval-every", str(config["eval_every"]),
        ]
        logger.info(" ".join(cmd))
        with open(run_dir / "refiner_train.log", "w") as log:
            rc = subprocess.run(cmd, cwd=wsm, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if rc != 0:
            logger.error((run_dir / "refiner_train.log").read_text()[-4000:])
            sys.exit(rc)
        (train_dir / "DONE").write_text("ok\n")

    # 3) best.pt で val / test を予測する（著者のデータセットとモデルのクラスをそのまま使う）
    sys.path.insert(0, str(wsm))
    sys.path.insert(0, str(wsm / "src" / "pipelines"))
    import categorical_refiner as cr

    ckpt = torch.load(train_dir / "best.pt", map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = cr.MaskedLatentClassifier(
        latent_channels=int(cfg["latent_channels"]), num_classes=int(cfg["num_classes"]),
        embedding_dim=int(cfg["embedding_dim"]),
    )
    cr.load_compact_classifier_state(model, ckpt["model"])
    model = model.cuda().eval()
    n_params = sum(p.numel() for p in model.parameters())

    def metrics(true, pred) -> dict:
        idx = list(range(len(classes)))
        return {
            "n": int(len(true)),
            "accuracy": float(accuracy_score(true, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
            "macro_f1": float(f1_score(true, pred, labels=idx, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(true, pred, labels=idx, average="weighted", zero_division=0)),
            "per_class_f1": dict(zip(classes, f1_score(true, pred, labels=idx, average=None, zero_division=0).round(4).tolist())),
        }

    results: dict = {
        "cache": args.cache, "seed": args.seed, "classes": classes,
        "classifier_params": int(n_params), "best_step": int(ckpt["step"]),
        "best_val_from_training": {k: v for k, v in ckpt["validation"].items() if isinstance(v, (int, float))},
    }
    expected_n = {sp: sum(1 for v in split_of.values() if v == sp) for sp in ("train", "val", "test")}
    results["n_per_split_in_cache"] = expected_n
    for split in ("val", "test"):
        ds = cr.DenseRegionDataset(sidecar_dir, shards_root, config["condition_modality"], target, split, 6)
        assert len(ds) == expected_n[split], (split, len(ds), expected_n[split])   # 公式分割どおりに選ばれているか
        true, pred = [], []
        with torch.no_grad():
            for start in range(0, len(ds), 256):
                items = [ds[i] for i in range(start, min(start + 256, len(ds)))]
                latent = torch.stack([it[0] for it in items]).cuda()
                mask = torch.stack([it[1] for it in items]).cuda()
                logits = model(cr.masked_condition_latents(latent, mask))
                pred += logits.argmax(1).cpu().tolist()
                true += [int(it[2]) for it in items]
        results[split] = metrics(true, pred)
        results[f"{split}_confusion_matrix_rows_true_cols_pred"] = confusion_matrix(
            true, pred, labels=list(range(len(classes)))).tolist()
    import shutil
    shutil.copy(train_dir / "metrics.csv", run_dir / "refiner_metrics.csv")
    logger.info(json.dumps(results, ensure_ascii=False))

    # ── Save results ──────────────────────────────────────────────────────────
    (run_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False))
    complete_run(run_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
