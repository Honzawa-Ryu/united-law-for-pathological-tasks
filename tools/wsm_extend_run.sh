#!/bin/bash
# 0005 の学習済み run を、より多いステップ数の run として続きから学習する（parent で実行）。
# run ディレクトリ名に最大ステップ数が入っているので、ckpt と state を新しい名前にコピーしてから投入する。
#   bash tools/wsm_extend_run.sh <cache> <旧ステップ数> <新ステップ数> [サフィックス 例: __bs4] [追加の --export 例: BATCH=4]
set -euo pipefail
cd /workspace/filesrv02/honzawa/united-law-for-pathological-tasks
CACHE=$1; OLD=$2; NEW=$3; SUFFIX=${4:-}; EXTRA=${5:-}
R=data/wsm_runs/0005_20261122_bracs_wsm_train_cls
SRC=$R/${CACHE}__steps${OLD}${SUFFIX}
DST=$R/${CACHE}__steps${NEW}${SUFFIX}
[ -d "$SRC/state" ] || { echo "no state in $SRC" >&2; exit 1; }
if [ ! -d "$DST" ]; then
  mkdir -p "$DST"
  cp -r "$SRC/ckpt" "$SRC/state" "$DST/"
  cp "$SRC"/stats.npz "$DST/" 2>/dev/null || true
fi
ls "$DST/state"
EXPORT="ALL,CACHE=${CACHE},MAX_STEPS=${NEW}"
[ -n "$EXTRA" ] && EXPORT="${EXPORT},${EXTRA}"
sbatch --partition=x-large-creator --time=14:00:00 --export="$EXPORT" experiments/0005_20261122_bracs_wsm_train_cls/run_slurm.sh
