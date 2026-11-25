#!/bin/bash
#SBATCH --job-name=bcss_fetch
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=03:00:00
#SBATCH --output=logs/bcss_fetch_%j.out
#SBATCH --error=logs/bcss_fetch_%j.out
#
# BCSS（Breast Cancer Semantic Segmentation、CC0）を共通データ置き場に取得する。計算ノードで実行。
#   sbatch --partition=<partition> tools/bcss/fetch_bcss.sh
# 取得元は公式リポジトリ（https://github.com/PathologyDataScience/BCSS）が案内する Google Drive のフォルダ（0.25 MPP）。
# gdown はフォルダあたり 50 ファイルまでしか一覧できないことがあるので、取れた数を最後に表示して確認する。
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
cd "${PROJECT_ROOT}"
source "${PROJECT_ROOT}/scripts/sif_env.sh"
require_sif "${PROJECT_ROOT}"
OUT=/workspace/filesrv02/honzawa/data/bcss
mkdir -p "${OUT}"

apptainer exec --bind "${PROJECT_ROOT}" --bind /workspace/filesrv02/honzawa/data "${SIF_PATH}" bash -c "
    set -euo pipefail
    cd ${PROJECT_ROOT}
    source .venv/bin/activate
    uv pip install -q --target ${PROJECT_ROOT}/data/pydeps gdown
    export PYTHONPATH=${PROJECT_ROOT}/data/pydeps
    python -m gdown --folder 'https://drive.google.com/drive/folders/1zqbdkQF8i5cEmZOGmbdQm-EP8dRYtvss' -O ${OUT}/ || echo 'gdown exited non-zero'
"
echo "== result"
find "${OUT}" -maxdepth 2 | head -30
echo "files: $(find "${OUT}" -type f | wc -l)"
du -sh "${OUT}"
