#!/bin/bash
#SBATCH --job-name=bracs_qpdata
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:20:00
#SBATCH --output=logs/bracs_qpdata_%j.out
#SBATCH --error=logs/bracs_qpdata_%j.out
#
# BRACS の .qpdata 注釈 387 件を GeoJSON に変換する（CPU のみ、x86 ノード用）。
#   sbatch --partition=<partition> tools/bracs/export_qpdata_geojson.sh
set -euo pipefail

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
BRACS_DIR="${BRACS_DIR:-/workspace/filesrv02/honzawa/data/bracs}"
QUPATH="${QUPATH:-/workspace/filesrv02/honzawa/00-utils/qupath/QuPath-v0.5.1-Linux/QuPath/bin/QuPath}"

"${QUPATH}" script \
    -a "${BRACS_DIR}/BRACS_WSI_Annotations" \
    -a "${BRACS_DIR}/roi_annotations_geojson" \
    "${PROJECT_ROOT}/tools/bracs/export_qpdata_geojson.groovy"

echo "geojson files: $(find "${BRACS_DIR}/roi_annotations_geojson" -name '*.geojson' | wc -l)"
