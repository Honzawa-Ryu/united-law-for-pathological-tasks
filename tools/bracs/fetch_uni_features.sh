#!/bin/bash
# NAS にある BRACS の UNI2-h パッチ埋め込み（TRIDENT、20 倍 256 px）を、注釈付き WSI の分だけ取る。
# parent で実行（計算ノードには NAS の鍵が無い）。再実行すると続きから。
#   bash tools/bracs/fetch_uni_features.sh
set -euo pipefail
# NAS の接続先は環境変数で渡す: NAS_HOST（user@host）、NAS_PORT、NAS_KEY（秘密鍵のパス）。値はワークスペースの CLAUDE.md を参照。
D=/workspace/filesrv02/honzawa/data/bracs
mkdir -p "$D/trident_uni_v2" "$D/logs"
# BRACS_WSI/{split}/Group_X/Type_Y/NAME.svs -> Group_X/Type_Y/20x_256px_0px_overlap/features_uni_v2/NAME.h5
cat "$D"/lists/annotated_wsi_train.txt "$D"/lists/annotated_wsi_val.txt "$D"/lists/annotated_wsi_test.txt \
  | sed -E 's#^BRACS_WSI/[^/]+/(Group_[^/]+/Type_[^/]+)/([^/]+)\.svs$#\1/20x_256px_0px_overlap/features_uni_v2/\2.h5#' \
  > "$D/lists/annotated_uni_features.txt"
rsync -rt --partial --files-from="$D/lists/annotated_uni_features.txt" \
  -e "ssh -i ${NAS_KEY:?} -o IdentitiesOnly=yes -o BatchMode=yes -p ${NAS_PORT:?}" \
  "${NAS_HOST:?}":/share/main/image/BRACS/processed/TRIDENT/ "$D/trident_uni_v2/"
echo "DONE uni_features $(date)"
