#!/bin/bash
# BRACS の転送・切り出しの状況をまとめて表示する（parent で実行。軽い処理のみ）。
#   bash tools/bracs/status.sh
D=/workspace/filesrv02/honzawa/data/bracs
R=/workspace/filesrv02/honzawa/united-law-for-pathological-tasks
date
echo "== fetch logs"
tail -n 2 "$D"/logs/fetch_*.log
echo "== staging"
echo "svs: $(find "$D/wsi_staging" -name '*.svs' | wc -l)  rsync procs: $(pgrep -u "$USER" -c rsync)"
du -sh "$D/wsi_staging"
echo "== done slides"
for s in train val test; do echo "$s $(ls "$D/roi_png/_done/$s" 2>/dev/null | wc -l)"; done
echo "== jobs"
squeue -u "$USER" -h -o "%i %j %T %M %R" | cut -c1-100
echo "== last 0002 result"
L=$(ls -t "$R"/logs/0002_20261121_bracs_roi_extract/*/slurm.out 2>/dev/null | head -1)
[ -n "$L" ] && grep -h "n_rois\|incomplete" "$L" | cut -c1-360
