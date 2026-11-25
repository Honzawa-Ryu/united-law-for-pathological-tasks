#!/bin/bash
# WSM の学習（0005）と評価（0006）、比較対象（0007）の状況をまとめて表示する（parent で実行。軽い処理のみ）。
#   bash tools/wsm_status.sh
R=/workspace/filesrv02/honzawa/united-law-for-pathological-tasks
cd "$R" || exit 1
date
echo "== jobs"
squeue -u "$USER" -h -o "%i %P %j %T %M %R" | cut -c1-110
echo "== recent job states"
sacct -u "$USER" -S now-6hours -X -n -o JobID%12,JobName%34,Partition%22,State%12,Elapsed | grep -E "0005|0006|0007" | tail -n 16
echo "== training runs"
for d in data/wsm_runs/0005_20261122_bracs_wsm_train_cls/*; do
  [ -d "$d" ] || continue
  L=$(ls -t "$d"/train_*.log 2>/dev/null | head -1)
  step=$(tail -c 4000 "$L" 2>/dev/null | tr '\r' '\n' | grep -o "Steps: [0-9]*" | tail -n 1)
  vals=$(cat "$d"/train_*.log 2>/dev/null | grep -o "val_loss step=[0-9]* loss=[0-9.]*" | sed -E 's/val_loss step=([0-9]+) loss=/\1:/' | tr '\n' ' ')
  echo "$(basename "$d") | $step | ckpt: $(ls "$d/ckpt" 2>/dev/null | tr '\n' ' ')"
  echo "   val: $vals"
done
echo "== eval results"
python3 - <<'EOF'
import glob, json
for f in sorted(glob.glob("outputs/0006_20261122_bracs_wsm_eval_cls/*/results.json")):
    d = json.load(open(f))
    v = d["vote"]
    print(f"{d['run']} {d['split']} ckpt={d['checkpoint']} n={d['n_samples']} acc={v['accuracy']:.4f} "
          f"macroF1={v['macro_f1']:.4f} wF1={v['weighted_f1']:.4f} top_frac={d['mean_vote_top_fraction']:.3f}")
    print("   per-class F1:", v["per_class_f1"])
    print("   pred dist:", d["pred_distribution"])
EOF
