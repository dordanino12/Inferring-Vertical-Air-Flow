#!/usr/bin/env bash
# One-screen progress of a (re)training run started with scripts/retrain.sh.
#   bash scripts/status.sh [runs/retrain]
cd "$(dirname "$0")/.."
OUT=${1:-runs/retrain}
echo "== queue";   tmux list-windows -t retrain -F "  window #{window_index}: #{window_name}" 2>/dev/null
echo "== now";      cat "$OUT/STATUS.txt" 2>/dev/null || echo "(no status yet)"
echo "== running";  if tmux has-session -t retrain 2>/dev/null; then echo "tmux session 'retrain' is alive"; else echo "tmux session 'retrain' NOT running"; fi
pgrep -af "python -u (train|evaluate).py" | sed 's/^/  /'
echo "== history";  tail -n 15 "$OUT/history.txt" 2>/dev/null
echo "== finished folds"
for d in "$OUT"/*/; do [ -f "$d/train_metrics.json" ] && printf '  %-14s %s\n' "$(basename "$d")" \
  "$(python3 -c "import json,sys;r=json.load(open(sys.argv[1]));print(len(r),'folds:',' '.join(f\"{k[:7]} MAE={v['test_mae']:.3f}\" for k,v in r.items()))" "$d/train_metrics.json")"; done
echo "== last epochs";  ls -t "$OUT"/logs/*.log 2>/dev/null | grep -v _eval | head -1 | xargs -r grep -A2 '^Epoch' | tail -n 6
echo "== gpu";      nvidia-smi --query-gpu=utilization.gpu,memory.used,temperature.gpu --format=csv,noheader
echo "== disk";     df -h "$OUT" | tail -1
[ -f "$OUT/retrain_vs_paper.md" ] && { echo "== vs paper"; grep '^|' "$OUT/retrain_vs_paper.md"; }
exit 0
