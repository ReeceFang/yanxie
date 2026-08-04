set -e

export OMP_NUM_THREADS=4

python cascade_test.py \
  --base-run-path runs/convnext_base \
  --val-path /root/autodl-tmp/yanxie_data/val \
  --base-input-size 96 \
  --base-weights best \
  --expert-checkpoint runs/pair_expert/best_expert.pth \
  --margin-threshold 0.15 \
  --expert-confidence 0.55 \
  --output-dir runs/pair_expert/cascade_results

