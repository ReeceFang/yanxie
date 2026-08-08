set -e

export OMP_NUM_THREADS=4

python cascade_test.py \
  --merged-run-path runs/merged \
  --second-runs-path runs \
  --val-path /root/autodl-tmp/yanxie_data/val \
  --mapping-json merge_dataset_classes.json \
  --weights best
  