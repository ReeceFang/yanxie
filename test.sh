set -e

export OMP_NUM_THREADS=4

python test.py \
  --run-path runs/convnext_base \
  --val-path /root/autodl-tmp/yanxie_data/val \
  --input-size 96
