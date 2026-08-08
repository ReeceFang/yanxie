set -e

export OMP_NUM_THREADS=4

python test.py \
  --run-path runs/convnext_base \
  --data-dir /root/autodl-tmp/yanxie_data \
  --weights best
