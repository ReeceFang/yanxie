set -e

export OMP_NUM_THREADS=4

python train_expert.py \
  --data-dir /root/autodl-tmp/yanxie_data \
  --model convnext_base \
  --init-checkpoint runs/convnext_base/best_model.pth \
  --input-size 224 \
  --epochs 12 \
  --freeze-backbone-epochs 2 \
  --batch-size 8 \
  --output-dir runs/pair_expert

