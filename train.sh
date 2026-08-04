set -e

export OMP_NUM_THREADS=4

# rm -r runs

python train.py \
  --data-dir /root/autodl-tmp/yanxie_data \
  --model convnext_base \
  --model-path /root/autodl-tmp/timm/convnext_base/pytorch_model.bin \
  --num-classes 20 \
  --epochs 50 \
  --batch-size 64 \
  --learning-rate 1e-3 \
  --input-size 96

echo "任务完成, 准备关机..."
/usr/bin/shutdown