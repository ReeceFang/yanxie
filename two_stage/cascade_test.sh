set -e

export OMP_NUM_THREADS=4

# python cascade_test.py \
#   --merged-run-path runs/merged \
#   --second-runs-path runs \
#   --val-path /root/autodl-tmp/yanxie_data/val \
#   --mapping-json merge_dataset_classes.json \
#   --route-top-k 2 \
#   --weights best

python cascade_test.py `
  --merged-run-path runs/merged `
  --second-runs-path runs `
  --val-path D:/Python_Data/mmpretrain_test4/val `
  --mapping-json merge_dataset_classes.json `
  --route-top-k 3 `
  --weights best