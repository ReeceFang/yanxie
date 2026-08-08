set -e

export OMP_NUM_THREADS=4

if [ "$#" -ne 2 ]; then
  echo "用法: $0 <模型名> <API Base URL>" >&2
  exit 2
fi

LLM_MODEL=$1
BASE_URL=$2

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_ROOT=$(dirname "$SCRIPT_DIR")

python "$SCRIPT_DIR/cascade_llm_test.py" \
  --merged-run-path "$PROJECT_ROOT/runs/merged" \
  --val-path /root/autodl-tmp/yanxie_data/val \
  --llm-config "$SCRIPT_DIR/llm_second_stage.example.json" \
  --llm-model "$LLM_MODEL" \
  --base-url "$BASE_URL" \
  --env-file "$PROJECT_ROOT/.env" \
  --weights best

python two_stage/cascade_llm_test.py `
  --merged-run-path runs/merged `
  --val-path "D:\Python_Data\mmpretrain_test4\val" `
  --llm-config two_stage/llm_second_stage.json `
  --llm-model "qwen3.6-flash" `
  --base-url "https://dashscope.aliyuncs.com/compatible-mode/v1" `
  --env-file ".env" `
  --weights best