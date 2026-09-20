#!/bin/bash
set -u
exec > /home/shreyaspatel/oss-eval/finetune-qwen-cloud-ipa.log 2>&1
cd /home/shreyaspatel/synthio_voice
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="/home/shreyaspatel/Qwen3-TTS:/home/shreyaspatel/Qwen3-TTS/finetuning:${PYTHONPATH:-}"
echo "=== start $(date -Is) ==="
# 1.7B only — 0.6B hits text/codec embed size mismatch in current SFT
python3 scripts/finetune_qwen_cloud_ipa.py \
  --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --work runs/finetune-qwen-cloud-ipa \
  --epochs 8 --batch-size 2 --lr 2e-6
echo "=== 1.7B done $(date -Is) ==="
