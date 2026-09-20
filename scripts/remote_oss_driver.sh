#!/bin/bash
# Runs on dose-oss-eval-l4. Logs to ~/oss-eval/driver.log
set -u
exec > >(tee -a "$HOME/oss-eval/driver.log") 2>&1
echo "=== driver start $(date -Is) ==="
cd "$HOME/synthio_voice"

# DLVM images keep CUDA torch on the system python.
PY=$(command -v python3)
echo "python $($PY -c 'import sys,torch; print(sys.executable, torch.__version__, torch.cuda.is_available())')"

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq espeak-ng git-lfs >/dev/null

$PY -m pip install -q -U pip setuptools wheel
# kokoro/qwen will otherwise upgrade torchaudio past the image's torch 2.9.1.
$PY -m pip install -q torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
$PY -m pip install -q \
  "transformers>=4.48" librosa soundfile panphon phonemizer \
  pandas pyarrow kokoro "misaki[en]" huggingface_hub \
  --upgrade-strategy only-if-needed
$PY -m pip install -q torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

echo "=== kokoro plain $(date -Is) ==="
$PY scripts/synth_kokoro.py --mode plain --out "$HOME/synthio_voice/runs/oss-eval/kokoro-plain"
echo "=== kokoro phoneme $(date -Is) ==="
$PY scripts/synth_kokoro.py --mode phoneme --out "$HOME/synthio_voice/runs/oss-eval/kokoro-phoneme"

echo "=== score kokoro $(date -Is) ==="
$PY scripts/score_oss_run.py --condition kokoro-plain --wav-dir "$HOME/synthio_voice/runs/oss-eval/kokoro-plain" || true
$PY scripts/score_oss_run.py --condition kokoro-phoneme --wav-dir "$HOME/synthio_voice/runs/oss-eval/kokoro-phoneme" || true

echo "=== cosyvoice setup $(date -Is) ==="
export PATH="$HOME/.local/bin:$PATH"
if [ ! -d "$HOME/CosyVoice/cosyvoice" ]; then
  rm -rf "$HOME/CosyVoice"
  git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git "$HOME/CosyVoice"
  git -C "$HOME/CosyVoice" submodule update --init --recursive --depth 1
fi
# Keep the image's CUDA torch. Skip torch pins in CosyVoice's requirements.
grep -v -E '^(torch|torchaudio|torchvision)' "$HOME/CosyVoice/requirements.txt" > /tmp/cosy-req.txt || true
$PY -m pip install -q -r /tmp/cosy-req.txt || echo "cosy requirements partial failure"
if [ ! -f "$HOME/models/Fun-CosyVoice3-0.5B-2512/cosyvoice.yaml" ]; then
  $PY - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("FunAudioLLM/Fun-CosyVoice3-0.5B-2512", local_dir="/home/shreyaspatel/models/Fun-CosyVoice3-0.5B-2512")
PY
fi
PROMPT="$HOME/CosyVoice/asset/zero_shot_prompt.wav"
echo "=== cosyvoice plain $(date -Is) ==="
$PY scripts/synth_cosyvoice.py --mode plain \
  --model-dir "$HOME/models/Fun-CosyVoice3-0.5B-2512" \
  --cosyvoice-root "$HOME/CosyVoice" \
  --prompt-wav "$PROMPT" \
  --out "$HOME/synthio_voice/runs/oss-eval/cosyvoice3-plain" || echo "cosy plain failed"
echo "=== cosyvoice phoneme $(date -Is) ==="
$PY scripts/synth_cosyvoice.py --mode phoneme \
  --model-dir "$HOME/models/Fun-CosyVoice3-0.5B-2512" \
  --cosyvoice-root "$HOME/CosyVoice" \
  --prompt-wav "$PROMPT" \
  --out "$HOME/synthio_voice/runs/oss-eval/cosyvoice3-phoneme" || echo "cosy phoneme failed"

echo "=== score cosy $(date -Is) ==="
$PY scripts/score_oss_run.py --condition cosyvoice3-plain --wav-dir "$HOME/synthio_voice/runs/oss-eval/cosyvoice3-plain" || true
$PY scripts/score_oss_run.py --condition cosyvoice3-phoneme --wav-dir "$HOME/synthio_voice/runs/oss-eval/cosyvoice3-phoneme" || true

echo "=== qwen $(date -Is) ==="
$PY -m pip install -q qwen-tts --upgrade-strategy only-if-needed
$PY -m pip install -q torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129
$PY scripts/synth_qwen.py --out "$HOME/synthio_voice/runs/oss-eval/qwen3-plain" || echo "qwen failed"
$PY scripts/score_oss_run.py --condition qwen3-plain --wav-dir "$HOME/synthio_voice/runs/oss-eval/qwen3-plain" || true

echo "=== driver done $(date -Is) ==="
