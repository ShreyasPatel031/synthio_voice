"""Make sft_12hz resolve HF hub ids to a local snapshot before copytree."""
from pathlib import Path

p = Path("/home/shreyaspatel/Qwen3-TTS/finetuning/sft_12hz.py")
t = p.read_text()
if "snapshot_download" in t:
    print("already patched")
else:
    old = "MODEL_PATH = args.init_model_path\n"
    new = (
        "MODEL_PATH = args.init_model_path\n"
        "    if not os.path.isdir(MODEL_PATH):\n"
        "        from huggingface_hub import snapshot_download\n"
        "        MODEL_PATH = snapshot_download(MODEL_PATH)\n"
    )
    if old not in t:
        raise SystemExit("assign missing")
    p.write_text(t.replace(old, new, 1))
    print("patched")
