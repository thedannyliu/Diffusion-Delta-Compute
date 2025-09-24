#!/usr/bin/env bash
if command -v conda &>/dev/null; then
  eval "$(conda shell.bash hook)"
  conda activate dcllm
  echo "[setup] Activated conda environment: dcllm"
else
  echo "[setup] ERROR: conda not found. Please load conda first." >&2
  exit 1
fi
set -euo pipefail

# =========================
# Config & Defaults
# =========================
ROOT=${1:-/storage/ice1/2/9/eliu354/Projects/Diffusion-Delta-Compute}
MODELS_ROOT="$ROOT/models"
REPOS_ROOT="$MODELS_ROOT/repos"
CKPT_ROOT="$MODELS_ROOT/checkpoints"
LORA_ROOT="$MODELS_ROOT/lora"

# === Hugging Face caches ===
export HF_HOME=${HF_HOME:-/storage/ice1/2/9/eliu354/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME}
export HUGGINGFACE_HUB_CACHE=${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$HF_HOME/modules}

# Model IDs (easy to customize)
DREAM_MODEL_ID=${DREAM_MODEL_ID:-Dream-org/Dream-v0-Instruct-7B}
D2F_LORA_ID=${D2F_LORA_ID:-SJTU-Deng-Lab/D2F_Dream_Base_7B_Lora}

# Local "views" (symlinked directories for this project)
DREAM_LOCAL_DIR="$CKPT_ROOT/Dream-v0-Instruct-7B"
D2F_LORA_LOCAL_DIR="$LORA_ROOT/D2F_Dream_Base_7B_Lora"

# Git repos
DREAM_REPO_URL=${DREAM_REPO_URL:-https://github.com/DreamLM/Dream.git}
D2F_REPO_URL=${D2F_REPO_URL:-https://github.com/zhijie-group/Discrete-Diffusion-Forcing.git}

# Datasets to pre-cache: (name,config,splits)
DATASETS_JSON='[
  ["Salesforce/wikitext", "wikitext-2-raw-v1", ["train","validation","test"]],
  ["EleutherAI/lambada_openai", null, ["test"]],
  ["openai/gsm8k", "main", ["train","test"]]
]'

# Simple lock to avoid concurrent re-entrancy
LOCK_DIR="${ROOT}/.setup_lock"
mkdir -p "$(dirname "$LOCK_DIR")"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "[setup] Another setup is running (lock: $LOCK_DIR). Exit."
  exit 1
fi
cleanup() { r=$?; rm -rf "$LOCK_DIR"; exit $r; }
trap cleanup EXIT

# =========================
# Prepare directories
# =========================
mkdir -p "$REPOS_ROOT" "$CKPT_ROOT" "$LORA_ROOT"
mkdir -p "$HF_HOME" "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$HF_MODULES_CACHE"

echo "== HF cache locations =="
echo "HF_HOME               = $HF_HOME"
echo "HUGGINGFACE_HUB_CACHE = $HUGGINGFACE_HUB_CACHE"
echo "HF_DATASETS_CACHE     = $HF_DATASETS_CACHE"
echo "TRANSFORMERS_CACHE    = $TRANSFORMERS_CACHE"
echo "HF_MODULES_CACHE      = $HF_MODULES_CACHE"
echo

# =========================
# Python helpers (inline)
# =========================
run_python() {
python - "$@" <<'PY'
import argparse, json, os, sys, importlib.util
def need(pkg, hint):
    if importlib.util.find_spec(pkg) is None:
        sys.stderr.write(f"Missing '{pkg}'. Install via `{hint}`.\n")
        sys.exit(1)

parser = argparse.ArgumentParser()
parser.add_argument("--mode", required=True)
parser.add_argument("--datasets_json")
parser.add_argument("--dream_id")
parser.add_argument("--d2f_lora_id")
parser.add_argument("--dream_local")
parser.add_argument("--d2f_lora_local")
args = parser.parse_args()

if args.mode == "datasets":
    need("datasets", "pip install datasets")
    from datasets import load_dataset
    datasets = json.loads(args.datasets_json)
    cache_dir = os.environ.get("HF_DATASETS_CACHE", "")
    print(f"[datasets] Using HF_DATASETS_CACHE={cache_dir}")
    for name, conf, splits in datasets:
        lab = conf or "default"
        print(f"[datasets] Caching {name} ({lab}) ...")
        kwargs = {"cache_dir": cache_dir, "download_mode": "reuse_dataset_if_exists"}
        ds = load_dataset(name, conf, **kwargs) if conf else load_dataset(name, **kwargs)
        for sp in splits:
            if sp not in ds:
                raise SystemExit(f"Expected split '{sp}' in {name} ({lab})")
            _ = len(ds[sp])  # trigger materialization
        print(f"[datasets]   cached splits: {', '.join(ds.keys())}")
    print("[datasets] Done.")

elif args.mode == "models":
    need("huggingface_hub", "pip install -U huggingface_hub")
    from huggingface_hub import snapshot_download

    def fetch(model_id: str, local_dir: str):
        os.makedirs(local_dir, exist_ok=True)
        print(f"[models] snapshot_download: {model_id}")
        snapshot_download(
            repo_id=model_id,
            local_dir=local_dir,
            local_dir_use_symlinks=True,  # symlink to HF cache (no data duplication)
            resume_download=True,
            ignore_patterns=["*.pt.lock", "*.json.lock", "*.tmp"]
        )
        print(f"[models]   linked into {local_dir}")

    if args.dream_id:
        fetch(args.dream_id, args.dream_local)
    if args.d2f_lora_id:
        fetch(args.d2f_lora_id, args.d2f_lora_local)
    print("[models] Done.")
else:
    raise SystemExit(f"Unknown mode: {args.mode}")
PY
}

# =========================
# Step 1: Datasets (shared cache)
# =========================
echo "=== (1/4) Caching datasets to HF cache ==="
run_python --mode datasets --datasets_json "$DATASETS_JSON"
echo

# =========================
# Step 2: Git repos (project local)
# =========================
echo "=== (2/4) Cloning/Updating repos into $REPOS_ROOT ==="
if [ ! -d "$REPOS_ROOT/Dream/.git" ]; then
  git clone "$DREAM_REPO_URL" "$REPOS_ROOT/Dream"
else
  git -C "$REPOS_ROOT/Dream" pull --ff-only || true
fi

if [ ! -d "$REPOS_ROOT/Discrete-Diffusion-Forcing/.git" ]; then
  git clone "$D2F_REPO_URL" "$REPOS_ROOT/Discrete-Diffusion-Forcing"
else
  git -C "$REPOS_ROOT/Discrete-Diffusion-Forcing" pull --ff-only || true
fi
echo

# =========================
# Step 3: Models via HF Hub (symlinked local dirs)
# =========================
echo "=== (3/4) Download/link Dream base & D2F LoRA into project ==="
run_python --mode models \
  --dream_id "$DREAM_MODEL_ID" \
  --d2f_lora_id "$D2F_LORA_ID" \
  --dream_local "$DREAM_LOCAL_DIR" \
  --d2f_lora_local "$D2F_LORA_LOCAL_DIR"
echo

# =========================
# Step 4: Summary
# =========================
echo "=== (4/4) Summary ==="
cat <<EOF
Project root:     $ROOT
Repos:            $REPOS_ROOT
Checkpoints:      $CKPT_ROOT
LoRA:             $LORA_ROOT

Dream repo:       $REPOS_ROOT/Dream
D2F repo:         $REPOS_ROOT/Discrete-Diffusion-Forcing

Dream 7B local:   $DREAM_LOCAL_DIR  -> symlinked to HF cache
D2F LoRA local:   $D2F_LORA_LOCAL_DIR -> symlinked to HF cache

HF cache in use:
  HF_HOME               = $HF_HOME
  HUGGINGFACE_HUB_CACHE = $HUGGINGFACE_HUB_CACHE
  HF_DATASETS_CACHE     = $HF_DATASETS_CACHE
  TRANSFORMERS_CACHE    = $TRANSFORMERS_CACHE
  HF_MODULES_CACHE      = $HF_MODULES_CACHE
EOF

echo "Done."