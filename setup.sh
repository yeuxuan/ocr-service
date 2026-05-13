#!/bin/bash
# ============================================================
# GLM-OCR Service — One-click Setup for Apple Silicon Mac
# Prerequisites: Homebrew installed
# Usage: git clone <repo> && cd ocr-service && ./setup.sh
# ============================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

step() { echo -e "\n${BOLD}[$1/$TOTAL] $2${NC}"; }
ok()   { echo -e "  ${GREEN}✓ $1${NC}"; }
warn() { echo -e "  ${YELLOW}! $1${NC}"; }

TOTAL=6

# ──────────────────────────────────────
# 1. Check platform
# ──────────────────────────────────────
step 1 "Checking platform..."

if [[ "$(uname)" != "Darwin" ]]; then
    echo -e "${RED}This script is for macOS only.${NC}"
    exit 1
fi

ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${RED}Apple Silicon (arm64) required, got: $ARCH${NC}"
    exit 1
fi
ok "macOS arm64"

if ! command -v brew &>/dev/null; then
    echo -e "${RED}Homebrew not found. Install: https://brew.sh${NC}"
    exit 1
fi
ok "Homebrew"

# ──────────────────────────────────────
# 2. Install Python 3.12
# ──────────────────────────────────────
step 2 "Ensuring Python 3.12..."

PYTHON312=""
for p in /opt/homebrew/opt/python@3.12/bin/python3.12 /usr/local/opt/python@3.12/bin/python3.12; do
    if [[ -x "$p" ]]; then
        PYTHON312="$p"
        break
    fi
done

if [[ -z "$PYTHON312" ]]; then
    echo "  Installing python@3.12 via Homebrew..."
    brew install python@3.12
    PYTHON312="$(brew --prefix python@3.12)/bin/python3.12"
fi
ok "Python $($PYTHON312 --version)"

# ──────────────────────────────────────
# 3. Create .venv-mlx (mlx-vlm server)
# ──────────────────────────────────────
step 3 "Setting up .venv-mlx (mlx-vlm server)..."

if [[ -d "$PROJECT_DIR/.venv-mlx" ]]; then
    warn "Already exists, skipping. Delete .venv-mlx to recreate."
else
    "$PYTHON312" -m venv "$PROJECT_DIR/.venv-mlx"
    (
        source "$PROJECT_DIR/.venv-mlx/bin/activate"
        pip install --upgrade pip
        pip install "mlx>=0.30" "mlx-lm>=0.30"
        pip install "git+https://github.com/Blaizzy/mlx-vlm.git"
        pip install torch torchvision
    )
    ok "mlx-vlm environment ready"
fi

# ──────────────────────────────────────
# 4. Create .venv-sdk (GLM-OCR SDK + API)
# ──────────────────────────────────────
step 4 "Setting up .venv-sdk (GLM-OCR SDK + FastAPI)..."

if [[ -d "$PROJECT_DIR/.venv-sdk" ]]; then
    warn "Already exists, skipping. Delete .venv-sdk to recreate."
else
    "$PYTHON312" -m venv "$PROJECT_DIR/.venv-sdk"
    (
        source "$PROJECT_DIR/.venv-sdk/bin/activate"
        pip install --upgrade pip
        pip install -e "$PROJECT_DIR/glm-ocr[layout]"
        pip install fastapi uvicorn python-multipart
    )
    ok "GLM-OCR SDK + FastAPI environment ready"
fi

# ──────────────────────────────────────
# 5. Pre-download models
# ──────────────────────────────────────
step 5 "Pre-downloading models (first run will be slow otherwise)..."

(
    source "$PROJECT_DIR/.venv-mlx/bin/activate"
    python -c "
from mlx_vlm import load
print('Downloading mlx-community/GLM-OCR-bf16...')
load('mlx-community/GLM-OCR-bf16', trust_remote_code=True)
print('Model cached.')
" 2>&1 | grep -v "^$"
) && ok "GLM-OCR-bf16 model cached" || warn "Model download failed — will retry on first start"

(
    source "$PROJECT_DIR/.venv-sdk/bin/activate"
    python -c "
from transformers import AutoModelForObjectDetection, AutoImageProcessor
name = 'PaddlePaddle/PP-DocLayoutV3_safetensors'
print('Downloading layout model...')
AutoImageProcessor.from_pretrained(name)
AutoModelForObjectDetection.from_pretrained(name)
print('Layout model cached.')
" 2>&1 | grep -v "^$"
) && ok "PP-DocLayoutV3 layout model cached" || warn "Layout model download failed — will retry on first start"

# ──────────────────────────────────────
# 6. Create data directories
# ──────────────────────────────────────
step 6 "Creating data directories..."

mkdir -p "$PROJECT_DIR/data/uploads" "$PROJECT_DIR/data/results"
ok "data/uploads, data/results"

# ──────────────────────────────────────
# Done
# ──────────────────────────────────────
echo ""
echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN} Setup complete!${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo -e "  Start:   ${CYAN}./ocr-service.sh start${NC}"
echo -e "  Stop:    ${CYAN}./ocr-service.sh stop${NC}"
echo -e "  Status:  ${CYAN}./ocr-service.sh status${NC}"
echo -e "  Logs:    ${CYAN}./ocr-service.sh logs${NC}"
echo ""
