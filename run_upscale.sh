#!/bin/bash
# ===============================================
# Stable Diffusion x4 Upscaler (MPS supported)
# ===============================================

# 🔹 원본 보존 모드 (사진)
# ./run_upscale.sh photo

# 🔹 디테일 강화 모드
# ./run_upscale.sh detail

# 🔹 애니메이션 / 일러스트 전용
# ./run_upscale.sh anime

# 🔹 샤프 강조 모드
# ./run_upscale.sh sharp


PROJECT_DIR="$HOME/Code/AI/ModernUpscale"
VENV_DIR="$PROJECT_DIR/.venv"
INPUT_DIR="$PROJECT_DIR/input"
OUTPUT_DIR="$PROJECT_DIR/results"

# ==============================
# 1️⃣ 프리셋 모드 설정
# ==============================
MODE=${1:-photo} # 기본 photo

PROMPT=""
GUIDANCE=0.0
STEPS=40

case "$MODE" in
  photo)
    PROMPT=""
    GUIDANCE=0.0
    ;;
  detail)
    PROMPT="ultra detailed, high quality, sharp texture, realistic lighting"
    GUIDANCE=1.0
    ;;
  anime)
    PROMPT="highly detailed anime style, vivid colors, crisp edges"
    GUIDANCE=1.2
    ;;
  sharp)
    PROMPT="super resolution, clear edges, ultra sharp focus"
    GUIDANCE=0.8
    ;;
  *)
    echo "❌ Unknown mode: $MODE"
    echo "Available modes: photo | detail | anime | sharp"
    exit 1
    ;;
esac

# ==============================
# 2️⃣ 환경 설정 및 설치
# ==============================
if [ ! -d "$VENV_DIR" ]; then
  echo "[🚀] 가상환경 생성 중..."
  python3.10 -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"

if [ ! -f "$PROJECT_DIR/requirements.txt" ]; then
  echo "[⚙️] requirements.txt 없음 — 기본 의존성 설치"
  cat <<EOF > "$PROJECT_DIR/requirements.txt"
torch>=2.1.0
diffusers>=0.27.0
transformers>=4.40.0
accelerate>=0.30.0
safetensors>=0.4.2
Pillow>=10.3.0
opencv-python>=4.8.1.78
tqdm>=4.66.4
EOF
fi

echo "[📦] 의존성 확인 중..."
pip install --upgrade pip > /dev/null
pip install -r "$PROJECT_DIR/requirements.txt" > /dev/null

mkdir -p "$INPUT_DIR" "$OUTPUT_DIR"

# ==============================
# 3️⃣ 실행
# ==============================
echo "[🎨] Upscaling 시작 (mode=$MODE)"
python "$PROJECT_DIR/upscale_sdx4.py" \
  -i "$INPUT_DIR" \
  -o "$OUTPUT_DIR" \
  --prompt "$PROMPT" \
  --guidance $GUIDANCE \
  --steps $STEPS

echo "[✅] 완료! 결과 폴더: $OUTPUT_DIR"