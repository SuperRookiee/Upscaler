python3.10 -m venv .venv

source .venv/bin/activate

pip install --upgrade pip

pip install -r requirements.txt

# 기본 (원본 유지)
python upscale_sdx4.py -i input -o results --guidance 0

# 조금 더 샤프하게
python upscale_sdx4.py -i input -o results \
  --prompt "ultra detailed, high quality, sharp focus" \
  --guidance 1.0


옵션
설명
--guidance
0은 원본 유지, 0.5~2.0은 디테일 보강
--steps
기본 40단계, 높을수록 품질↑ 시간↑
torch.backends.mps.is_available()
True이면 GPU(M1/M2)로 실행 중
