import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from pathlib import Path
from tqdm import tqdm

def colorize_image(input_path: str, output_path: str):
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    print(f"[🔧] Using device: {device}")

    # Stable Diffusion 모델 불러오기 (컬러화에 적합한 기본 모델)
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float32
    ).to(device)

    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    prompt = "restore and colorize an old black and white family photograph, realistic natural colors"
    img = Image.open(input_path).convert("RGB")

    print("[🎨] Colorizing in progress...")
    result = pipe(prompt=prompt, image=img, num_inference_steps=30, guidance_scale=7.5)
    colorized = result.images[0]

    colorized.save(output_path)
    print(f"[✅] Done! Saved to: {output_path}")

if __name__ == "__main__":
    input_dir = Path("input")
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    for file in tqdm(list(input_dir.glob("*.*"))):
        if file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            output_file = output_dir / f"{file.stem}_colorized{file.suffix}"
            colorize_image(str(file), str(output_file))