import argparse, sys, os
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm
from diffusers import StableDiffusionUpscalePipeline

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i","--input", required=True)
    ap.add_argument("-o","--output", default="results")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--guidance", type=float, default=0.0)
    ap.add_argument("--steps", type=int, default=40)
    args = ap.parse_args()

    inp = Path(args.input)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        "stabilityai/stable-diffusion-x4-upscaler",
        torch_dtype=torch.float16 if device.type != "cpu" else torch.float32
    ).to(device)
    pipe.enable_attention_slicing()

    imgs = [f for f in inp.glob("*") if f.suffix.lower() in [".jpg",".png",".jpeg",".webp"]]
    if not imgs:
        print("no images found in", inp); sys.exit(1)

    for f in tqdm(imgs, desc="Upscaling"):
        lr = Image.open(f).convert("RGB")
        prompt = args.prompt if args.prompt.strip() else "high-resolution photo"
        with torch.inference_mode():
            res = pipe(prompt=prompt, image=lr,
                       guidance_scale=args.guidance, num_inference_steps=args.steps)

        out_path = out / f.name
        res.images[0].save(out_path)

    print("done ->", out)

if __name__ == "__main__":
    main()