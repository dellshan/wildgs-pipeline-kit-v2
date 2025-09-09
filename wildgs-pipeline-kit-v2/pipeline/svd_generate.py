#!/usr/bin/env python3
import argparse, os, torch
from PIL import Image
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--image", required=True)
    p.add_argument("--out", default="out.mp4")
    p.add_argument("--frames", type=int, default=25)
    p.add_argument("--fps", type=int, default=7)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--decode-chunk-size", type=int, default=8)
    p.add_argument("--motion-bucket-id", type=int, default=127)
    p.add_argument("--noise-aug-strength", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42, help="Set -1 to let Diffusers handle RNG")
    p.add_argument("--no-xformers", action="store_true")
    p.add_argument("--sequential-offload", action="store_true")
    p.add_argument("--cpu-offload", action="store_true")
    p.add_argument("--hf-token", default=os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    return p.parse_args()

def main():
    a = parse_args()
    print("\n[SVD] Loading pipeline…")
    pipe_kwargs = {"torch_dtype": torch.float16, "variant": "fp16"}
    if a.hf_token:
        pipe_kwargs["token"] = a.hf_token

    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt-1-1",
        **pipe_kwargs
    )

    # Attention backend
    if a.no_xformers:
        print("[SVD] Disabling xformers; using PyTorch attention.")
    else:
        try:
            import xformers  # noqa
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            print("[SVD] xformers not available; continuing without it.")

    # Prefer memory-efficient SDP kernels where possible
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

    # Offload vs single-GPU
    use_offload = a.sequential_offload or a.cpu_offload
    if a.sequential_offload:
        pipe.enable_sequential_cpu_offload()
        print("[SVD] enable_sequential_cpu_offload() ON")
    elif a.cpu_offload:
        pipe.enable_model_cpu_offload()
        print("[SVD] enable_model_cpu_offload() ON")
    else:
        pipe = pipe.to("cuda")

    pipe.enable_attention_slicing()

    # Prepare input
    img = Image.open(a.image).convert("RGB").resize((a.width, a.height))

    # RNG must match the pipeline execution device. When offloading, keep it on CPU.
    generator = None
    if a.seed is not None and a.seed >= 0:
        rng_device = torch.device("cpu") if use_offload else torch.device("cuda")
        generator = torch.Generator(device=rng_device).manual_seed(a.seed)

    print(f"[SVD] Running: {a.width}x{a.height}, frames={a.frames}, fps={a.fps}, "
          f"decode_chunk_size={a.decode_chunk_size}, motion_bucket_id={a.motion_bucket_id}")

    # Build call kwargs, only pass generator if we created one
    call_kwargs = dict(
        num_frames=a.frames,
        decode_chunk_size=a.decode_chunk_size,
        motion_bucket_id=a.motion_bucket_id,
        noise_aug_strength=a.noise_aug_strength,
    )
    if generator is not None:
        call_kwargs["generator"] = generator

    try:
        with torch.inference_mode():
            result = pipe(img, **call_kwargs)
    except torch.cuda.OutOfMemoryError as e:
        used = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0
        print(f"[ERR] CUDA OOM: {e}\n[DBG] peak allocated ~{used:.2f} GiB")
        print("[HINT] Reduce --width/--height or --frames; or add --sequential-offload / --cpu-offload.")
        raise

    frames = result.frames[0]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    export_to_video(frames, a.out, fps=a.fps)
    print("Saved", a.out)

if __name__ == "__main__":
    main()
