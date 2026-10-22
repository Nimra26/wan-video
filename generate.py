#!/usr/bin/env python3
"""Local text-to-video, 1280x720 HD MP4.

  --model wan21 (default): Wan 2.1 1.3B at 480p -> Real-ESRGAN 2x upscale. Faster, softer.
  --model wan22: Wan 2.2 TI2V 5B (Q6_K GGUF) at native 1280x704, 24 fps. Sharper, slower.

Runs ComfyUI headless in the background, submits the workflow, waits, and
copies the finished video into ./outputs/.

Usage:
    ./venv/bin/python generate.py "A red fox running through snow at sunrise"
    ./venv/bin/python generate.py "..." --seconds 3 --steps 30 --seed 42
    ./venv/bin/python generate.py "..." --model wan22
"""
import argparse
import json
import random
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = ROOT / "ComfyUI"
PYTHON = ROOT / "venv" / "bin" / "python"
OUT_DIR = ROOT / "outputs"
HOST = "127.0.0.1:8188"
FPS = {"wan21": 16, "wan22": 24}  # native frame rates

NEGATIVE = (
    "blurry, low quality, worst quality, jpeg artifacts, distorted, deformed, "
    "static, still image, subtitles, text, watermark, extra fingers, bad hands, "
    "soft focus, out of focus, motion blur, hazy, foggy, depth of field, bokeh"
)
# Appended to every prompt; the 1.3B model drifts soft without it.
SHARP = "sharp focus, highly detailed, crisp, clear, high resolution"


def build_workflow(model, prompt, frames, steps, seed, width, height, out_w, out_h, prefix):
    wf = {
        "1": {"class_type": "UNETLoader",
              "inputs": {"unet_name": "wan2.1_t2v_1.3B_fp16.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoaderGGUF",
              "inputs": {"clip_name": "umt5-xxl-encoder-Q4_K_M.gguf", "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"text": f"{prompt}, {SHARP}", "clip": ["2", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": NEGATIVE, "clip": ["2", 0]}},
        "6": {"class_type": "EmptyHunyuanLatentVideo",
              "inputs": {"width": width, "height": height, "length": frames, "batch_size": 1}},
        "7": {"class_type": "ModelSamplingSD3", "inputs": {"shift": 8.0, "model": ["1", 0]}},
        "8": {"class_type": "KSampler",
              "inputs": {"seed": seed, "steps": steps, "cfg": 6.0, "sampler_name": "uni_pc",
                         "scheduler": "simple", "denoise": 1.0, "model": ["7", 0],
                         "positive": ["4", 0], "negative": ["5", 0], "latent_image": ["6", 0]}},
        "9": {"class_type": "VAEDecode", "inputs": {"samples": ["8", 0], "vae": ["3", 0]}},
        "10": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": "RealESRGAN_x2plus.pth"}},
        "11": {"class_type": "ImageUpscaleWithModel",
               "inputs": {"upscale_model": ["10", 0], "image": ["9", 0]}},
        "12": {"class_type": "ImageScale",
               "inputs": {"image": ["11", 0], "upscale_method": "lanczos",
                          "width": out_w, "height": out_h, "crop": "disabled"}},
        "15": {"class_type": "ImageSharpen",
               "inputs": {"image": ["12", 0], "sharpen_radius": 1, "sigma": 0.6, "alpha": 0.7}},
        "13": {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": float(FPS[model])}},
        "14": {"class_type": "SaveVideo",
               "inputs": {"video": ["13", 0], "filename_prefix": prefix,
                          "format": "mp4", "codec": "auto"}},
    }
    if model == "wan22":
        # 5B renders HD natively: no ESRGAN, just a lanczos nudge from 704 to 720 rows.
        wf["1"] = {"class_type": "UnetLoaderGGUF",
                   "inputs": {"unet_name": "Wan2.2-TI2V-5B-Q6_K.gguf"}}
        wf["3"]["inputs"]["vae_name"] = "wan2.2_vae.safetensors"
        wf["6"] = {"class_type": "Wan22ImageToVideoLatent",
                   "inputs": {"vae": ["3", 0], "width": width, "height": height,
                              "length": frames, "batch_size": 1}}
        wf["8"]["inputs"]["cfg"] = 5.0
        wf["12"]["inputs"]["image"] = ["9", 0]
        wf["15"]["inputs"]["alpha"] = 0.4
        del wf["10"], wf["11"]
    return wf


def http(method, path, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(f"http://{HOST}{path}", data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp)


def server_up():
    try:
        http("GET", "/system_stats")
        return True
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def start_server():
    log = open(ROOT / "comfyui.log", "w")
    proc = subprocess.Popen(
        [str(PYTHON), "main.py", "--listen", "127.0.0.1", "--port", "8188", "--lowvram"],
        cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(180):
        if proc.poll() is not None:
            sys.exit(f"ComfyUI exited early; see {ROOT / 'comfyui.log'}")
        if server_up():
            return proc
        time.sleep(1)
    proc.terminate()
    sys.exit(f"ComfyUI didn't start in 3 minutes; see {ROOT / 'comfyui.log'}")


def main():
    p = argparse.ArgumentParser(description="Local Wan 2.1 text-to-video, upscaled to HD")
    p.add_argument("prompt")
    p.add_argument("--seconds", type=float, default=2, help="clip length (default 2; longer is much slower)")
    p.add_argument("--steps", type=int, default=30, help="sampling steps (default 30)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--portrait", action="store_true", help="720x1280 instead of 1280x720")
    p.add_argument("--model", choices=["wan21", "wan22"], default="wan21",
                   help="wan21 = 1.3B + upscale (faster); wan22 = 5B native HD (sharper, slower)")
    args = p.parse_args()

    # Wan needs 4n+1 frames.
    fps = FPS[args.model]
    frames = max(5, round(args.seconds * fps / 4) * 4 + 1)
    seed = args.seed if args.seed is not None else random.randint(0, 2**32 - 1)
    if args.model == "wan22":
        width, height, out_w, out_h = (704, 1280, 720, 1280) if args.portrait else (1280, 704, 1280, 720)
    else:
        width, height, out_w, out_h = (480, 832, 720, 1280) if args.portrait else (832, 480, 1280, 720)
    # e.g. 2026-10-22_13-05_red-sports-car-driving-through-a-city_wan21.mp4
    slug = "-".join(re.findall(r"[a-z0-9]+", args.prompt.lower())[:6]) or "video"
    name = f"{time.strftime('%Y-%m-%d_%H-%M')}_{slug}_{args.model}"
    prefix = f"localvideo/{name}"

    proc = None
    if not server_up():
        print("Starting ComfyUI (first start can take a minute)...")
        proc = start_server()

    try:
        wf = build_workflow(args.model, args.prompt, frames, args.steps, seed, width, height, out_w, out_h, prefix)
        try:
            prompt_id = http("POST", "/prompt", {"prompt": wf})["prompt_id"]
        except urllib.error.HTTPError as e:
            sys.exit(f"ComfyUI rejected the workflow: {e.read().decode(errors='replace')}")
        print(f"Generating {frames} frames ({frames / fps:.1f}s) with {args.model}, {args.steps} steps, seed {seed}.")
        print(f"This is slow on a GTX 1070; progress is in {ROOT / 'comfyui.log'}")

        start = time.time()
        while True:
            try:
                hist = http("GET", f"/history/{prompt_id}").get(prompt_id)
            except (urllib.error.URLError, OSError):
                print()
                sys.exit("ComfyUI stopped mid-run, most likely out of RAM. Close other apps "
                         f"(browser), then retry. Check `journalctl -k | grep -i oom` and {ROOT / 'comfyui.log'}")
            if hist:
                status = hist.get("status", {})
                if status.get("status_str") == "error":
                    msgs = [m for m in status.get("messages", []) if m[0] == "execution_error"]
                    sys.exit(f"Generation failed: {json.dumps(msgs, indent=2)[:3000]}")
                if status.get("completed"):
                    break
            print(f"\r  elapsed {int(time.time() - start)}s", end="", flush=True)
            time.sleep(5)
        print()

        OUT_DIR.mkdir(exist_ok=True)
        saved = []
        for node_out in hist["outputs"].values():
            for item in node_out.get("images", []) + node_out.get("videos", []):
                src = COMFY / "output" / item.get("subfolder", "") / item["filename"]
                if src.suffix == ".mp4":
                    dst = OUT_DIR / f"{name}.mp4"
                    n = 2
                    while dst.exists():
                        dst = OUT_DIR / f"{name}_{n}.mp4"
                        n += 1
                    shutil.copy2(src, dst)
                    saved.append(dst)
        if not saved:
            sys.exit(f"Finished but found no MP4 in outputs: {json.dumps(hist['outputs'])[:2000]}")
        for s in saved:
            print(f"Done in {int(time.time() - start)}s: {s}")
    finally:
        if proc:
            # Another generate.py may have queued work on this server; don't kill it.
            try:
                q = http("GET", "/queue")
                busy = q["queue_running"] or q["queue_pending"]
            except (urllib.error.URLError, OSError):
                busy = False
            if busy:
                print("Other jobs are still queued; leaving ComfyUI running.")
            else:
                proc.terminate()
                proc.wait(timeout=30)


if __name__ == "__main__":
    main()
