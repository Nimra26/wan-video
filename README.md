# wan-video

Local text-to-video with [Wan](https://github.com/Wan-Video) models running on ComfyUI.

- `--model wan21` (default): Wan 2.1 T2V 1.3B at 480p, then Real-ESRGAN 2x upscale. Faster, softer.
- `--model wan22`: Wan 2.2 TI2V 5B (Q6_K GGUF) at native 1280x704, 24 fps. Sharper, slower.

```
./venv/bin/python generate.py "a red sports car driving through a city"
./venv/bin/python generate.py "..." --model wan22
```

Videos are written to `outputs/`.

## Setup

```
git clone https://github.com/comfyanonymous/ComfyUI.git    # tested at 7a28332
git clone https://github.com/city96/ComfyUI-GGUF.git ComfyUI/custom_nodes/ComfyUI-GGUF   # tested at 6ea2651
python3.12 -m venv venv
./venv/bin/pip install -r ComfyUI/requirements.txt -r ComfyUI/custom_nodes/ComfyUI-GGUF/requirements.txt
```

Model files go under `ComfyUI/models/`:

| Folder | File |
|---|---|
| `diffusion_models/` | `wan2.1_t2v_1.3B_fp16.safetensors`, `Wan2.2-TI2V-5B-Q6_K.gguf` |
| `text_encoders/` | `umt5-xxl-encoder-Q4_K_M.gguf` |
| `vae/` | `wan_2.1_vae.safetensors`, `wan2.2_vae.safetensors` |
| `upscale_models/` | `RealESRGAN_x2plus.pth` |
