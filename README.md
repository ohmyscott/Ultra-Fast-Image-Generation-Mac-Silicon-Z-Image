# Ultra Fast Image Generation for Mac Silicon (Z-Image)

6B parameter AI image generation that actually runs fast on your Mac. No cloud. No GPU rental.

## Features

- **Ultra Fast**: 14 seconds for 512x512, 31 seconds for 768x768
- **Tiny Model**: Only 3.5GB (4-bit quantized from 24GB)
- **Runs Locally**: No API keys, no cloud costs
- **Gradio UI**: Simple web interface included
- **CUDA Support**: Experimental support for NVIDIA GPUs

## Benchmarks

### Mac M2 Max (32GB)

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **14s** | 2.08s/step |
| 768x768 | 7 | **31s** | 4.43s/step |

### Mac M3 (18GB)

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **74s** | 10.6s/step |
| 768x768 | 7 | **2.5 min** | 21.4s/step |

**RAM Usage**: ~15.5 GB

**Recommended**: 768x768 @ 7-8 steps for best quality/speed balance.

> **Note**: M2 Max has significantly more GPU cores (30-38) vs base M3 (10), which explains the speed difference. M3 Pro/Max should be much faster.

## Model

Uses [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) - a 4-bit quantized version of Alibaba's [Z-Image Turbo](https://github.com/Tongyi-MAI/Z-Image) (6B parameter diffusion transformer).

## Quick Start (1-Click)

1. Download/clone the repo
2. **Double-click `Launch.command`**
3. First run will auto-install dependencies (~5 min)
4. Browser opens automatically to the UI

That's it! The launcher handles everything.

## Manual Installation

```bash
# Clone the repo
git clone https://github.com/newideas99/Ultra-Fast-Image-Generation-Mac-Silicon-Z-Image.git
cd Ultra-Fast-Image-Generation-Mac-Silicon-Z-Image

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# (Optional) Set up proxy if you can't access Hugging Face directly
cp proxy.env.example proxy.env
# Edit proxy.env with your proxy settings

# Install dependencies
uv sync
```

## Usage

### Gradio Web UI (Easiest)

Double-click `Launch.command` or run:

```bash
uv run app.py
```

Then open http://localhost:7860 in your browser.

### Command Line

```bash
uv run generate.py "A beautiful sunset over mountains"
```

Options:
- `--height`: Image height (default: 512)
- `--width`: Image width (default: 512)
- `--steps`: Inference steps (default: 5)
- `--seed`: Random seed (-1 for random)
- `--output`: Output file path (default: output.png)

Example:
```bash
uv run generate.py "Cyberpunk city at night, neon lights" --height 768 --width 768 --steps 7 --seed 42 --output cyberpunk.png
```

## Performance Tips

- **Resolution**: 512x512 is fastest. 768x768 is good quality/speed balance.
- **Steps**: 5 steps is usually enough for good results. More steps = slower.
- **First run**: First generation is slower due to model loading (~30-60s). Subsequent generations are faster.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- **16GB+ unified memory required** (~15.5GB used during inference)

## Troubleshooting

### Model Download Issues

If you can't download models from Hugging Face, set up a proxy:

1. **Create proxy config:**
   ```bash
   cp proxy.env.example proxy.env
   ```

2. **Edit proxy.env with your settings:**
   ```bash
   # Example for local proxy
   export https_proxy=http://127.0.0.1:7897
   export http_proxy=http://127.0.0.1:7897
   export all_proxy=socks5://127.0.0.1:7897
   ```

3. **Install dependencies with SOCKS support:**
   ```bash
   uv sync  # This includes httpx[socks] for SOCKS proxy support
   ```

4. **Restart the app** - the proxy settings will be automatically loaded.

**Note:** If using SOCKS5 proxy, the required `socksio` package is already included in the dependencies.

### Common Issues

- **Error 502/Connection failed**: Usually a network connectivity issue, try the proxy setup above
- **Memory issues**: Ensure you have 16GB+ unified memory
- **Slow first generation**: Model loading takes 30-60s on first run

## Credits

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [SDNQ Quantization](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0

## License

See the original model licenses for usage terms.
