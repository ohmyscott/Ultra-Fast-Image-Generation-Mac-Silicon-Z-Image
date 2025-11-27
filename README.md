# Z-Image Turbo UINT4

Fast image generation on Apple Silicon Macs using the quantized Z-Image Turbo model.

## Features

- **Tiny Model Size**: Only 3.5GB (quantized from 24GB)
- **Fast Inference**: ~14-31 seconds per image depending on resolution
- **Low Memory**: ~15.5GB RAM usage
- **Gradio UI**: Simple web interface included

## Benchmarks

Tested on **Mac M2 Max**:

| Resolution | Steps | Time | Speed |
|------------|-------|------|-------|
| 512x512 | 7 | **14s** | 2.08s/step |
| 768x768 | 7 | **31s** | 4.43s/step |

**RAM Usage**: ~15.48 GB

**Recommended settings**: 768x768 @ 7-8 steps for best quality/speed balance.

## Model

Uses [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) - a 4-bit quantized version of Z-Image Turbo.

## Installation

```bash
# Clone the repo
git clone https://github.com/yourusername/zimage-turbo-uint4.git
cd zimage-turbo-uint4

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Gradio Web UI

```bash
python app.py
```

Then open http://localhost:7860 in your browser.

### Command Line

```bash
python generate.py "A beautiful sunset over mountains"
```

Options:
- `--height`: Image height (default: 512)
- `--width`: Image width (default: 512)
- `--steps`: Inference steps (default: 5)
- `--seed`: Random seed (-1 for random)
- `--output`: Output file path (default: output.png)

Example:
```bash
python generate.py "Cyberpunk city at night, neon lights" --height 768 --width 768 --steps 7 --seed 42 --output cyberpunk.png
```

## Performance Tips

- **Resolution**: 512x512 is fastest. 768x768 is good quality/speed balance.
- **Steps**: 5 steps is usually enough for good results. More steps = slower.
- **First run**: First generation is slower due to model loading (~30-60s). Subsequent generations are faster.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- Python 3.10+
- **16GB+ unified memory required** (~15.5GB used during inference)

## Credits

- [Z-Image](https://github.com/Tongyi-MAI/Z-Image) by Alibaba
- [SDNQ Quantization](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32) by Disty0

## License

See the original model licenses for usage terms.
