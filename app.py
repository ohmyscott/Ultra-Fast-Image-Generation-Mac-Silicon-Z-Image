"""
Z-Image Turbo UINT4 - Gradio Web Interface

Fast image generation on Apple Silicon using the quantized uint4 model.
"""

import os
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

# Set up proxy if environment variables are available
proxy_vars = ['https_proxy', 'http_proxy', 'all_proxy']
for var in proxy_vars:
    if var in os.environ:
        os.environ[var.upper()] = os.environ[var]

import torch
import sdnq
import gradio as gr
from diffusers import ZImagePipeline

# Global pipeline and current device
pipe = None
current_device = None


# Image aspect ratio presets
ASPECT_RATIOS = {
    "1:1 Square": {"ratio": 1.0, "description": "Square format"},
    "4:3 Standard": {"ratio": 4/3, "description": "Standard photo"},
    "3:4 Portrait": {"ratio": 3/4, "description": "Vertical portrait"},
    "16:9 Widescreen": {"ratio": 16/9, "description": "Widescreen/landscape"},
    "9:16 Story": {"ratio": 9/16, "description": "Instagram story"},
    "3:2 Classic": {"ratio": 3/2, "description": "Classic 35mm"},
    "2:3 Classic Portrait": {"ratio": 2/3, "description": "Classic portrait"},
    "21:9 Cinema": {"ratio": 21/9, "description": "Ultra-wide cinema"},
}


def calculate_dimensions(ratio_name, max_size=1024):
    """Calculate height and width based on aspect ratio and max size constraint."""
    if ratio_name not in ASPECT_RATIOS:
        return 768, 768  # Default fallback

    ratio = ASPECT_RATIOS[ratio_name]["ratio"]

    if ratio >= 1.0:  # Landscape or square
        width = max_size
        height = int(max_size / ratio)
    else:  # Portrait
        height = max_size
        width = int(max_size * ratio)

    # Ensure both dimensions are multiples of 64 (model requirement)
    height = max(256, ((height + 31) // 32) * 32)
    width = max(256, ((width + 31) // 32) * 32)

    # Ensure we don't exceed max_size
    if height > max_size:
        height = max_size
        width = int(height * ratio)
        width = ((width + 31) // 32) * 32
    elif width > max_size:
        width = max_size
        height = int(width / ratio)
        height = ((height + 31) // 32) * 32

    return height, width


def update_image_dimensions(ratio_name):
    """Update height and width sliders based on selected ratio."""
    height, width = calculate_dimensions(ratio_name)
    return height, width


def create_aspect_ratio_html(ratios, selected_ratio="1:1 Square"):
    """Create HTML for visual aspect ratio selector."""
    html_content = """
    <style>
        .aspect-ratio-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(80px, 1fr));
            gap: 8px;
            margin: 10px 0;
        }
        .aspect-ratio-btn {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 8px;
            padding: 12px 8px;
            border: 2px solid #4a5568;
            border-radius: 8px;
            background: #1a202c;
            cursor: pointer;
            transition: all 0.2s ease;
            color: #e2e8f0;
            font-family: 'Courier New', monospace;
            font-size: 11px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .aspect-ratio-btn:hover {
            border-color: #3182ce;
            background: #2d3748;
        }
        .aspect-ratio-btn.selected {
            border-color: #63b3ed;
            background: #2b6cb0;
            color: #bee3f8;
        }
        .aspect-preview {
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }
        .aspect-box {
            border: 2px solid currentColor;
            background: currentColor;
            opacity: 0.2;
        }
        .aspect-center {
            position: absolute;
            width: 8px;
            height: 8px;
            background: currentColor;
            border-radius: 50%;
        }
        .ratio-label {
            font-weight: 600;
            font-size: 10px;
        }
    </style>

    <div class="aspect-ratio-grid">
    """

    for ratio_name, ratio_info in ratios.items():
        ratio_value = ratio_info["ratio"]
        selected_class = "selected" if ratio_name == selected_ratio else ""

        # Calculate visual dimensions for the preview box
        max_visual_size = 40  # Maximum dimension in pixels
        if ratio_value >= 1.0:  # Landscape or square
            visual_width = max_visual_size
            visual_height = max_visual_size / ratio_value
        else:  # Portrait
            visual_height = max_visual_size
            visual_width = max_visual_size * ratio_value

        # Ensure minimum size for visibility
        visual_width = max(visual_width, 12)
        visual_height = max(visual_height, 12)

        # Extract ratio text from name (e.g., "16:9" from "16:9 Widescreen")
        ratio_text = ratio_name.split()[0]

        html_content += f"""
        <div class="aspect-ratio-btn {selected_class}" onclick="selectAspectRatio('{ratio_name}')">
            <div class="aspect-preview">
                <div class="aspect-box" style="width: {visual_width}px; height: {visual_height}px;"></div>
                <div class="aspect-center"></div>
            </div>
            <span class="ratio-label">{ratio_text}</span>
        </div>
        """

    html_content += """
    </div>

    <script>
        function selectAspectRatio(ratioName) {
            // Update visual selection
            const buttons = document.querySelectorAll('.aspect-ratio-btn');
            buttons.forEach(btn => btn.classList.remove('selected'));
            event.currentTarget.classList.add('selected');

            // Trigger gradio update
            gradio_app.querySelector('#aspect_radio input[value="' + ratioName + '"]').click();
        }
    </script>
    """

    return html_content


def update_aspect_ratio_html(selected_ratio):
    """Update the HTML when aspect ratio changes."""
    return create_aspect_ratio_html(ASPECT_RATIOS, selected_ratio)


def get_available_devices():
    """Get list of available devices."""
    devices = []
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    devices.append("cpu")
    return devices


def load_pipeline(device="mps"):
    """Load the pipeline (cached globally)."""
    global pipe, current_device

    # Reload if device changed
    if pipe is not None and current_device == device:
        return pipe

    if pipe is not None:
        print(f"Switching device from {current_device} to {device}...")
        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"Loading Z-Image-Turbo UINT4 on {device}...")

    # Use float16 for CUDA, float32 for MPS/CPU
    dtype = torch.float16 if device == "cuda" else torch.float32

    pipe = ZImagePipeline.from_pretrained(
        "Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32",
        dtype=dtype,
        low_cpu_mem_usage=True,
    )

    pipe.to(device)
    pipe.enable_attention_slicing()

    if hasattr(pipe, "enable_vae_slicing"):
        pipe.enable_vae_slicing()

    if hasattr(getattr(pipe, "vae", None), "enable_tiling"):
        pipe.vae.enable_tiling()

    current_device = device
    print(f"Pipeline loaded on {device}!")
    return pipe


def generate_image(prompt, height, width, steps, seed, device):
    """Generate an image from the prompt."""
    pipe = load_pipeline(device)

    if seed == -1:
        seed = torch.randint(0, 2**32, (1,)).item()

    # Use appropriate generator for device
    if device == "cuda":
        generator = torch.Generator("cuda").manual_seed(int(seed))
    elif device == "mps":
        generator = torch.Generator("mps").manual_seed(int(seed))
    else:
        generator = torch.Generator().manual_seed(int(seed))

    with torch.inference_mode():
        image = pipe(
            prompt=prompt,
            height=int(height),
            width=int(width),
            num_inference_steps=int(steps),
            guidance_scale=0.0,
            generator=generator,
        ).images[0]

    return image, f"Seed: {seed} | Device: {device}"


# Get available devices at startup
available_devices = get_available_devices()
default_device = available_devices[0] if available_devices else "cpu"

# Create Gradio interface
with gr.Blocks(title="Z-Image Turbo UINT4") as demo:
    gr.Markdown("""
    # Z-Image Turbo UINT4

    Fast image generation using the quantized 3.5GB model.

    **Model:** [Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32](https://huggingface.co/Disty0/Z-Image-Turbo-SDNQ-uint4-svd-r32)
    """)

    with gr.Row():
        with gr.Column(scale=1):
            prompt = gr.Textbox(
                label="Prompt",
                placeholder="Describe the image you want to generate...",
                lines=3,
            )

            # Hidden radio button for aspect ratio (to maintain gradio state)
            aspect_ratio = gr.Radio(
                choices=list(ASPECT_RATIOS.keys()),
                value="1:1 Square",
                label="Aspect Ratio",
                visible=False,
                elem_id="aspect_radio"
            )

            # Visual aspect ratio selector
            aspect_ratio_html = gr.HTML(
                create_aspect_ratio_html(ASPECT_RATIOS, "1:1 Square"),
                label="Aspect Ratio",
                info="Select aspect ratio to auto-calculate dimensions"
            )

            with gr.Row():
                height = gr.Slider(256, 1024, value=768, step=64, label="Height")
                width = gr.Slider(256, 1024, value=768, step=64, label="Width")

            with gr.Row():
                steps = gr.Slider(1, 10, value=7, step=1, label="Steps")
                seed = gr.Number(value=-1, label="Seed (-1 = random)")

            with gr.Row():
                device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="Device",
                    info="MPS=Mac, CUDA=NVIDIA (experimental), CPU=slow"
                )

            generate_btn = gr.Button("Generate", variant="primary")
            seed_info = gr.Textbox(label="Generation Info", interactive=False)

        with gr.Column(scale=1):
            output_image = gr.Image(label="Generated Image", type="pil")

    # Examples
    gr.Examples(
        examples=[
            ["A majestic mountain landscape at sunset, dramatic lighting, cinematic"],
            ["Portrait of a young woman, soft studio lighting, professional photography"],
            ["Cyberpunk city street at night, neon lights, rain reflections"],
            ["A cute cat wearing a tiny hat, studio photo, soft lighting"],
            ["Abstract art, vibrant colors, fluid shapes, modern design"],
        ],
        inputs=[prompt],
    )

    # Aspect ratio change handler
    aspect_ratio.change(
        fn=update_image_dimensions,
        inputs=[aspect_ratio],
        outputs=[height, width],
    ).then(
        fn=update_aspect_ratio_html,
        inputs=[aspect_ratio],
        outputs=[aspect_ratio_html],
    )

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, seed, device],
        outputs=[output_image, seed_info],
    )


if __name__ == "__main__":
    demo.launch()
