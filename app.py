"""
Z-Image Turbo UINT4 - Gradio Web Interface

Fast image generation on Apple Silicon using the quantized uint4 model.
"""

import os
os.environ["PYTORCH_MPS_FAST_MATH"] = "1"

import torch
import sdnq
import gradio as gr
from diffusers import ZImagePipeline

# Global pipeline and current device
pipe = None
current_device = None


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
        torch_dtype=dtype,
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

    generate_btn.click(
        fn=generate_image,
        inputs=[prompt, height, width, steps, seed, device],
        outputs=[output_image, seed_info],
    )


if __name__ == "__main__":
    demo.launch()
