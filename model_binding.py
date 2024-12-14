# model_binding.py
import numpy as np
from PIL import Image
import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
import os

# initialize the device to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# paths to the pretrained base model and the ControlNet models
base_model_path = "runwayml/stable-diffusion-v1-5"

# baseline ControlNet model name on HuggingFace
baseline_controlnet_model_name = "lllyasviel/control_v11p_sd15_lineart"

# trained ControlNet model path
trained_controlnet_path = "diffusers/examples/controlnet/output/checkpoint-26000-nolambda/controlnet/"

# Check if trained ControlNet model exists
if not os.path.exists(trained_controlnet_path):
    raise FileNotFoundError(f"Trained ControlNet model not found at '{trained_controlnet_path}'.")
print(f"Loading baseline ControlNet model from '{baseline_controlnet_model_name}'...")
baseline_controlnet = ControlNetModel.from_pretrained(
    baseline_controlnet_model_name,
    torch_dtype=torch.float16  # Use float16 for GPU
)
print("Baseline ControlNet model loaded successfully.\n")
print(f"Loading trained ControlNet model from '{trained_controlnet_path}'...")
trained_controlnet = ControlNetModel.from_pretrained(
    trained_controlnet_path,
    torch_dtype=torch.float16  # Use float16 for GPU
)
print("Trained ControlNet model loaded successfully.\n")
print("Initializing baseline pipeline...")
baseline_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=baseline_controlnet,
    torch_dtype=torch.float16,  # Use float16 for GPU
)
baseline_pipe = baseline_pipe.to(device)
print("Baseline pipeline initialized.\n")

# Init the trained pipeline
print("Initializing trained pipeline...")
trained_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=trained_controlnet,
    torch_dtype=torch.float16,  # Use float16 for GPU
)
trained_pipe = trained_pipe.to(device)
print("Trained pipeline initialized.\n")
print("Optimizing pipelines for performance...")
baseline_pipe.scheduler = UniPCMultistepScheduler.from_config(baseline_pipe.scheduler.config)
trained_pipe.scheduler = UniPCMultistepScheduler.from_config(trained_pipe.scheduler.config)
print("Pipelines optimized.\n")

def prompt_model(text_prompt: str, sketch_path: str, additional_image_path: str = None) -> np.ndarray:
    """
    Generates an image based on the user's text prompt and sketch.
    Optionally, an additional image can be provided for conditioning.

    Args:
        text_prompt(str): The text prompt provided by the user.
        sketch_path(str): The file path to the saved sketch image.
        additional_image_path (str, optional): The file path to the additional image.

    Returns:
        np.ndarray: The generated image as a NumPy array (RGB).
    """
    # Load the sketch image from file
    sketch_image = Image.open(sketch_path).convert("RGB")
    sketch_image = sketch_image.resize((512,512))  # Ensure it matches model input size

    if additional_image_path:
        # Load the additional image
        additional_image = Image.open(additional_image_path).convert("RGB")
        additional_image = additional_image.resize((512, 512))  # Ensure it matches model input size
        
        pass 

    # Generate image using the trained ControlNet pipeline
    print("Generating image with trained ControlNet...")
    generated_image = trained_pipe(
        prompt=text_prompt,
        image=sketch_image,
        num_inference_steps=50,
        guidance_scale=7.5,
    ).images[0]
    print("Image generation completed.\n")

    # Convert the generated PIL Image to a NumPy array
    generated_image_np = np.array(generated_image)

    return generated_image_np
