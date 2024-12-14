from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler
)
from diffusers.utils import load_image
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import os

base_model_path = "runwayml/stable-diffusion-v1-5"

baseline_controlnet_model_name = "lllyasviel/control_v11p_sd15_lineart"

trained_controlnet_path = "./output/checkpoint-26000-nolambda/controlnet"

print(f"Loading baseline ControlNet model from '{baseline_controlnet_model_name}'...")
baseline_controlnet = ControlNetModel.from_pretrained(
    baseline_controlnet_model_name,
    torch_dtype=torch.float16
)
print("Baseline ControlNet model loaded successfully.\n")

if not os.path.exists(trained_controlnet_path):
    raise FileNotFoundError(f"Trained ControlNet model not found at '{trained_controlnet_path}'.")

print(f"Loading trained ControlNet model from '{trained_controlnet_path}'...")
trained_controlnet = ControlNetModel.from_pretrained(
    trained_controlnet_path,
    torch_dtype=torch.float16
)
print("Trained ControlNet model loaded successfully.\n")

print("Initializing baseline pipeline...")
baseline_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=baseline_controlnet,
    torch_dtype=torch.float16,
)
print("Baseline pipeline initialized.\n")

print("Initializing trained pipeline...")
trained_pipe = StableDiffusionControlNetPipeline.from_pretrained(
    base_model_path,
    controlnet=trained_controlnet,
    torch_dtype=torch.float16,
)
print("Trained pipeline initialized.\n")

print("Optimizing pipelines for performance...")
baseline_pipe.scheduler = UniPCMultistepScheduler.from_config(baseline_pipe.scheduler.config)
baseline_pipe.enable_xformers_memory_efficient_attention()
baseline_pipe.enable_model_cpu_offload()
print("Baseline pipeline optimized.\n")

trained_pipe.scheduler = UniPCMultistepScheduler.from_config(trained_pipe.scheduler.config)
trained_pipe.enable_xformers_memory_efficient_attention()
trained_pipe.enable_model_cpu_offload()
print("Trained pipeline optimized.\n")

conditioning_image_paths = [
    "./test_images/testimage1.png",
    "./test_images/testimage2.png",
    "./test_images/testimage3.png",
    "./test_images/testimage4.png"
]

for img_path in conditioning_image_paths:
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Conditioning image not found at '{img_path}'.")

prompt = "A horse drawn carriage with two people sitting on it art painting"
num_inference_steps = 50
guidance_scale = 7.5
generator = torch.manual_seed(42)

output_dir = "./generated_images"
os.makedirs(output_dir, exist_ok=True)

def generate_image(pipe, prompt, conditioning_image, generator, num_inference_steps, guidance_scale):
    image = pipe(
        prompt=prompt,
        image=conditioning_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return image

def create_multi_row_collage(conditioning_imgs, baseline_imgs, model_imgs, save_path):
    """
    Creates a collage where each row corresponds to a conditioning image, its baseline generated image,
    and the model-generated image side by side.

    Parameters:
    - conditioning_imgs: List of PIL Images (Conditioning Images)
    - baseline_imgs: List of PIL Images (Baseline Generated Images)
    - model_imgs: List of PIL Images (Model Generated Images)
    - save_path: Path to save the collage image
    """
    num_rows = len(conditioning_imgs)
    num_columns = 3

    base_size = (512, 512)

    column_titles = ["Conditioning Image", "Baseline", "Model"]

    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    spacing = 10
    title_height = 30
    spacing_between_rows = 20

    total_width = (base_size[0] * num_columns) + (spacing * (num_columns + 1))
    total_height = (base_size[1] + title_height + spacing_between_rows) * num_rows + spacing_between_rows

    collage = Image.new("RGB", (total_width, total_height), color=(255, 255, 255))

    for row_idx in range(num_rows):
        y_offset = spacing_between_rows + row_idx * (base_size[1] + title_height + spacing_between_rows)

        for col_idx in range(num_columns):
            x_offset = spacing + col_idx * (base_size[0] + spacing)

            if col_idx == 0:
                img = conditioning_imgs[row_idx].resize(base_size)
                title = column_titles[col_idx]
            elif col_idx == 1:
                img = baseline_imgs[row_idx].resize(base_size)
                title = column_titles[col_idx]
            else:
                img = model_imgs[row_idx].resize(base_size)
                title = column_titles[col_idx]

            collage.paste(img, (x_offset, y_offset + title_height))

            draw = ImageDraw.Draw(collage)
            bbox = draw.textbbox((0, 0), title, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            text_x = x_offset + (base_size[0] - text_width) // 2
            text_y = y_offset

            draw.text((text_x, text_y), title, fill=(0, 0, 0), font=font)

    collage.save(save_path)
    print(f"Saved multi-row collage to '{save_path}'.")

all_conditioning_images = []
all_baseline_images = []
all_model_images = []

for img_path in conditioning_image_paths:
    conditioning_img = load_image(img_path)
    all_conditioning_images.append(conditioning_img)

    print(f"Generating baseline image for '{img_path}'...")
    baseline_img = generate_image(
        baseline_pipe,
        prompt,
        conditioning_img,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    baseline_output_path = os.path.join(output_dir, f"baseline_{os.path.basename(img_path).split('.')[0]}.jpg")
    baseline_img.save(baseline_output_path)
    print(f"Saved baseline image to '{baseline_output_path}'.\n")
    all_baseline_images.append(baseline_img)

    print(f"Generating model image for '{img_path}'...")
    model_img = generate_image(
        trained_pipe,
        prompt,
        conditioning_img,
        generator=generator,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    model_output_path = os.path.join(output_dir, f"model_output_{os.path.basename(img_path).split('.')[0]}.jpg")
    model_img.save(model_output_path)
    print(f"Saved model-generated image to '{model_output_path}'.\n")
    all_model_images.append(model_img)

collage_path = os.path.join(output_dir, "three_column_collage.jpg")
create_multi_row_collage(
    conditioning_imgs=all_conditioning_images,
    baseline_imgs=all_baseline_images,
    model_imgs=all_model_images,
    save_path=collage_path
)
print("Three-column collage created successfully.\n")
