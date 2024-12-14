To train the model:
cd diffusers/examples/controlnet/
Follow the README instructions and install the requirements
This requires significant GPU power. Training worked on NVIDIA A100 40GB VRAM GPU

python train_controlnet_variable.py 	--pretrained_model_name_or_path=runwayml/stable-diffusion-v1-5 	--output_dir=./output/  --controlnet_model_name_or_path=ControlNet-1-1-preview/control_v11p_sd15_lineart   --cache_dir=~/cache/huggingface/datasets 	--seed=10 	--resolution=512 	--train_batch_size=8 	--num_train_epochs=5 	--max_train_steps=10000 	--checkpointing_steps=500 	--checkpoints_total_limit=5 	--gradient_checkpointing 	--dataloader_num_workers=0 	--logging_dir=logs 	--allow_tf32 	--report_to=wandb 	--mixed_precision=fp16 	--enable_xformers_memory_efficient_attention 	--dataset_name=wangherr/coco2017_caption_sketch 	--image_column=image 	--conditioning_image_column=conditioning_image 	--caption_column=text 	--max_train_samples=100000

To launch the drawing application (requires model to be trained):
python main.py
