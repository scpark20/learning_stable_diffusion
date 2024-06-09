accelerate launch train_controlnet_fixed.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/celeba_mesh_controlnet.toml \
    --output_dir=/data/sd-results/celeba_mesh_controlnet \
    --output_name=celeba_mesh_controlnet \
    --save_model_as=safetensors  \
    --max_train_steps=100000 \
    --save_every_n_steps=10000 \
    --train_batch_size 1 \
    --learning_rate=2e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --lowram \
    --persistent_data_loader_workers \
    --max_data_loader_n_workers 1 \
    --gradient_checkpointing
    
### Diffusers    
    
accelerate launch train_controlnet.py \
  --pretrained_model_name_or_path="/data/sd_files/stable-diffusion-v1-5" \
  --output_dir="/data/sd-results/celeba_controlnet" \
  --train_data_dir="/data/sd-dataset/celeba_controlnet" \
  --resolution=512 \
  --learning_rate=1e-5 \
  --train_batch_size=2 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam
