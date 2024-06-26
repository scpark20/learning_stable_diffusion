accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/jisoo_lora.toml \
    --output_dir=/data/sd-results/jisoo_lora \
    --output_name=jisoo_lora \
    --save_model_as=safetensors  \
    --prior_loss_weight=1.0  \
    --max_train_steps=1000 \
    --learning_rate=1e-4  \
    --optimizer_type="AdamW8bit"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --network_train_unet_only \
    --cache_latents \
    --cache_latents_to_disk \
    --network_module=networks.lora