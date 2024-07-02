1. 원하는 이미지 5장 download

2. PNG로 변환
raw images to png.ipynb

3. script 만들기
image to text.ipynb

4. training
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/shark_inversion.toml \
    --output_dir=/data/sd-results/shark_inversion \
    --output_name=shark_inversion \
    --save_model_as=safetensors   \
    --max_train_steps=10000  \
    --learning_rate=5e-6  \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --token_string=mychar4 \
    --init_word=shark \
    --num_vectors_per_token=4
    
    
# regularization version
accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/shark_inversion_prior.toml \
    --output_dir=/data/sd-results/shark_inversion_prior \
    --output_name=shark_inversion_prior \
    --save_model_as=safetensors   \
    --prior_loss_weight=1.0  \
    --max_train_steps=10000  \
    --learning_rate=5e-6  \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --token_string=mychar4 \
    --init_word=shark \
    --num_vectors_per_token=4