0. celeba 데이터셋 다운로드
gdown https://drive.google.com/uc?id=1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ

1. celeba 데이터셋 만들기 (5분 이내)
fine tuning/make celeba dataset.ipynb 실행
/data/sd-dataset/celeba 안에 .jpg, .txt데이터 만들어졌나 확인

2. controlnet 데이터셋 만들기 (10-20분)
controlnet/celeba landmarks dataset create.ipynb 실행

3. do train
accelerate launch train_controlnet_fixed.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/celeba_mesh_controlnet.toml \
    --output_dir=/data/sd-results/celeba_mesh_controlnet \
    --output_name=celeba_mesh_controlnet \
    --save_model_as=safetensors  \
    --max_train_steps=100000 \
    --save_every_n_steps=10000 \
    --learning_rate=2e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --lowram \
    --persistent_data_loader_workers \
    --max_data_loader_n_workers 1 \
    --gradient_checkpointing
    
4. inference
controlnet/diffusers controlnet run.ipynb 실행