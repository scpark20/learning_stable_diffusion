1. ControlNet Dataset 만들기
“sd-scripts-example/contolnet/celeba landmarks dataset create.ipynb” 실행
CelebA의 얼굴 이미지를 mediapipe 라이브러리를 이용하여 face mesh 이미지 생성

2. Dataset .toml 파일 만들기
sd-scripts/toml/celeba_mesh_controlnet.toml 파일을 만들어 image와 condition 정보를 저장

3. do train
accelerate launch train_controlnet_fixed.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/celeba_mesh_controlnet.toml \
    --output_dir=/data/sd_results/celeba_mesh_controlnet \
    --output_name=celeba_mesh_controlnet \
    --save_model_as=safetensors  \
    --max_train_steps=100000 \
    --save_every_n_steps=1000 \
    --learning_rate=2e-4 \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16" \
    --gradient_checkpointing
    
4. 트레이닝 결과 확인
“sd-scripts-example/controlnet/diffusers controlnet run.ipynb” 실행