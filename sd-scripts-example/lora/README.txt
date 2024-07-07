1. Data 수집
인터넷에서 크롤링을 하거나 직접 가져오는 방식으로 원하는 종류나 사람의 이미지를 수집
얼굴이나 포즈 등이 같을수록 적은 데이터로도 트레이닝 가능하고, 다양한 얼굴, 포즈의 경우 데이터가 많아야함
수십장 정도로도 트레이닝 가능하나 쓸만한 품질을 낳기 위해서는 수백-수천장 필요
얼굴 사진을 원할 경우 이미지에 찰 정도로 크게 자르는 작업이 필요 (stable diffusion은 작은 얼굴이 깨짐, VAE의 영향)

2. Image to PNG
Sd-scripts-example/lora/raw images to png.ipynb 실행
Sd-scripts에서 기본적인 이미지 파일들 .png, jpg 등을 지원하지만 일관성을 위해 .png 파일로 모두 변환

3. Image to Text
“sd-scripts-example/lora/image to text.ipynb” 실행하여 text caption을 만들고 저장
(모델 다운로드 하는데 시간 오래 걸림)
BLIP2 모델을 이용하며 필요에 따라 다른 다양한 모델을 이용 가능

4. Dataset .toml 파일 작성
sd-scripts/toml 디렉토리에 .toml 파일 작성
준비한 데이터 디렉토리를 datasets.subsets에 기록

5. Training 시작
sd-scripts 디렉토리에서 다음과 같이 입력

accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/jisoo_lora.toml \
    --output_dir=/data/sd_results/jisoo_lora \
    --output_name=jisoo_lora \
    --save_model_as=safetensors  \
    --max_train_steps=1000 \
    --learning_rate=1e-4  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --network_module=networks.lora

6. Training 결과 확인
diffusers run.ipynb, diffusers lora run.ipynb 비교