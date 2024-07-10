1. 데이터 수집
인터넷에서 원하는 오브젝트 5장을 다운로드 받습니다.

2. PNG로 변환
데이터의 일관성을 위해 .png로 변환합니다.
sd-scripts-example/textual_inversion/raw images to png.ipynb 실행

3. Image to Text
“sd-scripts-example/textual_inversion/image to text.ipynb” 실행하여 text caption을 만들고 저장
(모델 다운로드 하는데 시간 오래 걸림)

BLIP2로 만든 prompt에 적절하게 user-defined char (‘mychar4’)를 추가 혹은 대체
예) ‘a yellow and white stuffed shark with a big smile’
   ‘a yellow and white stuffed mychar4 with a big smile’
   
4. Dataset .toml 파일 작성

5. Training

accelerate launch --num_cpu_threads_per_process 1 train_textual_inversion.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/shark_inversion.toml \
    --output_dir=/data/sd_results/shark_inversion \
    --output_name=shark_inversion \
    --save_model_as=safetensors   \
    --max_train_steps=5000  \
    --save_every_n_steps=1000 \
    --learning_rate=2e-2  \
    --optimizer_type="AdamW" \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --token_string=mychar4 \
    --init_word=shark \
    --num_vectors_per_token=4