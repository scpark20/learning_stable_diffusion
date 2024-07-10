1. 데이터 수집 (Textual Inversion과 동일)
인터넷에서 원하는 오브젝트 5장을 다운로드 받습니다.

2. PNG 변환 (Textual Inversion과 동일)
데이터의 일관성을 위해 .png로 변환합니다. sd-scripts-example/dreambooth/raw images to png.ipynb 실행

3. Image to Text
“sd-scripts-example/textual_inversion/image to text.ipynb” 실행하여 text caption을 만들고 저장
(모델 다운로드 하는데 시간 오래 걸림)

BLIP2로 만든 prompt에 적절하게 rare한 identifier 추가
예) ‘a yellow and white stuffed shark with a big smile’
   -> ‘a yellow and white stuffed ohwx shark with a big smile’

4. Prior Dataset 만들기
“sd-scripts-example/dreambooth/make prior dataset.ipynb” 실행하여 identifier가 없는 prior 이미지들을 생성합니다.
(원래는 생성하지 않은 진짜 데이터셋 사용하는 것이 좋음)
Prior 이미지를 만들기 위한 prompt는 ChatGPT를 이용해서 만들 수 있습니다.

5. Dataset .toml 파일 작성
sd-scripts/toml에 .toml 파일 작성
sd-scripts-example/dreambooth/shark_db.toml 참고

6. Training
sd-scripts 디렉토리에서 다음과 같이 입력하여 트레이닝 실행
accelerate launch --num_cpu_threads_per_process 1 train_db.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/shark_db.toml  \
    --output_dir=/data/sd_results/shark_db \
    --output_name=shark_db \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0  \
    --max_train_steps=10000  \
    --save_every_n_steps=1000 \
    --learning_rate=1e-6  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing

7. 결과보기
sd-scripts-example/dreambooth/diffusers dreambooth run.ipynb 실행하여 결과 확인