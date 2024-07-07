1. Base Checkpoint Download
Civitai에서 원하는 checkpoint를 다운로드 받습니다. 최대한 우리의 dataset과 성격이 비슷한 것을 고릅니다. (SD 1.5, Checkpoint로 검색)

2. 현재 checkpoint의 결과 확인
“sd-scripts-example/fine tuning/diffusers run.ipynb”를 실행하여 현재 checkpoint로 얻어지는 결과를 확인해봅니다.

3. CelebA dataset download

https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
에서 celeba.zip를 다운로드 받고 /data에 압축을 풉니다
unzip celeba.zip -d /data

4. Image Pre-processing
“Sd-scripts-examples/fine tuning/make celeba dataset.ipynb”을 실행합니다.

5. Caption .json 파일 만들기

sd-scripts 디렉토리 상에서 다음과 같이 명령
1.
mkdir toml
2.
python -m finetune.merge_captions_to_metadata \
    /data/sd_dataset/celeba \
    toml/celeba.json \
    --caption_extension .txt \
    --full_path \
    --recursive

6. Dataset .toml 파일 작성
sd-scripts/toml 디렉토리에 celeba.toml 파일 만들고 dataset 정보 입력
“sd-scripts-example/fine tuning/celeba.toml” 참고

7. Training 시작
sd-scripts 디렉토리에서 다음과 같이 입력 (1000 steps 15분 정도 소요)

accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/celeba.toml \
    --output_dir=/data/sd_results/celeba \
    --output_name=celeba \
    --save_model_as=safetensors  \
    --max_train_steps=100000 \
    --save_every_n_steps=1000 \
    --learning_rate=1e-5  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing

8. 트레이닝 결과 확인
sd-scripts-example/fine tuning/diffusers run.ipynb”를 실행하여 트레이닝된 결과를 확인합니다.
