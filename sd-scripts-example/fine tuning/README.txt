0. base checkpoint model download from CIVITAI

1. 현재 checkpoint의 결과 확인
diffusers run.ipynb

2. celeba 데이터 다운로드 from internet
https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
unzip celeba.zip -d /data

3. image preprocessing (crop, resize), make text
make celeba dataset.ipynb 실행
/data/sd-dataset/celeba 밑에 .jpg, .txt 데이터 있는지 확인

4. captions 파일들로 json파일 만들기

python finetune/merge_captions_to_metadata.py \
    /data/sd-dataset/celeba \
    toml/celeba.json \
    --caption_extension .txt \
    --full_path \
    --recursive

5. toml 작성
celeba.toml

6. do train
accelerate launch --num_cpu_threads_per_process 1 fine_tune.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/celeba.toml \
    --output_dir=/data/sd-results/celeba \
    --output_name=celeba \
    --save_model_as=safetensors  \
    --max_train_steps=100000 \
    --save_every_n_steps=1000 \
    --learning_rate=1e-5  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing

7. fine-tuning된 checkpoint의 결과 확인
diffusers run.ipynb