1. prior dataset 만들기
make prior dataset.ipynb 실행

2. .toml 작성하기
jisoo_db.toml

3. do train!

accelerate launch --num_cpu_threads_per_process 1 train_db.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/jisoo_db.toml  \
    --output_dir=/data/sd-results/jisoo_db \
    --output_name=jisoo_db \
    --save_model_as=safetensors \
    --prior_loss_weight=1.0  \
    --max_train_steps=1000  \
    --learning_rate=1e-6  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing