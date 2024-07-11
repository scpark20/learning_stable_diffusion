# 실습 환경 설정

## 0. Repository 다운로드

```bash
git clone https://github.com/scpark20/learning_stable_diffusion.git lsd
cd lsd
git submodule update --init --recursive sd-scripts
```


## 1. Diffusers 작동 확인

- requirements 파일로 package 설치

```bash
# In lsd directory
pip install -r diffusers-example/requirements.txt
```

- 다음 jupyter notebook 파일 실행하여 동작 확인

diffusers-example/stable diffusion by diffusers.ipynb

## 2. SD-Scripts 작동 확인

- requirements 파일로 package 설치
```bash
# In lsd directory
pip install -r sd-scripts-example/requirements.txt
```

-checkpoint 다운로드 (/data 디렉토리가 존재해야 함)
```bash
# In any directory
mkdir -p /data/sd_files/checkpoint
wget -P /data/sd_files/checkpoint https://storage.googleapis.com/scpark20_lsd/beautifulRealistic_v7.safetensors
```

-data 다운로드 (/data 디렉토리가 존재해야 함)
```bash
# In any directory
mkdir -p /data/sd_dataset
wget -P /data/sd_dataset https://storage.googleapis.com/scpark20_lsd/jisoo_png.zip
unzip /data/sd_dataset/jisoo_png.zip -d /data/sd_dataset
```

-.toml 파일 만들기
```bash
# In lsd/sd-scripts directory
mkdir toml
cp ../sd-scripts-example/lora/jisoo_lora.toml toml/
```

-accelerate 설정하기
```bash
# In any directory
accelerate config
```

다음과 같이 입력
```bash
In which compute environment are you running?
➔ this machine
Which type of machine are you using?
➔  No distributed training
Do you want to run your training on CPU only?
➔ NO
Do you wish to optimize your script with torch dynamo?
➔ NO
Do you want to use DeepSpeed?
➔ NO
What GPU(s) (by id) should be used for training on this machine as a comma-seperated list?
➔ 원하는 GPU id 입력
Would you like to enable numa efficiency?
➔ NO
Do you wish to use FP16 or BF16 (mixed precision)?
➔  fp16
```

-Training 하기

```bash
# In lsd/sd-scripts directory
accelerate launch --num_cpu_threads_per_process 1 train_network.py \
    --pretrained_model_name_or_path="/data/sd_files/checkpoint/beautifulRealistic_v7.safetensors" \
    --dataset_config=toml/jisoo_lora.toml \
    --output_dir=/data/sd_results/jisoo_lora \
    --output_name=jisoo_lora \
    --save_model_as=safetensors  \
    --max_train_steps=100 \
    --learning_rate=1e-4  \
    --optimizer_type="AdamW"  \
    --mixed_precision="fp16"  \
    --gradient_checkpointing \
    --network_module=networks.lora
```

- 100 steps 트레이닝 후 /data/sd_results/jisoo_lora/jisoo_lora.safetensors 파일이 생기면 완료