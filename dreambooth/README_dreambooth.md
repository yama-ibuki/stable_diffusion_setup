## DreamBoothで画像を生成する

1. cdコマンドで`diffusers`から`./examples/dreambooth`ディレクトリに移動する

    ```
    cd ./examples/dreambooth
    ```

2. DreamBoothの実行に必要なモジュールをインストールする

    ```
    pip install -r requirements.txt && pip install -r requirements_sdxl.txt
    ```

3. `accelerate config`でサーバーの実行環境を定義する。詳しくは[Acclerate environment](https://github.com/huggingface/accelerate/)を確認してください

    ```
    accelerate config
    ```

    上記のコマンドを入力すると、実行環境を定義するために選択肢に答える必要がある。基本的な答え方は順に、

    1. `This machine`
    2. `NO distributed training`
    3. `NO`
    4. `NO`
    5. `NO`
    6. `0`
    7. `no`

4. DreamBoothのデータセット作成

    トレーニングに使用する画像をフォルダに保存する。

5. DreamBoothのトレーニング

    #### DreamBoothの場合

    ```
    python train_dreambooth.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5"  \
    --instance_data_dir="./training_images" \
    --output_dir="Training_DreamBooth" \
    --instance_prompt="" \
    --resolution=512 \
    --train_batch_size=1 \
    --learning_rate=5e-6 \
    --max_train_steps=400 \
    ```

    #### DreamBooth (Stable Diffusion XL)の場合

    ```
    accelerate launch train_dreambooth_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0"  \
    --instance_data_dir="training_images" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --output_dir="Training_DreamBooth_lora_xl" \
    --mixed_precision="fp16" \
    --instance_prompt="" \
    --resolution=512 \
    --train_batch_size=1 \
    --gradient_accumulation_steps=4 \
    --learning_rate=1e-5 \
    --report_to="wandb" \
    --lr_scheduler="constant" \
    --lr_warmup_steps=0 \
    --max_train_steps=500 \
    --seed="42"
    ```

    オプションの意味：

    - `--pretrained_model_name_or_path`: 事前学習済みモデルを指定する。その他の事前学習済みモデルは[Hugging Face](https://huggingface.co/models)に公開されています

    - `--instance_data_dir`: トレーニングに使用するデータセットを指定する

    - `--output_dir`: モデルを保存するディレクトリを指定する

    - `--instance_prompt`: 学習時に使用するプロンプトを指定する

    #### 詳しいハイパーパラメータの設定は[こちらから](https://huggingface.co/blog/dreambooth)


6. 画像を生成する

    #### DreamBoothの場合

    ```
    python inference.py -n 50
    ```

    画像を生成する際は[inference.py](inference.py)の`model_id`と`prompt`を直接編集してください

    #### DreamBooth (Stable Diffusion XL)の場合

    ```
    python inference_lora_sdxl.py -n 50
    ```

    画像を生成する際は[inference_lora_sdxl.py](inference_lora_sdxl.py)の`model_id`と`prompt`を直接編集してください

    オプションの意味：

    - `-n`: 生成する画像の枚数を指定する

## 参考文献

- [DreamBoothのGitHubページ](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

- [DreamBooth (Stable Diffusion XL)のGitHubページ](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md)

- [DreamBoothについての解説](https://huggingface.co/docs/diffusers/training/dreambooth)

- [DreamBoothのトレーニングのコツ](https://huggingface.co/blog/dreambooth)
