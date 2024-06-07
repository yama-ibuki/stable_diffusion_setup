## Text-to-Imageで画像を生成する

1. cdコマンドで`diffusers`から`./examples/text_to_image`ディレクトリに移動する

    ```
    cd ./examples/text_to_image
    ```

2. Text-to-Imageの実行に必要なモジュールをインストールする

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

4. Text-to-Imageのデータセット作成

    #### `create_caption.py`を使って画像にプロンプトを追加する（出力はcsvもしくはjsol）
    - 出力されたcsvはデータセットと同じフォルダに保存してください
    - 複数のクラス（画像）にプロンプトを追加することもできます。その際は全ての画像と出力されたcsvを1つのフォルダにまとめてください
    - データセットを作成する際は[create_caption.py](create_caption.py)を直接編集してください 

        ```
        python create_caption.py
        ```

5. Text-to-Imageのトレーニング

    #### Stable Diffusion with LoRAの場合

    ```
    accelerate launch --mixed_precision="fp16" train_text_to_image_lora.py \
    --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
    --train_data_dir="datasets" --caption_column="text" \
    --resolution=512 \
    --train_batch_size=1 \
    --num_train_epochs=2 --checkpointing_steps=1000 \
    --max_train_steps=10000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=4245 \
    --output_dir="Training_with_LoRA" \
    --validation_prompt="" --report_to="wandb"
    ```

    #### Stable Diffusion XL with LoRAの場合

    ```
    accelerate launch train_text_to_image_lora_sdxl.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
    --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
    --train_data_dir="" --caption_column="text" \
    --resolution=1024 \
    --train_batch_size=1 \
    --num_train_epochs=2 --checkpointing_steps=1000 \
    --max_train_steps=5000 \
    --learning_rate=1e-04 --lr_scheduler="constant" --lr_warmup_steps=0 \
    --seed=4245 \
    --output_dir="Training_XL_with_LoRA" \
    --train_text_encoder \
    --validation_prompt="" --report_to="wandb"
    ```

    オプションの意味：

    - `--pretrained_model_name_or_path`: 事前学習済みモデルを指定する。その他の事前学習済みモデルは[Hugging Face](https://huggingface.co/models)に公開されています

    - `--train_data_dir`: トレーニングに使用するデータセットを指定する

    - `--output_dir`: モデルを保存するディレクトリを指定する

    - `--validation_prompt`: 検証時に使用するプロンプトを指定する


6. 画像を生成する

    #### Stable Diffusion with LoRAの場合

    ```
    python inference_lora.py -n 50
    ```

    画像を生成する際は[inference_lora.py](inference_lora.py)の`model_id`と`prompt`を直接編集してください

    #### Stable Diffusion XL with LoRAの場合

    ```
    python inference_lora_sdxl.py -n 50
    ```

    画像を生成する際は[inference_lora_sdxl.py](inference_lora_sdxl.py)の`model_id`と`prompt`を直接編集してください

    オプションの意味：

    - `-n`: 生成する画像の枚数を指定する

## 参考文献

- [Stable Diffusion Text-to-ImageのGitHubページ](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image)

- [Stable Diffusion XL Text-to-ImageのGitHubページ](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/README_sdxl.md)