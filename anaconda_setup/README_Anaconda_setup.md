# Anacondaを用いたStable Diffusionのセットアップ方法

## 動作環境

以下の環境で動作済み

- Ubuntu 20.04
- Git 2.25.1
- Conda 22.9.0
- CUDA Drive (nvidia driver) 515.76

## セットアップ手順

1. サーバー(PC)にGit, Anaconda, NVIDIA Driverをインストール

2. サーバーのホームディレクトリ上に`huggingface/diffusers`をclone

    ```
    git clone https://github.com/huggingface/diffusers.git
    ```

3. diffusersディレクトリが作成されるので、cdコマンドでそのディレクトリに移動する

    ```
    cd diffusers
    ```

4. Anacondaで仮想環境を作成する

    ```
    conda create -n diffusers
    ```

5. 仮想環境をActivateする

    ```
    conda activate diffusers
    ```

6. Stable Diffusionの実行に必要なモジュールをインストールする

    ```
    pip install . && pip install wandb xformers bitsandbytes scipy
    ```

## Stable Diffusionの実行について

- このリポジトリではText-to-ImageとDreamBoothの実行方法について解説します。
詳しくは[Text-to-Imageで画像を生成する](text_to_image)、[DreamBoothで画像を生成する](dreambooth/)を確認してください

- その他のStable Diffusionのモデルについては、各々実行方法を調べてください

## 参考文献

- [diffusersのGitHubページ](https://github.com/NVlabs/stylegan3/)

- [Hugging Faceの公式ページ](https://github.com/NVlabs/stylegan3/issues/77)
