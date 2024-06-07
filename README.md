# Dockerを用いたStable Diffusionのセットアップ方法

### 注意点

- Stable DiffusionはGPUメモリを大量に消費するモデルのため、実行する際には12GB以上のGPUメモリを持つデバイスを使用してください

- 2023年9月時点での動作を確認していますが、`huggingface/diffusers`の今後の更新によってはセットアップに失敗する可能性があります。

- `huggingface/diffusers`をgit pullする際は`docker/diffusers-pytorch-cuda/Dockerfile`に書き込んだユーザー情報を削除してください

- Anacondaで環境構築する際は[Stable DiffusionをAnacondaで実行する](anaconda_setup/README_Anaconda_setup.md)を確認してください

## 動作環境

以下の環境で動作済み

- Ubuntu 20.04
- Git 2.25.1
- Docker 24.0.2
- CUDA Drive (nvidia driver) 515.76

## セットアップ手順

1. サーバー(PC)にGit, Docker, docker-compose, CUDAをインストール

2. サーバーのホームディレクトリ上に`huggingface/diffusers`をclone

    ```
    git clone https://github.com/huggingface/diffusers.git
    ```

3. diffusersディレクトリが作成されるので、cdコマンドでそのディレクトリに移動する

    ```
    cd diffusers
    ```

4. diffusersディレクトリ上にstable_diffusion_setupをclone

    ```
    git clone https://github.com/w034ff/stable_diffusion_setup.git
    ```

5. stable_diffusion_setupディレクトリが作成されるので、cdコマンドでそのディレクトリに移動する

    ```
    cd stable_diffusion_setup
    ```

6. 現在ログイン中のIDなどを`.env`に書き込む

    > **注意**: 
    > `.env`ファイルは隠しファイルとなっているので注意してください
    
    - 以下のコマンドでuser_name, uid, gidを調べる
    
        ```
        id
        ```
        
        出力例
        ```
        uid=9999(user_name) gid=9999(group_name) groups=9999(group_name)
        ```

    - `.env`に先ほど調べたuser_name, uid, gidを記述

        ```
        USER_NAME=user_name
        USER_UID=9999
        USER_GID=9999
        ```
        
    > **注意**: 
    > `.env`にuser_name, uid, gidを正確に記入しなければ、コンテナの外部からファイル削除などの操作が難しくなります

7. ユーザー情報を`../docker/diffusers-pytorch-cuda/Dockerfile`に書き込む

    ```
    python add_Dockerfile.py
    ```

    > 上記の方法でビルドできない場合は`../docker/diffusers-pytorch-cuda/Dockerfile`の最後の行に、以下の内容を直接書き込んでください

        ```
        ARG USER_NAME=${USER_NAME}
        ARG USER_UID=${USER_UID}
        ARG USER_GID=${USER_GID}
        WORKDIR /workspace
        RUN groupadd --gid $USER_GID $USER_NAME && \\
            useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME
        USER $USER_NAME
        ```

8. 以下のコマンドでDockerのコンテナをビルドする
    
    ```
    docker compose up -d --build
    ```
    
9. 以下のコマンドでDockerのコンテナ環境に入る

    ```
    docker exec -it diffusers bash
    ```

10. venvで仮想環境を作成する（作成する際は仮想環境名を日付にすることをオススメします）

    ```
    python3 -m venv [yyyy_mm_dd]
    ```

11. 仮想環境をActivateする

    ```
    source [yyyy_mm_dd]/bin/activate
    ```

12. Stable Diffusionの実行に必要なモジュールをインストールする

    ```
    pip install . && pip install wandb xformers bitsandbytes scipy
    ```

## Stable Diffusionの実行について

- このリポジトリではText-to-ImageとDreamBoothの実行方法について解説します。
詳しくは[Text-to-Imageで画像を生成する](text_to_image/README_TtI.md)、[DreamBoothで画像を生成する](dreambooth/README_dreambooth.md)を確認してください

- その他のStable Diffusionのモデルについては、各々実行方法を調べてください

## 参考文献

- [diffusersのGitHubページ](https://github.com/NVlabs/stylegan3/)

- [Hugging Faceの公式ページ](https://github.com/NVlabs/stylegan3/issues/77)
