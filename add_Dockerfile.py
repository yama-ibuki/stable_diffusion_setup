# ユーザー情報を定義
user_information = """
ARG USER_NAME=${USER_NAME}
ARG USER_UID=${USER_UID}
ARG USER_GID=${USER_GID}
WORKDIR /workspace
RUN groupadd --gid $USER_GID $USER_NAME && \\
    useradd --uid $USER_UID --gid $USER_GID -m $USER_NAME
USER $USER_NAME
"""

# Dockerfileを読み込む
with open('../docker/diffusers-pytorch-cuda/Dockerfile', 'r') as file:
    original_content = file.read()

# 新しい内容を追加
full_content = original_content + user_information

# Dockerfileに新しい内容を保存
with open('../docker/diffusers-pytorch-cuda/Dockerfile', 'w') as file:
    file.write(full_content)