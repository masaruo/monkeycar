# ラズパイOS (Bookworm) とベースを合わせる
FROM python:3.11-slim-bookworm

# Raspberry Pi repository
RUN apt-get update && apt-get install -y wget gnupg && \
    # GPGキーのダウンロードと保存
    wget -qO - https://archive.raspberrypi.org/debian/raspberrypi.gpg.key | \
    gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg && \
    # リポジトリリストへの追加
    echo "deb [signed-by=/usr/share/keyrings/raspberrypi-archive-keyring.gpg] http://archive.raspberrypi.org/debian/ bookworm main" \
    > /etc/apt/sources.list.d/raspi.list && \
    apt-get update

# ---------------------------------------------------------------------
# 2. システム依存パッケージのインストール
# ---------------------------------------------------------------------
# これで python3-libcamera が見つかるようになります
RUN apt-get update && apt-get install -y \
    build-essential \
    make \
    vim \
    gnupg \
    python3-libcamera \
    python3-kms++ \
    libcap-dev \
    libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# システム依存ライブラリのインストール
# libcamera と Pygame (SDL) に必要な最小限のパッケージ
RUN apt-get update && apt-get install -y \
    make \
    vim \
    python3-libcamera \
    libcap-dev \
    libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /monkeycar
COPY cnn ./cnn
COPY car-app ./car-app

WORKDIR /monkeycar/car-app

# COPY car-app/pyproject.toml car-app/uv.lock ./

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
#COPY pyproject.toml .

# システムの libcamera を利用するため --system-site-packages 相当の設定で同期
RUN uv venv --system-site-packages
RUN uv sync --frozen --no-dev


COPY . .

# 実行
CMD ["uv", "run", "python", "main.py"]
