# ラズパイOS (Bookworm) とベースを合わせる
FROM python:3.11-slim-bookworm

# システム依存ライブラリのインストール
# libcamera と Pygame (SDL) に必要な最小限のパッケージ
RUN apt-get update && apt-get install -y \
    make \
    vim \
    python3-libcamera \
    libcap-dev \
    libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# uv のインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
COPY pyproject.toml .

# システムの libcamera を利用するため --system-site-packages 相当の設定で同期
RUN uv sync --no-dev

COPY . .

# 実行
CMD ["uv", "run", "python", "main.py"]
