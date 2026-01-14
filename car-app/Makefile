.PHONY: setup run record install clean fclean all

all: run

# # プロジェクトの初期化（libcameraへのアクセスを許可する仮想環境を作成）
# setup:
# 	rm -rf .venv
# 	uv venv --system-site-packages
# 	uv add tensorflow opencv-python pillow numpy pygame

# 通常実行（自律走行を想定）
run:
	uv run python main.py

# データ記録モード（引数や環境変数を渡す場合に拡張可能）
record:
	uv run python main.py

# 依存関係の同期
install:
	uv venv --system-site-packages
	uv sync

# キャッシュの削除
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# 仮想環境を含む完全な削除
fclean: clean
	rm -rf .venv
