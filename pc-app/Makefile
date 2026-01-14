# ホストPC用 Makefile（uv）

.PHONY: help sync dev train convert visualize clean

help:
	@echo "利用可能なコマンド:"
	@echo "  make sync       - 依存をインストール（uvを使用）"
	@echo "  make dev        - 開発用パッケージをインストール"
	@echo "  make train      - モデル学習を実行"
	@echo "  make convert    - TFLite変換を実行"
	@echo "  make visualize  - データ可視化を実行"
	@echo "  make clean      - キャッシュとビルド生成物をクリア"

sync:
	uv sync
	@echo "✓ 依存をインストール完了"

dev: sync
	uv sync --extra dev
	@echo "✓ 開発用パッケージをインストール完了"

train: sync
	uv run loader.py
	uv run model_train.py

convert: sync
	uv run model_convert.py

visualize: sync
	uv run visualize.py

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.pyc .uv-cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ クリーンアップ完了"

fclean: clean
	rm -rf .venv uv.lock

delete:
	rm -rf ./data

pull:
	./sync.sh pull

push:
	./sync.sh push
