# ホストPC用 Makefile（uv）

.PHONY: help sync dev train convert visualize clean all

all: run

run:
	uv run python main.py

sync:
	uv sync
	@echo "✓ 依存をインストール完了"

clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache *.pyc .uv-cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ クリーンアップ完了"

fclean: clean
	rm -rf .venv uv.lock

clean-data:
	rm -rf ./data ./output ./model

pull:
	./sync.sh pull

push:
	./sync.sh push
