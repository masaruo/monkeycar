# ホストPC用 Makefile（uv）

.PHONY: all run install clean fclean pull push lint lint-strict

all: run

run:
	uv run python main.py

install:
	uv sync

clean:
	rm -rf .pytest_cache .ruff_cache *.pyc .uv-cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +
	@echo "✓ クリーンアップ完了"

fclean: clean
	rm -rf .venv uv.lock

clean-data:
	rm -rf ./data ./output ./model

pull:
	./sync.sh pull

push:
	./sync.sh push

lint:
	-uv run ruff check .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-uv run ruff check .
	uv run mypy . --strict
	
