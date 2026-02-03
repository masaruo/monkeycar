.PHONY: all, run, autopilot, record, install, clean, calibrate, clean-data, fclean, lint, lint-strict

all: run

run:
	uv run python main.py

autopilot:
	uv run python main.py --autopilot

record:
	uv run python main.py --record

install:
	uv venv --system-site-packages
	uv sync

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# 仮想環境を含む完全な削除
fclean: clean
	rm -rf .venv

clean-data:
	rm -rf ./data ./output

calibrate:
	uv run python motor.py

lint:
	-uv run ruff check .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-uv run ruff check .
	uv run mypy . --strict
