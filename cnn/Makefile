.PHONY: all install clean fclean lint lint-strict

all: install

install:
	uv sync

clean:
	rm -rf .pytest_cache .ruff_cache *.pyc .uv-cache .mypy_cache
	find . -type d -name "__pycache__" -exec rm -rf {} +

fclean: clean
	rm -rf .venv uv.lock

lint:
	-uv run ruff check .
	uv run mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	-uv run ruff check .
	uv run mypy . --strict
