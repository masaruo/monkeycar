.PHONY: all run autopilot record install clean calibrate shell build down fclean lint

# Service name defined in docker-compose.yml
SERVICE_NAME = monkeycar

all: run

# ----------------------------------------------------------------
# Docker Execution Commands
# ----------------------------------------------------------------

# 'make run' -> Starts the container normally (detached or attached)
# Uses 'docker compose up' so it respects the 'restart' policy and logs.
run:
	docker compose up

# 'make autopilot' -> Spins up a temporary container just for this task
# --rm means "remove the container after it finishes" so you don't pile up dead containers.
autopilot:
	docker compose run --rm $(SERVICE_NAME) uv run python main.py --autopilot

# 'make record' -> Spins up a temporary container for recording
record:
	docker compose run --rm $(SERVICE_NAME) uv run python main.py --record

# 'make calibrate' -> Runs the motor calibration script inside Docker
calibrate:
	docker compose run --rm $(SERVICE_NAME) uv run python motor.py

# 'make shell' -> (NEW) Very useful! Gives you a terminal INSIDE the container
# Use this to check if files are there, or to run 'ls -l /dev/video0' inside Docker.
shell:
	docker compose run --rm --entrypoint bash $(SERVICE_NAME)

# ----------------------------------------------------------------
# Management Commands
# ----------------------------------------------------------------

# 'make install' or 'make build' -> Builds the Docker image
install: build
build:
	docker compose build

# 'make down' -> Stops and removes the main container
down:
	docker compose down

# 'make clean' -> Cleans up pycache (requires sudo if Docker created them as root)
clean:
	find . -type d -name __pycache__ -exec sudo rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf .mypy_cache .ruff_cache

# 'make fclean' -> Deep clean (prunes Docker system)
fclean: clean
	docker system prune -f

# ----------------------------------------------------------------
# Linting (Runs inside Docker to ensure environment consistency)
# ----------------------------------------------------------------

lint:
	docker compose run --rm $(SERVICE_NAME) uv run ruff check .
	docker compose run --rm $(SERVICE_NAME) uv run mypy . --warn-return-any --ignore-missing-imports
