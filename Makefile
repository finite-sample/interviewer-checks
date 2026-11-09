.PHONY: install lint format test check ci-docker build clean

PYTHON_IMAGE ?= python:3.13

install:
	uv sync

lint:
	uv run black --check .
	uv run isort --check-only .
	uv run flake8 src tests examples

format:
	uv run isort .
	uv run black .

test:
	uv run pytest

check: lint test

# The project mounts read-only and is copied inside, so the container never writes a .venv into the checkout.
ci-docker:
	docker run --rm -v "$(CURDIR)":/src:ro $(PYTHON_IMAGE) sh -c \
		"cp -r /src /work && cd /work && rm -rf .venv && pip install --quiet uv && make check"

build:
	uv build

clean:
	rm -rf build dist .pytest_cache
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
