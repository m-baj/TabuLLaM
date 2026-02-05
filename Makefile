.PHONY: help test test-cov test-file test-match install run

help:
	@echo "Available targets:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage"
	@echo "  make test-file     - Run specific test file (usage: make test-file FILE=path/to/test.py)"
	@echo "  make test-match    - Run tests matching pattern (usage: make test-match PATTERN=pattern)"
	@echo "  make install       - Install dependencies"
	@echo "  make run           - Run the experiment app"

test:
	uv run pytest tabullam/tests -v

test-cov:
	uv run pytest tabullam/tests -v --cov=tabullam --cov-report=term-missing

test-file:
	uv run pytest $(FILE) -v

test-match:
	uv run pytest tabullam/tests -v -k "$(PATTERN)"

install:
	uv sync

run:
	uv run python -m experiment_app
