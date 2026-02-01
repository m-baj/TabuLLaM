# TabuLLaM justfile

# Default recipe
default:
    @just --list

# Run all tests
test:
    uv run pytest tabullam/tests -v

# Run tests with coverage
test-cov:
    uv run pytest tabullam/tests -v --cov=tabullam --cov-report=term-missing

# Run specific test file
test-file FILE:
    uv run pytest {{FILE}} -v

# Run tests matching pattern
test-match PATTERN:
    uv run pytest tabullam/tests -v -k "{{PATTERN}}"

# Install dependencies
install:
    uv sync

# Run the experiment app
run:
    uv run python -m experiment_app
