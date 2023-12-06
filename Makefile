VENV := .venv

PROJECT := src
TESTS := tests
NOTEBOOKS := notebooks

# Prepare

.venv:
	poetry install --no-root
	poetry check

setup: .venv

# Lint

black:
	poetry run black --check --diff $(PROJECT) $(NOTEBOOKS)

flake: .venv
	poetry run flake8 $(PROJECT) $(NOTEBOOKS)

pylint: .venv
	poetry run pylint $(PROJECT) $(NOTEBOOKS)

lint: black flake pylint


# Test

.pytest:
	poetry run pytest $(TESTS)

test: .venv .pytest
