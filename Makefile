.PHONY: install download parquet res lint fmt typecheck check

COMP := nfl-big-data-bowl-2027

install:
	uv sync

download:
	uv run kaggle competitions download $(COMP) -p data/raw
	cd data/raw && unzip -oq $(COMP).zip && rm $(COMP).zip

parquet:
	uv run python -m bowl.convert

res:
	uv run python -m bowl.res

lint:
	uv run ruff check .

fmt:
	uv run ruff format .
	uv run ruff check --fix .

typecheck:
	uv run pyrefly check

check: lint typecheck
