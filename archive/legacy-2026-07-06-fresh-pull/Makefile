.PHONY: install demo lint test

install:
	pip install -e ".[ml,dev]"

demo:
	python scripts/demo_pose.py

demo-video:
	@echo "Usage: make demo-video SRC=path/to/video.mp4"
	python scripts/demo_pose.py $(SRC)

lint:
	ruff check src/ scripts/
	ruff format --check src/ scripts/

test:
	pytest tests/ -v
