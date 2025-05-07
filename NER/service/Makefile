.PHONY: run test build docs clean

run:
	uvicorn src.app:app --reload

test:
	python -m pytest tests/

build:
	python setup.py sdist bdist_wheel

docs:
	cp docs/api.md ./README.md

clean:
	rm -rf *.egg-info
	rm -rf dist/
	rm -rf __pycache__