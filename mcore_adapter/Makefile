.DEFAULT_GOAL := help

help:
	@echo "Available commands:"
	@echo "  make clean           - Remove build artifacts and temporary files"
	@echo "  make build           - Build the wheel package"
	@echo "  make build-release   - Clean and build"
	@echo "  make install-build   - Install build tool if not present"

clean:
	@echo "Cleaning up..."
	rm -rf dist/
	rm -rf build/
	rm -rf *.egg-info
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

install-build:
	@echo "Installing build tool..."
	pip install --upgrade build

build:
	@echo "Building wheel and source distribution..."
	python -m build

build-release: clean build

.PHONY: help clean build build-release install-build
