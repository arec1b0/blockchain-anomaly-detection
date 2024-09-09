# Makefile for Blockchain Anomaly Detection

# Install all the dependencies specified in requirements.txt
install:
	pip install -r requirements.txt

# Run the application (main.py)
run:
	python src/main.py

# Run tests using pytest
test:
	pytest

# Check code style and lint using flake8
lint:
	flake8 src tests

# Run the full pipeline: fetch data, process, detect anomalies, and visualize
pipeline:
	python examples/example_usage.py

# Clean up cache and temporary files
clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -rf {} +
	rm -rf .pytest_cache
	rm -rf logs/app.log

# Build Docker image
docker-build:
	docker build -t blockchain-anomaly-detection .

# Run Docker container
docker-run:
	docker run -d -p 80:80 blockchain-anomaly-detection
