.PHONY: help install test test-unit test-integration test-cov test-fast lint format clean

.DEFAULT_GOAL := help

help: ## Display this help message
	@echo "PopGPT v2.0 - Available Make Targets"
	@echo "======================================"
	@echo ""
	@awk 'BEGIN {FS = ": ## "} /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } /^[a-zA-Z_-]+: ## / { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } ' $(MAKEFILE_LIST)

##@ Development

install: ## Install package in development mode
	pip3 install -e .

install-dev: install ## Install package with development dependencies
	pip3 install pytest pytest-cov pytest-timeout pytest-xdist

##@ Testing

test: ## Run fast tests (excludes slow/integration tests)
	python3 -m pytest tests/unit/ tests/integration/ -m "not slow" -v --tb=short --cov=popgpt --cov-report=term-missing

test-all: ## Run all tests including slow and integration tests
	python3 -m pytest tests/ -v --tb=short --cov=popgpt --cov-report=term-missing

test-unit: ## Run unit tests only (fast)
	python3 -m pytest tests/unit/ -v --tb=short

test-integration: ## Run integration tests only
	python3 -m pytest tests/integration/ -v --tb=short -m integration

test-fast: ## Run unit tests excluding slow tests (alias for test-unit -m not slow)
	python3 -m pytest tests/unit/ -v --tb=short -m "not slow" -x

test-cov: ## Run tests with detailed coverage report
	python3 -m pytest tests/ --cov=popgpt --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/index.html"

test-cov-fail: ## Run tests and fail if coverage below 80%
	python3 -m pytest tests/ --cov=popgpt --cov-fail-under=80 --cov-report=term-missing

test-parallel: ## Run tests in parallel
	python3 -m pytest tests/ -n auto -v

test-verbose: ## Run tests with maximum verbosity
	python3 -m pytest tests/ -vv --tb=long

test-debug: ## Run tests with debugger on failure
	python3 -m pytest tests/ -vv --pdb --tb=short

test-last-failed: ## Run only previously failed tests
	python3 -m pytest tests/ --lf -v

##@ Code Quality

lint: ## Run linters (if configured)
	@echo "Running linters..."
	@if command -v ruff > /dev/null; then \
		ruff check src/; \
	else \
		echo "ruff not installed, skipping"; \
	fi
	@if command -v mypy > /dev/null; then \
		mypy src/ --ignore-missing-imports; \
	else \
		echo "mypy not installed, skipping"; \
	fi

format: ## Format code with black
	@if command -v black > /dev/null; then \
		black src/ tests/; \
	else \
		echo "black not installed, skipping"; \
	fi

##@ Cleanup

clean: ## Remove build artifacts and cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned up build artifacts and cache"

clean-data: ## Remove generated data files (careful!)
	@echo "This will remove generated dataset files. Press Ctrl+C to cancel..."
	@sleep 3
	rm -rf data/*/train.bin data/*/val.bin data/*/input.txt

##@ Data Generation

generate-shakespeare: ## Generate Shakespeare dataset
	cd data_generators && python3 shakespeare.py

generate-calculator: ## Generate calculator dataset
	cd data_generators && python3 calculator.py

generate-reverser: ## Generate reverser dataset
	cd data_generators && python3 reverser.py

generate-wordcalc: ## Generate word calculator dataset
	cd data_generators && python3 wordcalc.py

generate-gpt: ## Generate GPT dataset (Shakespeare with GPT-2 tokenizer)
	cd data_generators && python3 gpt.py

generate-all: ## Generate all datasets
generate-all: generate-shakespeare generate-calculator generate-reverser generate-wordcalc generate-gpt

##@ Training

# Detect number of GPUs available (defaults to 1 if no GPUs found)
NGPUS := $(shell python3 -c "import torch; print(torch.cuda.device_count() if torch.cuda.is_available() else 1)")

# Determine if we should use torchrun (only if we have GPUs)
ifeq ($(shell python3 -c "import torch; print('1' if torch.cuda.is_available() and torch.cuda.device_count() > 0 else '0')"),1)
    TRAIN_CMD = MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS)
else
    TRAIN_CMD = python3
endif

# Usage:
#   make train-shakespeare              # Auto-scales to all available GPUs
#   make train-reverser NGPUS=2         # Use specific number of GPUs
#   make train-shakespeare-single       # Force single GPU/CPU
#   make train CONFIG=configs/my.yaml   # Train with custom config (auto-scales)
#
# Examples for multi-GPU training:
#   make train-shakespeare NGPUS=4      # Use 4 GPUs explicitly
#   CUDA_VISIBLE_DEVICES=0,1 make train-shakespeare  # Use GPUs 0 and 1 only

train: ## Train with custom config (use CONFIG=path/to/config.yaml, auto-scales to all GPUs)
ifndef CONFIG
	@echo "Error: CONFIG variable is required. Usage: make train CONFIG=configs/train_model.yaml"
	@exit 1
endif
	@echo "Training with $(NGPUS) GPU(s) using config: $(CONFIG)"
	$(TRAIN_CMD) -m popgpt.cli train --config $(CONFIG)

train-shakespeare: ## Train Shakespeare model with default config (auto-scales to all GPUs)
	@echo "Training with $(NGPUS) GPU(s)..."
	$(TRAIN_CMD) -m popgpt.cli train --config configs/train_shakespeare_char.yaml

train-calculator: ## Train calculator model with default config (auto-scales to all GPUs)
	@echo "Training with $(NGPUS) GPU(s)..."
	$(TRAIN_CMD) -m popgpt.cli train --config configs/train_calculator.yaml

train-reverser: ## Train reverser model with default config (auto-scales to all GPUs)
	@echo "Training with $(NGPUS) GPU(s)..."
	$(TRAIN_CMD) -m popgpt.cli train --config configs/train_reverser.yaml

train-wordcalc: ## Train word calculator model with default config (auto-scales to all GPUs)
	@echo "Training with $(NGPUS) GPU(s)..."
	$(TRAIN_CMD) -m popgpt.cli train --config configs/train_wordcalc.yaml

train-gpt: ## Train GPT model with default config (auto-scales to all GPUs)
	@echo "Training with $(NGPUS) GPU(s)..."
	$(TRAIN_CMD) -m popgpt.cli train --config configs/train_gpt.yaml

# Advanced training targets
train-shakespeare-single: ## Train Shakespeare model on single GPU
	python3 -m popgpt.cli train --config configs/train_shakespeare_char.yaml

train-calculator-single: ## Train calculator model on single GPU
	python3 -m popgpt.cli train --config configs/train_calculator.yaml

train-reverser-single: ## Train reverser model on single GPU
	python3 -m popgpt.cli train --config configs/train_reverser.yaml

train-wordcalc-single: ## Train word calculator model on single GPU
	python3 -m popgpt.cli train --config configs/train_wordcalc.yaml

train-gpt-single: ## Train GPT model on single GPU
	python3 -m popgpt.cli train --config configs/train_gpt.yaml

train-shakespeare-ddp: ## Train Shakespeare model with explicit GPU count (use NGPUS=N)
	MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS) -m popgpt.cli train --config configs/train_shakespeare_char.yaml

train-calculator-ddp: ## Train calculator model with explicit GPU count (use NGPUS=N)
	MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS) -m popgpt.cli train --config configs/train_calculator.yaml

train-reverser-ddp: ## Train reverser model with explicit GPU count (use NGPUS=N)
	MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS) -m popgpt.cli train --config configs/train_reverser.yaml

train-wordcalc-ddp: ## Train word calculator model with explicit GPU count (use NGPUS=N)
	MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS) -m popgpt.cli train --config configs/train_wordcalc.yaml

train-gpt-ddp: ## Train GPT model with explicit GPU count (use NGPUS=N)
	MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 torchrun --standalone --nproc_per_node=$(NGPUS) -m popgpt.cli train --config configs/train_gpt.yaml

##@ Sampling

sample-shakespeare: ## Generate Shakespeare samples
	python3 -m popgpt.cli sample --config configs/sample_shakespeare.yaml

sample-calculator: ## Generate calculator samples
	python3 -m popgpt.cli sample --config configs/sample_calculator.yaml

sample-reverser: ## Generate reverser samples
	python3 -m popgpt.cli sample --config configs/sample_reverser.yaml

sample-wordcalc: ## Generate word calculator samples
	python3 -m popgpt.cli sample --config configs/sample_wordcalc.yaml

sample-gpt: ## Generate GPT samples
	python3 -m popgpt.cli sample --config configs/sample_gpt.yaml

##@ CI/CD

ci: ## Run continuous integration checks
ci: clean test-cov-fail lint
	@echo "All CI checks passed!"

check: ## Quick check before commit
check: clean test-fast lint
	@echo "Quick checks passed!"
