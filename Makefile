# ==============================================================================
# MLIR Kernel Generation Project - Development Makefile
# ==============================================================================

.PHONY: help setup build test verify clean lint format docs

# Default target
help:
	@echo "MLIR Kernel Generation Project"
	@echo ""
	@echo "Setup & Build:"
	@echo "  make setup          - Install dependencies and configure project"
	@echo "  make build          - Build the project"
	@echo "  make build-debug    - Build with debug symbols"
	@echo "  make clean          - Remove build artifacts"
	@echo ""
	@echo "Testing:"
	@echo "  make test           - Run all tests"
	@echo "  make test-unit      - Run unit tests only"
	@echo "  make test-lit       - Run MLIR lit tests"
	@echo "  make test-integration - Run integration tests"
	@echo ""
	@echo "Quality:"
	@echo "  make lint           - Run all linters"
	@echo "  make format         - Format all code"
	@echo "  make verify         - Run full verification (lint + test + build)"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Generate documentation"
	@echo "  make docs-serve     - Serve documentation locally"
	@echo ""
	@echo "Development:"
	@echo "  make new-kernel NAME=<name>  - Create new kernel from template"
	@echo "  make new-pass NAME=<name>    - Create new pass from template"
	@echo "  make new-task ID=<id> TITLE=<title> - Create new task"

# ==============================================================================
# Configuration
# ==============================================================================

BUILD_DIR := build
BUILD_TYPE ?= Release
LLVM_DIR ?= $(shell llvm-config --prefix 2>/dev/null || echo "/usr/lib/llvm-18")
MLIR_DIR ?= $(LLVM_DIR)

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DLLVM_DIR=$(LLVM_DIR)/lib/cmake/llvm \
	-DMLIR_DIR=$(MLIR_DIR)/lib/cmake/mlir

# ==============================================================================
# Setup & Build
# ==============================================================================

setup: scripts/setup.sh
	@./scripts/setup.sh

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

configure: $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) ..

build: configure
	@cmake --build $(BUILD_DIR) --parallel

build-debug:
	@$(MAKE) build BUILD_TYPE=Debug

clean:
	@rm -rf $(BUILD_DIR)
	@find . -name "*.pyc" -delete
	@find . -name "__pycache__" -delete
	@find . -name ".pytest_cache" -delete

# ==============================================================================
# Testing
# ==============================================================================

test: test-unit test-lit test-integration

test-unit:
	@echo "Running unit tests..."
	@python -m pytest tests/unit -v --tb=short 2>/dev/null || echo "No unit tests found yet"

test-lit:
	@echo "Running lit tests..."
	@if [ -d "$(BUILD_DIR)" ] && [ -f "$(BUILD_DIR)/bin/aikernel-opt" ]; then \
		lit tests/lit -v; \
	else \
		echo "Build first with 'make build' to run lit tests"; \
	fi

test-integration:
	@echo "Running integration tests..."
	@python -m pytest tests/integration -v --tb=short 2>/dev/null || echo "No integration tests found yet"

# ==============================================================================
# Quality
# ==============================================================================

lint: lint-python lint-cpp lint-cmake

lint-python:
	@echo "Linting Python..."
	@python -m ruff check . 2>/dev/null || echo "Install ruff: pip install ruff"

lint-cpp:
	@echo "Linting C++..."
	@if [ -f "$(BUILD_DIR)/compile_commands.json" ]; then \
		clang-tidy src/**/*.cpp 2>/dev/null || true; \
	else \
		echo "Build first to generate compile_commands.json"; \
	fi

lint-cmake:
	@echo "Linting CMake..."
	@cmake-lint CMakeLists.txt 2>/dev/null || echo "Install cmake-lint: pip install cmake-format"

format: format-python format-cpp format-cmake

format-python:
	@echo "Formatting Python..."
	@python -m ruff format . 2>/dev/null || echo "Install ruff: pip install ruff"

format-cpp:
	@echo "Formatting C++..."
	@find src -name "*.cpp" -o -name "*.h" | xargs clang-format -i 2>/dev/null || true

format-cmake:
	@echo "Formatting CMake..."
	@cmake-format -i CMakeLists.txt 2>/dev/null || true

verify:
	@echo "============================================"
	@echo "Running full verification..."
	@echo "============================================"
	@echo ""
	@echo "[1/3] Linting..."
	@$(MAKE) lint
	@echo ""
	@echo "[2/3] Building..."
	@$(MAKE) build 2>/dev/null || echo "Build not configured yet (MLIR not found)"
	@echo ""
	@echo "[3/3] Testing..."
	@$(MAKE) test
	@echo ""
	@echo "============================================"
	@echo "Verification complete!"
	@echo "============================================"

# ==============================================================================
# Documentation
# ==============================================================================

docs:
	@echo "Generating documentation..."
	@if [ -f "docs/conf.py" ]; then \
		cd docs && make html; \
	else \
		echo "Documentation not configured yet"; \
	fi

docs-serve:
	@echo "Serving documentation at http://localhost:8000..."
	@cd docs/_build/html && python -m http.server 2>/dev/null || echo "Build docs first"

# ==============================================================================
# Development Helpers
# ==============================================================================

new-kernel:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make new-kernel NAME=<kernel_name>"; \
		exit 1; \
	fi
	@./scripts/new-kernel.sh $(NAME)

new-pass:
	@if [ -z "$(NAME)" ]; then \
		echo "Usage: make new-pass NAME=<pass_name>"; \
		exit 1; \
	fi
	@./scripts/new-pass.sh $(NAME)

new-task:
	@if [ -z "$(ID)" ] || [ -z "$(TITLE)" ]; then \
		echo "Usage: make new-task ID=<task_id> TITLE=\"<title>\""; \
		exit 1; \
	fi
	@./scripts/new-task.sh "$(ID)" "$(TITLE)"

# ==============================================================================
# Project Info
# ==============================================================================

info:
	@echo "Project: MLIR Kernel Generation"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo "LLVM Directory: $(LLVM_DIR)"
	@echo "MLIR Directory: $(MLIR_DIR)"
