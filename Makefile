# ==============================================================================
# KernelSmith - Development Makefile
# ==============================================================================

.PHONY: help setup build test verify clean lint format docs

# Default target
help:
	@echo "KernelSmith - Craft Optimized AI Kernels"
	@echo ""
	@echo "Setup & Build:"
	@echo "  make setup          - Install dependencies and configure"
	@echo "  make configure      - Configure CMake build"
	@echo "  make build          - Build the project"
	@echo "  make build-debug    - Build with debug symbols"
	@echo "  make clean          - Remove build artifacts"
	@echo ""
	@echo "Testing (CTest):"
	@echo "  make test           - Run all tests via CTest"
	@echo "  make test-lit       - Run MLIR lit tests"
	@echo "  make test-unit      - Run unit tests"
	@echo "  make test-verbose   - Run tests with verbose output"
	@echo ""
	@echo "Quality:"
	@echo "  make lint           - Run all linters"
	@echo "  make format         - Format all code"
	@echo "  make verify         - Full verification (lint + test)"
	@echo ""
	@echo "Development:"
	@echo "  make new-kernel NAME=<name>  - Create kernel from template"
	@echo "  make new-pass NAME=<name>    - Create pass from template"
	@echo "  make new-design ID=<id> TITLE=<title> - Create design doc"
	@echo ""
	@echo "Documentation:"
	@echo "  make docs           - Generate documentation"

# ==============================================================================
# Configuration
# ==============================================================================

BUILD_DIR := build
BUILD_TYPE ?= Release
MLIR_DIR ?= $(shell llvm-config --prefix 2>/dev/null)/lib/cmake/mlir

CMAKE_FLAGS := \
	-DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
	-DMLIR_DIR=$(MLIR_DIR)

# ==============================================================================
# Setup & Build
# ==============================================================================

setup:
	@./scripts/setup.sh

$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

configure: $(BUILD_DIR)
	@cd $(BUILD_DIR) && cmake $(CMAKE_FLAGS) ..
	@ln -sf $(BUILD_DIR)/compile_commands.json compile_commands.json

build: configure
	@cmake --build $(BUILD_DIR) --parallel

build-debug:
	@$(MAKE) build BUILD_TYPE=Debug

clean:
	@rm -rf $(BUILD_DIR)
	@rm -f compile_commands.json

# ==============================================================================
# Testing (CTest)
# ==============================================================================

test: build
	@cd $(BUILD_DIR) && ctest --output-on-failure

test-verbose: build
	@cd $(BUILD_DIR) && ctest --verbose

test-lit: build
	@cd $(BUILD_DIR) && ctest -R "lit" --output-on-failure

test-unit: build
	@cd $(BUILD_DIR) && ctest -R "unit" --output-on-failure

# ==============================================================================
# Quality
# ==============================================================================

lint: lint-python lint-cpp

lint-python:
	@echo "Linting Python..."
	@python -m ruff check . 2>/dev/null || echo "Install ruff: pip install ruff"

lint-cpp:
	@echo "Linting C++..."
	@if [ -f "$(BUILD_DIR)/compile_commands.json" ]; then \
		find lib include -name "*.cpp" -o -name "*.h" | head -20 | \
		xargs clang-tidy -p $(BUILD_DIR) 2>/dev/null || true; \
	fi

format: format-python format-cpp

format-python:
	@python -m ruff format . 2>/dev/null || true

format-cpp:
	@find lib include -name "*.cpp" -o -name "*.h" | \
		xargs clang-format -i 2>/dev/null || true

verify: lint test
	@echo ""
	@echo "============================================"
	@echo "Verification complete!"
	@echo "============================================"

# ==============================================================================
# Development Helpers
# ==============================================================================

new-kernel:
	@if [ -z "$(NAME)" ]; then echo "Usage: make new-kernel NAME=<name>"; exit 1; fi
	@./scripts/new-kernel.sh $(NAME)

new-pass:
	@if [ -z "$(NAME)" ]; then echo "Usage: make new-pass NAME=<name>"; exit 1; fi
	@./scripts/new-pass.sh $(NAME)

new-design:
	@if [ -z "$(ID)" ] || [ -z "$(TITLE)" ]; then \
		echo "Usage: make new-design ID=<id> TITLE=\"<title>\""; exit 1; fi
	@./scripts/new-design.sh "$(ID)" "$(TITLE)"

new-task:
	@if [ -z "$(ID)" ] || [ -z "$(TITLE)" ]; then \
		echo "Usage: make new-task ID=<id> TITLE=\"<title>\""; exit 1; fi
	@./scripts/new-task.sh "$(ID)" "$(TITLE)"

# ==============================================================================
# Documentation
# ==============================================================================

docs:
	@echo "Generating documentation..."
	@cd docs && mkdocs build 2>/dev/null || echo "Install mkdocs: pip install mkdocs"

# ==============================================================================
# Info
# ==============================================================================

info:
	@echo "Project: KernelSmith"
	@echo "Build Directory: $(BUILD_DIR)"
	@echo "Build Type: $(BUILD_TYPE)"
	@echo "MLIR Directory: $(MLIR_DIR)"
