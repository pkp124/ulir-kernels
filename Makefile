SHELL := /usr/bin/env bash

.PHONY: setup configure build test lit unit capi lint format-check docker-verify verify clean

setup:
	./scripts/setup.sh

configure:
	cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release

build:
	cmake --build build --parallel

build-debug:
	cmake -S . -B build-debug -G Ninja -DCMAKE_BUILD_TYPE=Debug
	cmake --build build-debug --parallel

test:
	ctest --test-dir build --output-on-failure

lit:
	cmake --build build --target check-kernelsmith-lit

unit:
	ctest --test-dir build --output-on-failure -R kernelsmith-unit

capi:
	ctest --test-dir build --output-on-failure -R 'capi|numpy'

lint:
	ruff check .
	ruff format --check .

format-check:
	@if command -v clang-format-21 >/dev/null 2>&1; then 	  find lib include tools \( -name '*.cpp' -o -name '*.h' \) -print0 | xargs -0 clang-format-21 --dry-run --Werror; 	elif command -v clang-format >/dev/null 2>&1; then 	  find lib include tools \( -name '*.cpp' -o -name '*.h' \) -print0 | xargs -0 clang-format --dry-run --Werror; 	else 	  echo "clang-format not found; run ./scripts/docker-verify.sh for container format checks"; 	  exit 1; 	fi

docker-verify:
	./scripts/docker-verify.sh

verify: lint build test
	@echo "Native verification passed. Run ./scripts/docker-verify.sh before push when Docker is available."

clean:
	rm -rf build
