# ==============================================================================
# KernelSmith — development & test container
# Multi-stage: 'dev' for interactive work, 'test' for CI
#
# Usage:
#   docker compose run test       # build + run all tests
#   docker compose run dev        # interactive shell
#   docker compose run lint       # ruff + clang-format checks
# ==============================================================================

# ---------- stage: base (system deps) ----------
FROM ubuntu:22.04 AS base

LABEL org.opencontainers.image.source="https://github.com/pkp124/ulir-kernels"
LABEL org.opencontainers.image.description="KernelSmith MLIR compiler framework"
LABEL org.opencontainers.image.licenses="AGPL-3.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV LLVM_VERSION=18

# Base system packages (no LLVM yet)
RUN apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      ca-certificates wget gnupg lsb-release software-properties-common \
      cmake ninja-build git \
      python3 python3-pip python3-venv \
      libgtest-dev && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Add LLVM 18 repo (signed-by, not deprecated apt-key) then install
RUN wget -qO- https://apt.llvm.org/llvm-snapshot.gpg.key | \
      gpg --batch --dearmor -o /usr/share/keyrings/llvm-archive-keyring.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-${LLVM_VERSION} main" > /etc/apt/sources.list.d/llvm.list && \
    apt-get update -qq && \
    apt-get install -y -qq --no-install-recommends \
      clang-${LLVM_VERSION} lld-${LLVM_VERSION} \
      mlir-${LLVM_VERSION}-tools libmlir-${LLVM_VERSION}-dev \
      llvm-${LLVM_VERSION}-dev \
      clang-format-${LLVM_VERSION} && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ENV LLVM_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/llvm
ENV MLIR_DIR=/usr/lib/llvm-${LLVM_VERSION}/lib/cmake/mlir
ENV CC=clang-${LLVM_VERSION}
ENV CXX=clang++-${LLVM_VERSION}
ENV PATH="/usr/lib/llvm-${LLVM_VERSION}/bin:${PATH}"

# ---------- stage: dev (full toolchain + source) ----------
FROM base AS dev

WORKDIR /workspace

# Python venv + deps (layer caching — deps installed before source copy)
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --quiet --upgrade pip && \
    /opt/venv/bin/pip install --quiet \
      numpy pytest ruff lit filecheck

ENV PATH="/opt/venv/bin:${PATH}"
ENV VIRTUAL_ENV=/opt/venv

# Copy full source
COPY . .

CMD ["/bin/bash"]

# ---------- stage: build ----------
FROM dev AS build

RUN cmake -S . -B build \
      -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_DIR="${LLVM_DIR}" \
      -DMLIR_DIR="${MLIR_DIR}" && \
    cmake --build build --parallel

# ---------- stage: test (default) ----------
FROM build AS test

CMD ["ctest", "--test-dir", "build", "--output-on-failure"]
