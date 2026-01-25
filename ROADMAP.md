# KernelSmith Roadmap

Milestones are **quantifiable** and **verifiable**. Each milestone produces a working, testable deliverable.

---

## Milestone Philosophy

- **End-to-end first**: Get something fully working before expanding
- **Verifiable**: Each milestone has concrete pass/fail criteria
- **Measurable**: Specific numbers, not vague goals
- **Testable**: Can be verified with automated tests or demos

---

## M1: First Kernel End-to-End

### Goal
**One kernel (matmul) running on RISC-V RVV simulator with multiple variants.**

### Deliverables

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | `ks.matmul` operation fully implemented | Lit tests pass |
| 2 | Lowering pipeline: ks → linalg → vector → LLVM | Pipeline test passes |
| 3 | RISC-V RVV code generation | Assembly output valid |
| 4 | **Runs on QEMU or Spike with RVV** | Correct output verified |
| 5 | At least 3 matmul variants (sizes, types) | All variants run correctly |

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Matrix sizes supported | 3+ sizes (e.g., 64x64, 128x128, 256x256) | Test each size on simulator |
| Data types | At least f32 | Type-specific tests |
| Correctness | 100% bit-exact vs reference | Compare against NumPy |
| Simulator | Runs on QEMU or Spike | CI job passes |
| VLEN compatibility | Works with VLEN=128, 256, 512 | Test each VLEN |

### Verification Script

```bash
# M1 Verification
#!/bin/bash
set -e

echo "=== M1: First Kernel End-to-End ==="

# 1. Build
cmake --build build

# 2. Compile matmul kernel to RVV
./build/bin/ks-opt tests/e2e/matmul_64x64.mlir \
  --ks-lower-to-linalg \
  --ks-tile \
  --ks-vectorize \
  --ks-lower-to-rvv \
  --convert-to-llvm \
  -o /tmp/matmul.llvm.mlir

# 3. Generate assembly
mlir-translate --mlir-to-llvmir /tmp/matmul.llvm.mlir | \
  llc -march=riscv64 -mattr=+v -o /tmp/matmul.s

# 4. Build executable
riscv64-linux-gnu-gcc /tmp/matmul.s tests/e2e/matmul_main.c \
  -o /tmp/matmul_test

# 5. Run on QEMU with RVV
qemu-riscv64 -cpu rv64,v=true,vlen=256 /tmp/matmul_test

# 6. Verify output
echo "Comparing output to reference..."
diff /tmp/output.bin tests/e2e/matmul_64x64_expected.bin

echo "=== M1 PASSED ==="
```

### Success = All True

- [ ] `ks.matmul` parses, prints, verifies correctly
- [ ] Lowering pipeline produces valid LLVM IR
- [ ] Generated RISC-V assembly is valid
- [ ] Executable runs on QEMU with `rv64,v=true`
- [ ] Output matches reference for 64x64 f32 matmul
- [ ] Output matches reference for 128x128 f32 matmul  
- [ ] Output matches reference for 256x256 f32 matmul
- [ ] Works with VLEN=128
- [ ] Works with VLEN=256
- [ ] Works with VLEN=512

**M1 Complete when: 10/10 criteria pass**

---

## M2: Core Kernel Library

### Goal
**5 kernels running on RVV simulator with test suite.**

### Deliverables

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | ks.matmul (from M1) | Tests pass |
| 2 | ks.relu | Tests pass on simulator |
| 3 | ks.gelu | Tests pass on simulator |
| 4 | ks.softmax | Tests pass on simulator |
| 5 | ks.layer_norm | Tests pass on simulator |
| 6 | Automated test suite | CI runs all kernels on QEMU |

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Kernels implemented | 5 | Each runs on simulator |
| Test coverage | 100% of kernels | CI report |
| Shape variants per kernel | 3+ | Test matrix |
| CI pipeline | All tests green | GitHub Actions |
| Documentation | All kernels documented | Docs exist |

### Success = All True

- [ ] ks.matmul: 3 sizes pass on QEMU
- [ ] ks.relu: 3 sizes pass on QEMU
- [ ] ks.gelu: 3 sizes pass on QEMU
- [ ] ks.softmax: 3 sizes pass on QEMU
- [ ] ks.layer_norm: 3 sizes pass on QEMU
- [ ] CI runs full test suite on every commit
- [ ] All 15+ test cases pass (5 kernels × 3 sizes)

**M2 Complete when: 7/7 criteria pass**

---

## M3: Attention & Transformers

### Goal
**Complete attention mechanism running, sufficient for transformer inference.**

### Deliverables

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | ks.attention (scaled dot-product) | Tests pass on simulator |
| 2 | ks.batch_matmul | Tests pass |
| 3 | ks.rms_norm | Tests pass |
| 4 | Multi-head attention composite | End-to-end test |
| 5 | Transformer block demo | Runs on QEMU |

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Attention sizes | batch=1,4,8; seq=64,128,256; dim=64 | Test matrix |
| Memory efficiency | O(n) not O(n²) for long sequences | Profile |
| Correctness | Match PyTorch reference | Numerical comparison |

### Success = All True

- [ ] ks.attention runs for seq_len=256 on QEMU
- [ ] ks.batch_matmul runs correctly
- [ ] ks.rms_norm runs correctly
- [ ] Transformer block (attn + ffn + norm) runs end-to-end
- [ ] Output matches PyTorch reference within 1e-5

**M3 Complete when: 5/5 criteria pass**

---

## M4: Performance Optimization

### Goal
**Kernels achieve 50%+ of theoretical peak on RVV.**

### Deliverables

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | Performance benchmarking harness | Reports GFLOPS |
| 2 | Optimized matmul | Benchmark results |
| 3 | Tiling auto-tuner | Best tile size found |
| 4 | Performance regression tests | CI tracks perf |

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| matmul GFLOPS | 50%+ of peak | Benchmark on real/sim HW |
| Performance regression | <5% regression allowed | CI comparison |
| Tile size optimization | Automated selection | Tuner runs |

### Success = All True

- [ ] Benchmark harness reports accurate GFLOPS
- [ ] matmul 256x256 achieves 50%+ theoretical peak
- [ ] Performance tracked in CI
- [ ] No >5% regression in any commit

**M4 Complete when: 4/4 criteria pass**

---

## M5: Production Ready

### Goal
**Usable library with documentation, packaging, and real hardware validation.**

### Deliverables

| # | Deliverable | Verification |
|---|-------------|--------------|
| 1 | Python bindings | Python tests pass |
| 2 | Package/install system | pip install works |
| 3 | User documentation | Docs build, reviewed |
| 4 | Real hardware test | Runs on actual RVV hardware |
| 5 | Example applications | 2+ demos work |

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Python API coverage | All kernels accessible | API tests |
| Documentation pages | 10+ pages | Doc count |
| Real hardware | 1+ actual RVV board | Hardware test |
| Examples | 2+ working demos | Demo scripts run |

---

## M6: Multi-Architecture (Future)

### Goal
**Support 2+ architectures (RVV + one other).**

### Potential Targets
- ARM SVE/SVE2
- x86 AVX-512
- GPU (CUDA/ROCm)

### Quantifiable Criteria

| Criterion | Target | How to Verify |
|-----------|--------|---------------|
| Architectures | 2+ | Tests pass on each |
| Kernel coverage | Same kernels on all targets | Test matrix |
| Performance | 50%+ peak on each target | Benchmarks |

---

## Current Focus

### Active Milestone: M1

**Objective:** One kernel (matmul) running on RISC-V RVV simulator

**Why M1 First:**
- Proves the entire pipeline works
- Catches integration issues early
- Provides foundation for all other kernels

### M1 Task Breakdown

| Task | Description | Status |
|------|-------------|--------|
| M1.1 | ks.matmul operation (parse, verify) | ⬜ |
| M1.2 | Lower to linalg pass | ⬜ |
| M1.3 | Tiling pass | ⬜ |
| M1.4 | Vectorization pass | ⬜ |
| M1.5 | RVV lowering pass | ⬜ |
| M1.6 | QEMU test harness | ⬜ |
| M1.7 | 64x64 matmul verified | ⬜ |
| M1.8 | 128x128 matmul verified | ⬜ |
| M1.9 | 256x256 matmul verified | ⬜ |
| M1.10 | Multi-VLEN testing | ⬜ |

**M1 Progress: 0/10 tasks**

---

## Milestone Timeline Visualization

```
     M1                M2              M3              M4           M5
     │                 │               │               │            │
     ▼                 ▼               ▼               ▼            ▼
┌─────────┐      ┌──────────┐    ┌──────────┐    ┌─────────┐   ┌─────────┐
│ matmul  │      │ 5 kernels│    │attention │    │ 50%     │   │ Python  │
│ on QEMU │  →  │ + tests  │ → │ + xformer│ → │ peak    │ → │ + docs  │
│ 3 sizes │      │ CI green │    │ e2e demo │    │ perf    │   │ + pkg   │
└─────────┘      └──────────┘    └──────────┘    └─────────┘   └─────────┘
     │                 │               │               │            │
   10 ✓              7 ✓            5 ✓            4 ✓          5 ✓
  criteria          criteria       criteria       criteria     criteria
```

---

## Verification Commands

### Quick Milestone Status Check

```bash
# Run M1 verification
./scripts/verify-m1.sh

# Check current milestone progress
cat ROADMAP.md | grep -A 20 "M1 Progress"
```

### Full Verification (when milestone claimed complete)

```bash
# Comprehensive milestone verification
./scripts/verify-milestone.sh M1
```
