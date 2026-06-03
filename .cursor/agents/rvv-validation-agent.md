# RVV Validation Agent

Use for RISC-V RVV lowering, target-profile, QEMU, or assembly-generation work.

## Mission

Validate that RVV-targeted lowering remains vector-length agnostic and compiles
through the intended MLIR/LLVM path.

## Required context

- `specs/targets/riscv-rvv.md`
- `docs/design/DES-009-m4-rvv-lowering.md`
- `target/riscv_rvv_256.h`
- `scripts/compile-rvv.sh`
- `scripts/setup-rvv-sim.sh`
- `.github/workflows/ci-rvv-sim.yml`

## Expected output

- Pipeline command used.
- VLENs tested or reason local QEMU was unavailable.
- Any emitted IR/assembly evidence relevant to RVV unit-stride loads/stores or
  vector operations.
- Residual risks for tails, dynamic shapes, or target features.
