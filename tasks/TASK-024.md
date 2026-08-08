# TASK-024: Validate Generated Transformer Helper on RVV QEMU

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General or `.cursor/agents/mlir-pass-agent.md`.

## Description

Close the remaining `TASK-008` simulation criterion by compiling a transformer
helper from the KS dialect through the complete RVV pipeline, linking the
generated object into a self-checking RISC-V executable, and running it under
QEMU at each supported VLEN.

Use `ks.rms_norm` so this task depends only on the compiler lowering already
merged in `TASK-022`.

## Acceptance Criteria

- [ ] Add a self-checking `ks.rms_norm` MLIR program with non-trivial expected
      outputs.
- [ ] Compile it through `scripts/compile-rvv.sh` and link a static RISC-V ELF.
- [ ] Confirm the generated ELF contains RVV instructions rather than only
      scalar code.
- [ ] Run the generated helper under QEMU at VLEN 256 and 512 in the RVV CI
      workflow.
- [ ] Reproduce the generated-helper QEMU test locally.
- [ ] Run full CTest and Python lint/format before completion.

## Dependencies

- `TASK-008`
- `TASK-022`
- `docs/design/DES-009-m4-rvv-lowering.md`
- `specs/targets/riscv-rvv.md`

## Verification

```bash
MLIR_TRANSLATE=mlir-translate-21 LLC=llc-21 \
  ./scripts/compile-rvv.sh tests/riscv/generated_rms_norm.mlir \
  /tmp/generated_rms_norm.o
riscv64-linux-gnu-gcc -static /tmp/generated_rms_norm.o \
  -o /tmp/generated_rms_norm
riscv64-linux-gnu-objdump -d /tmp/generated_rms_norm
python tests/riscv_runner.py --binary /tmp/generated_rms_norm \
  --vlens 256 512
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

## Notes

- This validates generated compiler output. Descriptor-backed scalar C helper
  tests already cover RMSNorm and softmax under QEMU separately.

## Log

### 2026-08-08
- Created from the final open acceptance criterion in `TASK-008`.
