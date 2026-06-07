# TASK-011: Add Host Reference Execution Comparator

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`general`

## Description

Add the host/x86 execution leg for golden-reference verification. Before RISC-V
simulation runs, KernelSmith should prove the descriptor inputs, ABI, and host
reference/generic implementation produce outputs that match the NumPy golden
reference.

## Acceptance Criteria

- [ ] Add a host execution path for at least f32 ReLU and f32 matmul cases.
- [ ] Load descriptor-defined golden inputs and pass them through the host
      reference or generated x86 path.
- [ ] Capture host outputs in the shared artifact format.
- [ ] Compare host outputs against NumPy golden outputs with the descriptor's
      compare policy.
- [ ] Emit machine-parseable result lines and detailed mismatch diagnostics.
- [ ] Integrate the host golden checks into CTest or a script invoked by CI.

## Dependencies

- `TASK-010`
- `DES-013-golden-reference-verification-infrastructure`
- Existing C reference kernels under `lib/kernelsmith/`

## Verification

```bash
python tests/verify.py --target host_reference --case <case>
ctest --test-dir build --output-on-failure
```

## Notes

- Host comparison must pass before RISC-V simulation is considered meaningful
  for the same case.
- Host output can later serve as a secondary comparator for RISC-V when NumPy
  reference generation is expensive or framework-dependent.

## Log

### 2026-06-07
- Created as the host/x86 leg of the DES-013 verification design.
