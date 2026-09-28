# Specifications

This directory contains specification documents that define the behavior and implementation approach for all components.

## Purpose

Specifications serve as:
1. **Design documents**: Define behavior before implementation
2. **Reference**: Source of truth for expected behavior
3. **Test basis**: Define what tests should verify
4. **Documentation**: Help understand the system

## Structure

```text
specs/
├── kernels/
│   ├── matmul.md         ks.matmul. Lowers through the RVV pipeline.
│   ├── conv2d.md         ks.conv2d. Parse and verify only.
│   ├── attention.md      ks.attention. Parse and verify only.
│   └── quantization.md   quantize, dequantize, INT8, and W4A8 dot/GEMV.
└── targets/
    ├── riscv-rvv.md      Primary hardware target.
    └── system-description.md
```

The [README kernel table](../README.md#supported-kernels) is the short status
list. These specs are the contracts. If a spec and the code disagree, fix the
spec in the same change as the code.

## Specification Template

Each specification should include:

1. **Overview**: What the component does
2. **Interface**: Inputs, outputs, attributes
3. **Semantics**: Mathematical or logical definition
4. **Constraints**: Preconditions, postconditions
5. **Implementation Notes**: Lowering strategy, optimizations
6. **Test Cases**: Key scenarios to test

## Workflow

1. **Before implementing**: Check for existing spec or create one
2. **During implementation**: Follow the spec
3. **After implementation**: Update spec if design changed
4. **Reviews**: Spec should match implementation
