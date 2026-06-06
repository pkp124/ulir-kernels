# Specifications

This directory contains specification documents that define the behavior and implementation approach for all components.

## Purpose

Specifications serve as:
1. **Design documents**: Define behavior before implementation
2. **Reference**: Source of truth for expected behavior
3. **Test basis**: Define what tests should verify
4. **Documentation**: Help understand the system

## Structure

```
specs/
├── kernels/          # Kernel operation specifications
│   ├── matmul.md
│   ├── conv2d.md
│   ├── attention.md
│   ├── quantization.md
│   └── ...
├── targets/          # Target architecture specifications
│   ├── riscv-rvv.md
│   └── ...
└── README.md         # This file
```

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
