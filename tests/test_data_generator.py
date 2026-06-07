#!/usr/bin/env python3
"""
Test Data Generator for KernelSmith
Generates test tensors for functional validation of kernels
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class TensorConfig:
    """Configuration for generating test tensors"""

    shape: tuple[int, ...]
    dtype: np.dtype
    seed: int = 42
    range_min: float = -1.0
    range_max: float = 1.0
    description: str = ""


class TestDataGenerator:
    """Generate test data for kernel validation"""

    __test__ = False

    def __init__(self, output_dir: str = "test_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_tensor(self, config: TensorConfig) -> np.ndarray:
        """Generate a random tensor according to config"""
        rng = np.random.default_rng(config.seed)

        if np.issubdtype(config.dtype, np.floating):
            data = rng.uniform(config.range_min, config.range_max, config.shape)
            return data.astype(config.dtype)
        elif np.issubdtype(config.dtype, np.integer):
            data = rng.integers(int(config.range_min), int(config.range_max), config.shape)
            return data.astype(config.dtype)
        else:
            raise ValueError(f"Unsupported dtype: {config.dtype}")

    def generate_relu_f32_golden(
        self,
        shape: tuple[int, ...] = (16,),
        *,
        seed: int = 42,
        range_min: float = -2.0,
        range_max: float = 2.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate deterministic f32 ReLU input and NumPy golden output."""
        input_tensor = self.generate_tensor(
            TensorConfig(shape, np.float32, seed=seed, range_min=range_min, range_max=range_max)
        )
        return input_tensor, np.maximum(input_tensor, 0).astype(np.float32)

    def generate_matmul_f32_golden(
        self,
        lhs_shape: tuple[int, int] = (4, 5),
        rhs_shape: tuple[int, int] = (5, 3),
        *,
        seed: int = 42,
        range_min: float = -1.0,
        range_max: float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate deterministic f32 matmul inputs and NumPy golden output."""
        if lhs_shape[1] != rhs_shape[0]:
            raise ValueError("matmul inner dimensions must match")
        lhs = self.generate_tensor(
            TensorConfig(lhs_shape, np.float32, seed=seed, range_min=range_min, range_max=range_max)
        )
        rhs = self.generate_tensor(
            TensorConfig(
                rhs_shape,
                np.float32,
                seed=seed + 1009,
                range_min=range_min,
                range_max=range_max,
            )
        )
        return lhs, rhs, np.matmul(lhs, rhs).astype(np.float32)

    def save_tensor(self, tensor: np.ndarray, name: str) -> str:
        """Save tensor to binary file and return path"""
        filepath = self.output_dir / f"{name}.bin"
        tensor.astype(tensor.dtype).tofile(filepath)

        # Also save metadata
        metadata_path = self.output_dir / f"{name}.meta"
        with open(metadata_path, "w") as f:
            f.write(f"shape: {tensor.shape}\n")
            f.write(f"dtype: {tensor.dtype}\n")
            f.write(f"size_bytes: {tensor.nbytes}\n")

        return str(filepath)

    def save_reference_result(self, result: np.ndarray, name: str) -> str:
        """Save reference result for comparison"""
        filepath = self.output_dir / f"{name}_ref.bin"
        result.astype(result.dtype).tofile(filepath)
        return str(filepath)

    # MatMul test cases
    def generate_matmul_tests(self) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate MatMul test cases with reference results"""
        cases = []

        # Small test case (4x4 x 4x4)
        A_small = self.generate_tensor(TensorConfig((4, 4), np.float32, seed=1))
        B_small = self.generate_tensor(TensorConfig((4, 4), np.float32, seed=2))
        C_small = np.matmul(A_small, B_small)
        cases.append(("matmul_small", A_small, B_small, C_small))

        # Medium test case (64x64 x 64x64)
        A_med = self.generate_tensor(TensorConfig((64, 64), np.float32, seed=3))
        B_med = self.generate_tensor(TensorConfig((64, 64), np.float32, seed=4))
        C_med = np.matmul(A_med, B_med)
        cases.append(("matmul_medium", A_med, B_med, C_med))

        # Rectangular (64x128 x 128x256)
        A_rect = self.generate_tensor(TensorConfig((64, 128), np.float32, seed=5))
        B_rect = self.generate_tensor(TensorConfig((128, 256), np.float32, seed=6))
        C_rect = np.matmul(A_rect, B_rect)
        cases.append(("matmul_rect", A_rect, B_rect, C_rect))

        # Float16 (32x32 x 32x32)
        A_f16 = self.generate_tensor(TensorConfig((32, 32), np.float16, seed=7))
        B_f16 = self.generate_tensor(TensorConfig((32, 32), np.float16, seed=8))
        C_f16 = np.matmul(A_f16.astype(np.float32), B_f16.astype(np.float32)).astype(np.float16)
        cases.append(("matmul_f16", A_f16, B_f16, C_f16))

        # Large test case (256x256 x 256x256)
        A_large = self.generate_tensor(TensorConfig((256, 256), np.float32, seed=9))
        B_large = self.generate_tensor(TensorConfig((256, 256), np.float32, seed=10))
        C_large = np.matmul(A_large, B_large)
        cases.append(("matmul_large", A_large, B_large, C_large))

        return cases

    # Conv2D test cases
    def generate_conv2d_tests(self) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """Generate Conv2D test cases (NHWC format)"""
        # Note: Reference implementation handled separately due to complexity
        cases = []

        # Small test: 1x8x8x3 input, 3x3x3x16 kernel
        input_small = self.generate_tensor(TensorConfig((1, 8, 8, 3), np.float32, seed=11))
        kernel_small = self.generate_tensor(TensorConfig((3, 3, 3, 16), np.float32, seed=12))
        cases.append(("conv2d_small", input_small, kernel_small, None))  # None = compute separately

        # Medium test: 1x32x32x3 input
        input_med = self.generate_tensor(TensorConfig((1, 32, 32, 3), np.float32, seed=13))
        kernel_med = self.generate_tensor(TensorConfig((3, 3, 3, 32), np.float32, seed=14))
        cases.append(("conv2d_medium", input_med, kernel_med, None))

        return cases

    # Attention test cases
    def generate_attention_tests(
        self,
    ) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]:
        """Generate Attention (SDPA) test cases"""
        cases = []

        # Small: seq_len=8, embed_dim=64
        Q_small = self.generate_tensor(
            TensorConfig((1, 8, 64), np.float32, seed=15, range_min=-0.1, range_max=0.1)
        )
        K_small = self.generate_tensor(
            TensorConfig((1, 8, 64), np.float32, seed=16, range_min=-0.1, range_max=0.1)
        )
        V_small = self.generate_tensor(
            TensorConfig((1, 8, 64), np.float32, seed=17, range_min=-0.1, range_max=0.1)
        )
        cases.append(("attention_small", Q_small, K_small, V_small, None))

        # Medium: seq_len=64, embed_dim=128
        Q_med = self.generate_tensor(
            TensorConfig((1, 64, 128), np.float32, seed=18, range_min=-0.1, range_max=0.1)
        )
        K_med = self.generate_tensor(
            TensorConfig((1, 64, 128), np.float32, seed=19, range_min=-0.1, range_max=0.1)
        )
        V_med = self.generate_tensor(
            TensorConfig((1, 64, 128), np.float32, seed=20, range_min=-0.1, range_max=0.1)
        )
        cases.append(("attention_medium", Q_med, K_med, V_med, None))

        return cases

    def save_all_test_data(self):
        """Generate and save all test data"""
        print("Generating MatMul test data...")
        for name, A, B, C in self.generate_matmul_tests():
            self.save_tensor(A, f"{name}_A")
            self.save_tensor(B, f"{name}_B")
            self.save_reference_result(C, name)
            print(f"  ✓ {name}: A{A.shape} x B{B.shape} -> C{C.shape}")

        print("\nGenerating Conv2D test data...")
        for name, inp, kernel, _ in self.generate_conv2d_tests():
            self.save_tensor(inp, f"{name}_input")
            self.save_tensor(kernel, f"{name}_kernel")
            print(f"  ✓ {name}: input{inp.shape}, kernel{kernel.shape}")

        print("\nGenerating Attention test data...")
        for name, Q, K, V, _ in self.generate_attention_tests():
            self.save_tensor(Q, f"{name}_Q")
            self.save_tensor(K, f"{name}_K")
            self.save_tensor(V, f"{name}_V")
            print(f"  ✓ {name}: Q,K,V shape {Q.shape}")

        print(f"\n✓ All test data saved to {self.output_dir}")


if __name__ == "__main__":
    generator = TestDataGenerator("tests/test_data")
    generator.save_all_test_data()
