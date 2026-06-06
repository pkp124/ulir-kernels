/*
 * Generic RISC-V functional harness for generated KernelSmith RVV objects.
 *
 * The generated object must export a no-argument function returning the
 * maximum absolute error observed by its self-checking kernel body.
 */

#include <stdint.h>
#include <stdio.h>
#include <time.h>

#ifndef KS_RISCV_TEST_NAME
#define KS_RISCV_TEST_NAME "unnamed"
#endif

#ifndef KS_RISCV_MAX_ERROR_FUNCTION
#define KS_RISCV_MAX_ERROR_FUNCTION ks_rvv_max_abs_error
#endif

#ifndef KS_RISCV_TOLERANCE
#define KS_RISCV_TOLERANCE 1.0e-5f
#endif

extern float KS_RISCV_MAX_ERROR_FUNCTION(void);

static uint64_t now_ns(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

int main(void) {
  uint64_t start = now_ns();
  float max_abs_error = KS_RISCV_MAX_ERROR_FUNCTION();
  uint64_t elapsed_ns = now_ns() - start;

  printf("CASE: %s\n", KS_RISCV_TEST_NAME);
  printf("MAX_ABS_ERROR: %.9g\n", (double)max_abs_error);
  printf("TIME_NS: %llu\n", (unsigned long long)elapsed_ns);

  if (max_abs_error <= KS_RISCV_TOLERANCE) {
    puts("PASS");
    return 0;
  }

  puts("FAIL");
  return 1;
}
