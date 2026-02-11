/*
 * KernelSmith Common Definitions
 *
 * Version, error codes, and target information.
 * Pure C99 — no C++ in this header.
 */

#ifndef KERNELSMITH_COMMON_H
#define KERNELSMITH_COMMON_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Version ────────────────────────────────────────────────────────────── */

#define KS_VERSION_MAJOR 0
#define KS_VERSION_MINOR 1
#define KS_VERSION_PATCH 0

/* ── Error Codes ────────────────────────────────────────────────────────── */

#define KS_OK              0
#define KS_ERR_INVALID_ARG (-1)
#define KS_ERR_WORKSPACE   (-2)
#define KS_ERR_UNSUPPORTED (-3)

/* ── Target Information ─────────────────────────────────────────────────── */

/*
 * ks_target_name - Returns the name of the compiled target profile.
 *
 * The string is a compile-time constant (e.g. "generic", "x86-avx2").
 * Never returns NULL.
 */
const char *ks_target_name(void);

/*
 * ks_simd_width_f32 - Returns the number of f32 elements per SIMD register.
 *
 * Returns 1 for scalar (generic) target.
 */
size_t ks_simd_width_f32(void);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_COMMON_H */
