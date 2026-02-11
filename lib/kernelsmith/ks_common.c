/*
 * KernelSmith Common — target information compiled from profile.
 */

#include "kernelsmith/ks_common.h"

/* Target profile is force-included via -include in CMake. */

const char *ks_target_name(void) { return KS_TARGET_NAME; }

size_t ks_simd_width_f32(void) { return KS_SIMD_WIDTH_F32; }
