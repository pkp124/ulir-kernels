/*
 * Small ReLU caller for the public C API.
 *
 *   cc -std=c99 examples/relu.c -I include -include target/generic.h \
 *     lib/kernelsmith/ks_common.c lib/kernelsmith/ks_activations.c \
 *     -lm -o /tmp/ks_relu && /tmp/ks_relu
 */

#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_common.h"

#include <stdio.h>

int main(void) {
  float in[4] = {-1.0f, 0.0f, 0.5f, 2.0f};
  float out[4];
  size_t i;

  if (ks_relu_f32(in, out, 4) != KS_OK)
    return 1;

  for (i = 0; i < 4; i++)
    printf("%g\n", out[i]);
  return 0;
}
