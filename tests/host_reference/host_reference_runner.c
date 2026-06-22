#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_elementwise.h"
#include "kernelsmith/ks_matmul.h"
#include "kernelsmith/ks_normalization.h"
#include "kernelsmith/ks_quantized.h"

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int parse_size(const char *text, size_t *value) {
  char *end = NULL;
  errno = 0;
  unsigned long long parsed = strtoull(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0')
    return 0;
  *value = (size_t)parsed;
  return 1;
}

static int parse_i32(const char *text, int32_t *value) {
  char *end = NULL;
  errno = 0;
  long parsed = strtol(text, &end, 10);
  if (errno != 0 || end == text || *end != '\0' || parsed < INT32_MIN ||
      parsed > INT32_MAX)
    return 0;
  *value = (int32_t)parsed;
  return 1;
}

static int parse_float(const char *text, float *value) {
  char *end = NULL;
  errno = 0;
  float parsed = strtof(text, &end);
  if (errno != 0 || end == text || *end != '\0')
    return 0;
  *value = parsed;
  return 1;
}

static int read_raw_file(const char *path, void *buffer, size_t element_size,
                         size_t elements, const char *dtype) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "failed to open %s for reading: %s\n", path,
            strerror(errno));
    return 1;
  }

  size_t read = fread(buffer, element_size, elements, file);
  int extra = fgetc(file);
  if (ferror(file)) {
    fprintf(stderr, "failed to read %s\n", path);
    fclose(file);
    return 1;
  }
  fclose(file);

  if (read != elements || extra != EOF) {
    fprintf(stderr, "unexpected size for %s: read %zu %s values, expected %zu\n",
            path, read, dtype, elements);
    return 1;
  }
  return 0;
}

static int write_raw_file(const char *path, const void *buffer,
                          size_t element_size, size_t elements) {
  FILE *file = fopen(path, "wb");
  if (!file) {
    fprintf(stderr, "failed to open %s for writing: %s\n", path,
            strerror(errno));
    return 1;
  }

  size_t written = fwrite(buffer, element_size, elements, file);
  if (written != elements || ferror(file)) {
    fprintf(stderr, "failed to write %s\n", path);
    fclose(file);
    return 1;
  }
  fclose(file);
  return 0;
}

static int read_f32_file(const char *path, float *buffer, size_t elements) {
  return read_raw_file(path, buffer, sizeof(float), elements, "f32");
}

static int read_i8_file(const char *path, int8_t *buffer, size_t elements) {
  return read_raw_file(path, buffer, sizeof(int8_t), elements, "i8");
}

static int read_u8_file(const char *path, uint8_t *buffer, size_t elements) {
  return read_raw_file(path, buffer, sizeof(uint8_t), elements, "u8");
}

static int write_f32_file(const char *path, const float *buffer,
                          size_t elements) {
  return write_raw_file(path, buffer, sizeof(float), elements);
}

static int write_i32_file(const char *path, const int32_t *buffer,
                          size_t elements) {
  return write_raw_file(path, buffer, sizeof(int32_t), elements);
}

static int run_relu(int argc, char **argv) {
  if (argc != 5) {
    fprintf(stderr, "usage: %s relu <input.raw> <output.raw> <n>\n", argv[0]);
    return 2;
  }

  size_t n = 0;
  if (!parse_size(argv[4], &n)) {
    fprintf(stderr, "invalid relu element count: %s\n", argv[4]);
    return 2;
  }

  float *input = (float *)malloc(n * sizeof(float));
  float *output = (float *)malloc(n * sizeof(float));
  if (!input || !output) {
    fprintf(stderr, "failed to allocate relu buffers\n");
    free(input);
    free(output);
    return 1;
  }

  int status = read_f32_file(argv[2], input, n);
  if (status == 0)
    status = ks_relu_f32(input, output, n);
  if (status == 0)
    status = write_f32_file(argv[3], output, n);

  free(input);
  free(output);
  return status;
}

static int run_binary_f32(int argc, char **argv, const char *name) {
  if (argc != 6) {
    fprintf(stderr, "usage: %s %s <lhs.raw> <rhs.raw> <output.raw> <n>\n",
            argv[0], name);
    return 2;
  }

  size_t n = 0;
  if (!parse_size(argv[5], &n)) {
    fprintf(stderr, "invalid %s element count: %s\n", name, argv[5]);
    return 2;
  }

  float *lhs = (float *)malloc(n * sizeof(float));
  float *rhs = (float *)malloc(n * sizeof(float));
  float *output = (float *)malloc(n * sizeof(float));
  if (!lhs || !rhs || !output) {
    fprintf(stderr, "failed to allocate %s buffers\n", name);
    free(lhs);
    free(rhs);
    free(output);
    return 1;
  }

  int status = read_f32_file(argv[2], lhs, n);
  if (status == 0)
    status = read_f32_file(argv[3], rhs, n);
  if (status == 0 && strcmp(name, "add") == 0)
    status = ks_add_f32(lhs, rhs, output, n);
  else if (status == 0)
    status = ks_mul_f32(lhs, rhs, output, n);
  if (status == 0)
    status = write_f32_file(argv[4], output, n);

  free(lhs);
  free(rhs);
  free(output);
  return status;
}

static int run_matmul(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: %s matmul <lhs.raw> <rhs.raw> <output.raw> <M> <N> <K>\n",
            argv[0]);
    return 2;
  }

  size_t m = 0;
  size_t n = 0;
  size_t k = 0;
  if (!parse_size(argv[5], &m) || !parse_size(argv[6], &n) ||
      !parse_size(argv[7], &k)) {
    fprintf(stderr, "invalid matmul dimensions\n");
    return 2;
  }

  float *lhs = (float *)malloc(m * k * sizeof(float));
  float *rhs = (float *)malloc(k * n * sizeof(float));
  float *output = (float *)malloc(m * n * sizeof(float));
  size_t workspace_size = ks_matmul_f32_workspace(m, n, k);
  void *workspace = workspace_size == 0 ? NULL : malloc(workspace_size);
  if (!lhs || !rhs || !output || (workspace_size != 0 && !workspace)) {
    fprintf(stderr, "failed to allocate matmul buffers\n");
    free(lhs);
    free(rhs);
    free(output);
    free(workspace);
    return 1;
  }

  int status = read_f32_file(argv[2], lhs, m * k);
  if (status == 0)
    status = read_f32_file(argv[3], rhs, k * n);
  if (status == 0)
    status = ks_matmul_f32(lhs, k, rhs, n, output, n, m, n, k, workspace,
                           workspace_size);
  if (status == 0)
    status = write_f32_file(argv[4], output, m * n);

  free(lhs);
  free(rhs);
  free(output);
  free(workspace);
  return status;
}

static int run_rms_norm(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: %s rms_norm <input.raw> <weight.raw> <output.raw> "
            "<outer> <inner> <eps>\n",
            argv[0]);
    return 2;
  }

  size_t outer = 0;
  size_t inner = 0;
  float eps = 0.0f;
  if (!parse_size(argv[5], &outer) || !parse_size(argv[6], &inner) ||
      !parse_float(argv[7], &eps)) {
    fprintf(stderr, "invalid rms_norm arguments\n");
    return 2;
  }

  float *input = (float *)malloc(outer * inner * sizeof(float));
  float *weight = (float *)malloc(inner * sizeof(float));
  float *output = (float *)malloc(outer * inner * sizeof(float));
  if (!input || !weight || !output) {
    fprintf(stderr, "failed to allocate rms_norm buffers\n");
    free(input);
    free(weight);
    free(output);
    return 1;
  }

  int status = read_f32_file(argv[2], input, outer * inner);
  if (status == 0)
    status = read_f32_file(argv[3], weight, inner);
  if (status == 0)
    status = ks_rms_norm_f32(input, weight, output, outer, inner, eps);
  if (status == 0)
    status = write_f32_file(argv[4], output, outer * inner);

  free(input);
  free(weight);
  free(output);
  return status;
}

static int run_softmax(int argc, char **argv) {
  if (argc != 6) {
    fprintf(stderr,
            "usage: %s softmax <input.raw> <output.raw> <outer> <inner>\n",
            argv[0]);
    return 2;
  }

  size_t outer = 0;
  size_t inner = 0;
  if (!parse_size(argv[4], &outer) || !parse_size(argv[5], &inner)) {
    fprintf(stderr, "invalid softmax arguments\n");
    return 2;
  }

  float *input = (float *)malloc(outer * inner * sizeof(float));
  float *output = (float *)malloc(outer * inner * sizeof(float));
  if (!input || !output) {
    fprintf(stderr, "failed to allocate softmax buffers\n");
    free(input);
    free(output);
    return 1;
  }

  int status = read_f32_file(argv[2], input, outer * inner);
  if (status == 0)
    status = ks_softmax_f32(input, output, outer, inner);
  if (status == 0)
    status = write_f32_file(argv[3], output, outer * inner);

  free(input);
  free(output);
  return status;
}

static int run_dot_i8(int argc, char **argv) {
  if (argc != 8) {
    fprintf(stderr,
            "usage: %s dot_i8 <input.raw> <weight.raw> <output.raw> <K> "
            "<input_zp> <weight_zp>\n",
            argv[0]);
    return 2;
  }

  size_t k = 0;
  int32_t input_zp = 0;
  int32_t weight_zp = 0;
  if (!parse_size(argv[5], &k) || !parse_i32(argv[6], &input_zp) ||
      !parse_i32(argv[7], &weight_zp)) {
    fprintf(stderr, "invalid dot_i8 arguments\n");
    return 2;
  }

  int8_t *input = (int8_t *)malloc(k * sizeof(int8_t));
  int8_t *weight = (int8_t *)malloc(k * sizeof(int8_t));
  int32_t output = 0;
  if (!input || !weight) {
    fprintf(stderr, "failed to allocate dot_i8 buffers\n");
    free(input);
    free(weight);
    return 1;
  }

  int status = read_i8_file(argv[2], input, k);
  if (status == 0)
    status = read_i8_file(argv[3], weight, k);
  if (status == 0)
    status = ks_dot_i8(input, weight, k, input_zp, weight_zp, &output, NULL, 0);
  if (status == 0)
    status = write_i32_file(argv[4], &output, 1);

  free(input);
  free(weight);
  return status;
}

static int run_matvec_i8(int argc, char **argv) {
  if (argc != 9) {
    fprintf(stderr,
            "usage: %s matvec_i8 <input.raw> <weights.raw> <output.raw> "
            "<rows> <cols> <input_zp> <weight_zp>\n",
            argv[0]);
    return 2;
  }

  size_t rows = 0;
  size_t cols = 0;
  int32_t input_zp = 0;
  int32_t weight_zp = 0;
  if (!parse_size(argv[5], &rows) || !parse_size(argv[6], &cols) ||
      !parse_i32(argv[7], &input_zp) || !parse_i32(argv[8], &weight_zp)) {
    fprintf(stderr, "invalid matvec_i8 arguments\n");
    return 2;
  }

  int8_t *input = (int8_t *)malloc(cols * sizeof(int8_t));
  int8_t *weights = (int8_t *)malloc(rows * cols * sizeof(int8_t));
  int32_t *output = (int32_t *)malloc(rows * sizeof(int32_t));
  if (!input || !weights || !output) {
    fprintf(stderr, "failed to allocate matvec_i8 buffers\n");
    free(input);
    free(weights);
    free(output);
    return 1;
  }

  int status = read_i8_file(argv[2], input, cols);
  if (status == 0)
    status = read_i8_file(argv[3], weights, rows * cols);
  if (status == 0)
    status = ks_matvec_i8(input, weights, cols, output, rows, cols, input_zp,
                          weight_zp, NULL, 0);
  if (status == 0)
    status = write_i32_file(argv[4], output, rows);

  free(input);
  free(weights);
  free(output);
  return status;
}

static int run_dot_w4a8(int argc, char **argv) {
  if (argc != 11) {
    fprintf(stderr,
            "usage: %s dot_w4a8 <input.raw> <weight.raw> <scales.raw> "
            "<output.raw> <K> <group_size> <input_scale> <input_zp> "
            "<weight_zp>\n",
            argv[0]);
    return 2;
  }

  size_t k = 0;
  size_t group_size = 0;
  float input_scale = 0.0f;
  int32_t input_zp = 0;
  int32_t weight_zp = 0;
  if (!parse_size(argv[6], &k) || !parse_size(argv[7], &group_size) ||
      !parse_float(argv[8], &input_scale) || !parse_i32(argv[9], &input_zp) ||
      !parse_i32(argv[10], &weight_zp)) {
    fprintf(stderr, "invalid dot_w4a8 arguments\n");
    return 2;
  }

  size_t packed_bytes = ks_w4a8_packed_row_bytes(k);
  size_t scale_groups = ks_w4a8_scale_groups(k, group_size);
  int8_t *input = (int8_t *)malloc(k * sizeof(int8_t));
  uint8_t *weight = (uint8_t *)malloc(packed_bytes * sizeof(uint8_t));
  float *scales = (float *)malloc(scale_groups * sizeof(float));
  float output = 0.0f;
  if (!input || !weight || !scales) {
    fprintf(stderr, "failed to allocate dot_w4a8 buffers\n");
    free(input);
    free(weight);
    free(scales);
    return 1;
  }

  int status = read_i8_file(argv[2], input, k);
  if (status == 0)
    status = read_u8_file(argv[3], weight, packed_bytes);
  if (status == 0)
    status = read_f32_file(argv[4], scales, scale_groups);
  if (status == 0)
    status = ks_dot_w4a8(input, weight, scales, k, group_size, input_scale,
                         input_zp, weight_zp, &output, NULL, 0);
  if (status == 0)
    status = write_f32_file(argv[5], &output, 1);

  free(input);
  free(weight);
  free(scales);
  return status;
}

static int run_matvec_w4a8(int argc, char **argv) {
  if (argc != 12) {
    fprintf(stderr,
            "usage: %s matvec_w4a8 <input.raw> <weights.raw> <scales.raw> "
            "<output.raw> <rows> <cols> <group_size> <input_scale> "
            "<input_zp> <weight_zp>\n",
            argv[0]);
    return 2;
  }

  size_t rows = 0;
  size_t cols = 0;
  size_t group_size = 0;
  float input_scale = 0.0f;
  int32_t input_zp = 0;
  int32_t weight_zp = 0;
  if (!parse_size(argv[6], &rows) || !parse_size(argv[7], &cols) ||
      !parse_size(argv[8], &group_size) || !parse_float(argv[9], &input_scale) ||
      !parse_i32(argv[10], &input_zp) || !parse_i32(argv[11], &weight_zp)) {
    fprintf(stderr, "invalid matvec_w4a8 arguments\n");
    return 2;
  }

  size_t packed_row_bytes = ks_w4a8_packed_row_bytes(cols);
  size_t scale_groups = ks_w4a8_scale_groups(cols, group_size);
  int8_t *input = (int8_t *)malloc(cols * sizeof(int8_t));
  uint8_t *weights =
      (uint8_t *)malloc(rows * packed_row_bytes * sizeof(uint8_t));
  float *scales = (float *)malloc(rows * scale_groups * sizeof(float));
  float *output = (float *)malloc(rows * sizeof(float));
  if (!input || !weights || !scales || !output) {
    fprintf(stderr, "failed to allocate matvec_w4a8 buffers\n");
    free(input);
    free(weights);
    free(scales);
    free(output);
    return 1;
  }

  int status = read_i8_file(argv[2], input, cols);
  if (status == 0)
    status = read_u8_file(argv[3], weights, rows * packed_row_bytes);
  if (status == 0)
    status = read_f32_file(argv[4], scales, rows * scale_groups);
  if (status == 0)
    status = ks_matvec_w4a8(input, weights, packed_row_bytes, scales,
                            scale_groups, output, rows, cols, group_size,
                            input_scale, input_zp, weight_zp, NULL, 0);
  if (status == 0)
    status = write_f32_file(argv[5], output, rows);

  free(input);
  free(weights);
  free(scales);
  free(output);
  return status;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr,
            "usage: %s <add|relu|mul|matmul|rms_norm|softmax|dot_i8|"
            "matvec_i8|dot_w4a8|matvec_w4a8> ...\n",
            argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "add") == 0)
    return run_binary_f32(argc, argv, "add");
  if (strcmp(argv[1], "relu") == 0)
    return run_relu(argc, argv);
  if (strcmp(argv[1], "mul") == 0)
    return run_binary_f32(argc, argv, "mul");
  if (strcmp(argv[1], "matmul") == 0)
    return run_matmul(argc, argv);
  if (strcmp(argv[1], "rms_norm") == 0)
    return run_rms_norm(argc, argv);
  if (strcmp(argv[1], "softmax") == 0)
    return run_softmax(argc, argv);
  if (strcmp(argv[1], "dot_i8") == 0)
    return run_dot_i8(argc, argv);
  if (strcmp(argv[1], "matvec_i8") == 0)
    return run_matvec_i8(argc, argv);
  if (strcmp(argv[1], "dot_w4a8") == 0)
    return run_dot_w4a8(argc, argv);
  if (strcmp(argv[1], "matvec_w4a8") == 0)
    return run_matvec_w4a8(argc, argv);

  fprintf(stderr, "unsupported kernel: %s\n", argv[1]);
  return 2;
}
