#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_matmul.h"

#include <errno.h>
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

static int read_f32_file(const char *path, float *buffer, size_t elements) {
  FILE *file = fopen(path, "rb");
  if (!file) {
    fprintf(stderr, "failed to open %s for reading: %s\n", path,
            strerror(errno));
    return 1;
  }

  size_t read = fread(buffer, sizeof(float), elements, file);
  int extra = fgetc(file);
  if (ferror(file)) {
    fprintf(stderr, "failed to read %s\n", path);
    fclose(file);
    return 1;
  }
  fclose(file);

  if (read != elements || extra != EOF) {
    fprintf(stderr, "unexpected size for %s: read %zu f32 values, expected %zu\n",
            path, read, elements);
    return 1;
  }
  return 0;
}

static int write_f32_file(const char *path, const float *buffer,
                          size_t elements) {
  FILE *file = fopen(path, "wb");
  if (!file) {
    fprintf(stderr, "failed to open %s for writing: %s\n", path,
            strerror(errno));
    return 1;
  }

  size_t written = fwrite(buffer, sizeof(float), elements, file);
  if (written != elements || ferror(file)) {
    fprintf(stderr, "failed to write %s\n", path);
    fclose(file);
    return 1;
  }
  fclose(file);
  return 0;
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

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <relu|matmul> ...\n", argv[0]);
    return 2;
  }

  if (strcmp(argv[1], "relu") == 0)
    return run_relu(argc, argv);
  if (strcmp(argv[1], "matmul") == 0)
    return run_matmul(argc, argv);

  fprintf(stderr, "unsupported kernel: %s\n", argv[1]);
  return 2;
}
