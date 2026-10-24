// C ABI image generation tests (MOSS-Image, CUDA, 512x512, 4 steps, seed 42).
//
//  (a) the pixels are identical to `sb-moss sample` under the same settings,
//      converted the same way (the CHW floats in its MOSIMG01 output);
//  (b) the same request twice produces the same bytes;
//  (c) image 0 of a batch of 2 equals the single-image result;
//  (d) a progress callback that cancels at step 2 returns SB_ERR_CANCELLED.
//
// Also checks the two-call sizing of sb_image_model_arch and
// sb_image_generate. Skips (exit 77) when the model, the sb-moss binary or a
// CUDA device is absent.

#include <sonicboom/capi.h>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#ifndef SONICBOOM_SRC_DIR
#define SONICBOOM_SRC_DIR "."
#endif
#ifndef SB_MOSS_BIN
#define SB_MOSS_BIN "sb-moss"
#endif

static int g_failures = 0;

static void check(int ok, const char *what) {
  if (!ok) {
    fprintf(stderr, "  FAIL: %s\n", what);
    ++g_failures;
  }
}

static void check_status(sb_status st, sb_error *err, const char *what) {
  if (st != SB_OK) {
    fprintf(stderr, "  FAIL: %s: %s\n", what,
            err ? sb_error_message(err) : "(no error object)");
    if (err) sb_error_free(err);
    ++g_failures;
  }
}

#define IMAGE_SIDE 512u
#define IMAGE_BYTES (IMAGE_SIDE * IMAGE_SIDE * 3u)
#define STEPS 4u
#define SEED 42u
#define PROMPT "a red fox sitting in the snow"

// The MOSIMG01 reference written by `sb-moss sample --image-out`:
// "MOSIMG01", then int64 batch/channels/height/width, then the CHW floats.
typedef struct {
  int64_t batch, channels, height, width;
  float *values;
} moss_image_file;

static int read_mosimg(const char *path, moss_image_file *out) {
  FILE *f = fopen(path, "rb");
  if (!f) return 0;
  char magic[8];
  if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MOSIMG01", 8) != 0) {
    fclose(f);
    return 0;
  }
  int64_t dims[4];
  if (fread(dims, sizeof dims, 1, f) != 1) {
    fclose(f);
    return 0;
  }
  out->batch = dims[0];
  out->channels = dims[1];
  out->height = dims[2];
  out->width = dims[3];
  const size_t count = (size_t)out->batch * out->channels * out->height *
                       out->width;
  out->values = (float *)malloc(count * sizeof(float));
  if (!out->values || fread(out->values, sizeof(float), count, f) != count) {
    free(out->values);
    out->values = NULL;
    fclose(f);
    return 0;
  }
  fclose(f);
  return 1;
}

// The C API's conversion: [-1, 1] CHW floats -> uint8 RGB HWC.
static void convert(const float *chw, int64_t height, int64_t width,
                    uint8_t *rgb) {
  const int64_t plane = height * width;
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      uint8_t *px = rgb + (y * width + x) * 3;
      for (int64_t c = 0; c < 3; ++c) {
        const float v = chw[c * plane + y * width + x];
        const float t = fminf(fmaxf((v + 1.0f) * 0.5f, 0.0f), 1.0f);
        px[c] = (uint8_t)lrintf(t * 255.0f);
      }
    }
  }
}

static int run_sb_moss(const char *model_dir, const char *image_out) {
  pid_t pid = fork();
  if (pid < 0) return 0;
  if (pid == 0) {
    const int devnull = open("/dev/null", O_WRONLY);
    if (devnull >= 0) {
      dup2(devnull, STDOUT_FILENO);
      dup2(devnull, STDERR_FILENO);
      close(devnull);
    }
    char steps[16], seed[32], side[16];
    snprintf(steps, sizeof steps, "%u", STEPS);
    snprintf(seed, sizeof seed, "%u", SEED);
    snprintf(side, sizeof side, "%u", IMAGE_SIDE);
    execl(SB_MOSS_BIN, SB_MOSS_BIN, "sample", "--model", model_dir, "--text",
          PROMPT, "--height", side, "--width", side, "--steps", steps,
          "--seed", seed, "--image-out", image_out, (char *)NULL);
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid) return 0;
  return WIFEXITED(status) && WEXITSTATUS(status) == 0;
}

typedef struct {
  uint32_t calls;
  uint32_t last_step;
  uint32_t total;
  uint32_t cancel_at;
} cancel_state;

static int cancel_cb(void *user, uint32_t image_index, uint32_t step,
                     uint32_t total_steps) {
  cancel_state *s = (cancel_state *)user;
  (void)image_index;
  ++s->calls;
  s->last_step = step;
  s->total = total_steps;
  return s->cancel_at != 0 && step >= s->cancel_at ? 1 : 0;
}

static sb_image_request make_request(uint32_t batch) {
  sb_image_request req;
  memset(&req, 0, sizeof req);
  req.struct_size = sizeof req;
  req.prompt = PROMPT;
  req.negative_prompt = NULL;
  req.width = IMAGE_SIDE;
  req.height = IMAGE_SIDE;
  req.steps = STEPS;
  req.guidance_scale = 3.0f;
  req.seed = SEED;
  req.batch = batch;
  return req;
}

int main(void) {
  const char *root = SONICBOOM_SRC_DIR;
  char model_dir[4096];
  snprintf(model_dir, sizeof model_dir, "%s/models/%s", root,
           "MOSS-Image-SFT");
  char probe[4200];
  snprintf(probe, sizeof probe, "%s/transformer", model_dir);
  if (access(probe, F_OK) != 0) {
    fprintf(stderr, "SKIP: %s absent\n", model_dir);
    return 77;
  }
  if (access(SB_MOSS_BIN, X_OK) != 0) {
    fprintf(stderr, "SKIP: %s not built\n", SB_MOSS_BIN);
    return 77;
  }
  int ndev = 0;
  if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev < 1) {
    fprintf(stderr, "SKIP: no CUDA device\n");
    return 77;
  }

  char ref_path[4096];
  snprintf(ref_path, sizeof ref_path, "/tmp/sb-image-ref-%ld.mosimg",
           (long)getpid());

  // (a) reference first, in its own process: two copies of the pipeline do not
  // fit in device memory at once.
  if (!run_sb_moss(model_dir, ref_path)) {
    fprintf(stderr, "FAIL: sb-moss sample did not succeed\n");
    return 1;
  }
  moss_image_file ref;
  if (!read_mosimg(ref_path, &ref)) {
    fprintf(stderr, "FAIL: cannot read %s\n", ref_path);
    return 1;
  }
  unlink(ref_path);
  check(ref.batch == 1 && ref.channels == 3 && ref.height == IMAGE_SIDE &&
            ref.width == IMAGE_SIDE,
        "(a) reference is 1x3x512x512");
  if (ref.batch != 1 || ref.channels != 3 || ref.height != IMAGE_SIDE ||
      ref.width != IMAGE_SIDE) {
    free(ref.values);
    return 1;
  }
  uint8_t *ref_rgb = (uint8_t *)malloc(IMAGE_BYTES);
  convert(ref.values, ref.height, ref.width, ref_rgb);
  free(ref.values);

  sb_error *err = NULL;
  sb_image_model *model = NULL;
  check_status(sb_image_model_load(model_dir, &model, &err), err, "load");
  if (model == NULL) return 1;

  // Architecture name, two-call sizing.
  size_t arch_len = 0;
  check(sb_image_model_arch(model, NULL, 0, &arch_len, &err) ==
            SB_ERR_BINDING,
        "arch: a NULL buffer asks for the size");
  if (err) {
    sb_error_free(err);
    err = NULL;
  }
  char arch[32];
  check(arch_len == strlen("moss-image"), "arch: len excludes the NUL");
  check_status(sb_image_model_arch(model, arch, sizeof arch, &arch_len, &err),
               err, "arch");
  check(strcmp(arch, "moss-image") == 0, "arch: moss-image");

  // (a) generate, two-call sizing first.
  uint8_t *rgb = (uint8_t *)malloc(IMAGE_BYTES);
  sb_image_request req = make_request(1);
  size_t needed = 0;
  check(sb_image_generate(model, &req, NULL, NULL, NULL, 0, &needed, &err) ==
            SB_ERR_BINDING,
        "(a) a NULL buffer asks for the byte count");
  if (err) {
    sb_error_free(err);
    err = NULL;
  }
  check(needed == IMAGE_BYTES, "(a) the byte count is 512*512*3");
  check_status(sb_image_generate(model, &req, NULL, NULL, rgb, IMAGE_BYTES,
                                 &needed, &err),
               err, "(a) generate");
  if (memcmp(rgb, ref_rgb, IMAGE_BYTES) != 0) {
    size_t i = 0;
    while (i < IMAGE_BYTES && rgb[i] == ref_rgb[i]) ++i;
    fprintf(stderr, "    first differing byte at %zu: %u vs %u\n", i, rgb[i],
            ref_rgb[i]);
  }
  check(memcmp(rgb, ref_rgb, IMAGE_BYTES) == 0,
        "(a) pixels equal sb-moss sample");

  // (b) the same request twice is identical.
  uint8_t *again = (uint8_t *)malloc(IMAGE_BYTES);
  check_status(sb_image_generate(model, &req, NULL, NULL, again, IMAGE_BYTES,
                                 &needed, &err),
               err, "(b) generate again");
  check(memcmp(rgb, again, IMAGE_BYTES) == 0,
        "(b) the same request is deterministic");

  // (c) image 0 of a batch of 2 equals the single image.
  uint8_t *pair = (uint8_t *)malloc(2 * IMAGE_BYTES);
  sb_image_request req2 = make_request(2);
  check_status(sb_image_generate(model, &req2, NULL, NULL, pair,
                                 2 * IMAGE_BYTES, &needed, &err),
               err, "(c) generate a batch of 2");
  check(needed == 2 * IMAGE_BYTES, "(c) the batch needs 2x the bytes");
  check(memcmp(pair, rgb, IMAGE_BYTES) == 0,
        "(c) batch image 0 equals the single image");

  // (d) cancelling at step 2 returns SB_ERR_CANCELLED with the cancelled kind.
  cancel_state state = {0, 0, 0, 2};
  err = NULL;
  const sb_status st = sb_image_generate(model, &req, cancel_cb, &state, again,
                                         IMAGE_BYTES, &needed, &err);
  check(st == SB_ERR_CANCELLED, "(d) cancelling returns SB_ERR_CANCELLED");
  check(err != NULL && sb_error_get_kind(err) == SB_ERRKIND_CANCELLED,
        "(d) the error kind is cancelled");
  if (err) {
    sb_error_free(err);
    err = NULL;
  }
  check(state.calls == 2 && state.last_step == 2 && state.total == STEPS,
        "(d) the callback saw steps 1 and 2 of 4");

  free(pair);
  free(again);
  free(rgb);
  free(ref_rgb);
  sb_image_model_free(model);

  if (g_failures != 0) {
    fprintf(stderr, "%d image failure(s)\n", g_failures);
    return 1;
  }
  printf("test_capi_image: all checks passed\n");
  return 0;
}
