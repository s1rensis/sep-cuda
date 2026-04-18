#include <cassert>
#include <cmath>
#include <cstdio>

#include "sep_cuda.h"

using sepcuda::Background;
using sepcuda::BackgroundOptions;
using sepcuda::Error;

static void handle_error(const Error &error, const char *label) {
  if (error.status() == SEP_CUDA_UNAVAILABLE || error.status() == SEP_CUDA_RUNTIME_ERROR) {
    std::fprintf(stderr, "%s skipped: %s\n", label, error.what());
    std::exit(77);
  }
  std::fprintf(stderr, "%s failed: %s\n", label, error.what());
  std::exit(1);
}

static void test_constant_background() {
  float data[64];
  for (float &value : data) {
    value = 10.0f;
  }

  try {
    auto background = Background::from_pixels(data, 8, 8, BackgroundOptions{8, 8, 1, 1, 0.0});
    const auto bg = background.back<>();
    const auto rms = background.rms<>();

    for (size_t i = 0; i < bg.size(); ++i) {
      assert(std::fabs(bg[i] - 10.0f) < 1.0e-4f);
      assert(std::fabs(rms[i]) < 1.0e-4f);
    }

    assert(std::fabs(background.global() - 10.0f) < 1.0e-4f);
    assert(std::fabs(background.global_rms() - 1.0f) < 1.0e-4f);
  } catch (const Error &error) {
    handle_error(error, "constant background");
  }
}

static void test_masked_background() {
  float data[36];
  unsigned char mask[36];
  for (int i = 0; i < 36; ++i) {
    data[i] = 0.1f;
    mask[i] = 0;
  }

  data[1 + 1 * 6] = 1.0f;
  data[4 + 1 * 6] = 1.0f;
  data[1 + 4 * 6] = 1.0f;
  data[4 + 4 * 6] = 1.0f;
  mask[1 + 1 * 6] = 1;
  mask[4 + 1 * 6] = 1;
  mask[1 + 4 * 6] = 1;
  mask[4 + 4 * 6] = 1;

  try {
    auto background = Background::from_pixels(
        data,
        6,
        6,
        mask,
        0.0,
        BackgroundOptions{3, 3, 1, 1, 0.0});
    const auto bg = background.back<>();
    for (float value : bg) {
      assert(std::fabs(value - 0.1f) < 1.0e-4f);
    }
  } catch (const Error &error) {
    handle_error(error, "masked background");
  }
}

static void test_integer_path() {
  int data[16];
  for (int &value : data) {
    value = 100;
  }
  data[5] = 140;

  try {
    auto background = Background::from_pixels(data, 4, 4, BackgroundOptions{4, 4, 1, 1, 0.0});
    auto bg = background.back<int>();
    auto rms = background.rms<int>();

    for (size_t i = 0; i < bg.size(); ++i) {
      assert(bg[i] >= 100 && bg[i] <= 103);
      assert(rms[i] >= 0);
    }

    background.subtract_from(data, 16);
  } catch (const Error &error) {
    handle_error(error, "integer background");
  }
}

int main() {
  test_constant_background();
  test_masked_background();
  test_integer_path();
  return 0;
}
