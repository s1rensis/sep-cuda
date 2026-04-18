#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "sep_cuda_addon.h"
#include "sep_cuda.h"
#include "sep_internal.h"

namespace {

void fill_scene(std::vector<uint16_t> &image, int64_t width, int64_t height) {
  for (int64_t y = 0; y < height; ++y) {
    for (int64_t x = 0; x < width; ++x) {
      image[static_cast<size_t>(y * width + x)] =
          static_cast<uint16_t>((x * 7 + y * 13 + ((x + y) % 17) * 23) % 4096 + 900);
    }
  }
}

int max_abs_diff(const std::vector<uint16_t> &lhs, const std::vector<uint16_t> &rhs) {
  int diff = 0;
  for (size_t i = 0; i < lhs.size(); ++i) {
    const int delta = std::abs(static_cast<int>(lhs[i]) - static_cast<int>(rhs[i]));
    if (delta > diff) {
      diff = delta;
    }
  }
  return diff;
}

}  // namespace

int main() {
  constexpr int64_t width = 128;
  constexpr int64_t height = 96;
  std::vector<uint16_t> image(static_cast<size_t>(width * height));
  std::vector<uint16_t> ref_sub(image.size());
  std::vector<uint16_t> ref_rms(image.size());
  std::vector<uint16_t> opt_sub(image.size());
  std::vector<uint16_t> opt_rms(image.size());

  fill_scene(image, width, height);

  int status = sep_cuda_subtract_background_and_fill_rms_u16_reference(
      image.data(), width, height, 32, 32, 3, 3, 0.0, ref_sub.data(), ref_rms.data());
  if (status == SEP_CUDA_UNAVAILABLE || status == SEP_CUDA_RUNTIME_ERROR) {
    return 77;
  }
  if (status != RETURN_OK) {
    char err[128];
    char detail[512];
    sep_get_errmsg(status, err);
    sep_get_errdetail(detail);
    std::fprintf(stderr, "reference u16 pipeline failed: %s (%s)\n", err, detail);
    return 1;
  }

  status = sep_cuda_subtract_background_and_fill_rms_u16(
      image.data(), width, height, 32, 32, 3, 3, 0.0, opt_sub.data(), opt_rms.data());
  if (status == SEP_CUDA_UNAVAILABLE || status == SEP_CUDA_RUNTIME_ERROR) {
    return 77;
  }
  if (status != RETURN_OK) {
    char err[128];
    char detail[512];
    sep_get_errmsg(status, err);
    sep_get_errdetail(detail);
    std::fprintf(stderr, "optimized u16 pipeline failed: %s (%s)\n", err, detail);
    return 1;
  }

  if (max_abs_diff(ref_sub, opt_sub) > 1 || max_abs_diff(ref_rms, opt_rms) > 1) {
    std::fprintf(stderr, "u16 pipeline mismatch exceeds tolerance\n");
    return 1;
  }

  return 0;
}
