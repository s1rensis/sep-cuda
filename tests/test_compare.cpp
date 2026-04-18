#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "sep_cuda.h"

namespace {

constexpr int kWidth = 4096;
constexpr int kHeight = 4096;
constexpr int kNumSources = 10000;
constexpr uint32_t kSeed = 20260418u;
constexpr double kBackgroundMean = 1000.0;
constexpr double kBackgroundSigma = 15.0;
constexpr double kFluxMin = 2000.0;
constexpr double kFluxMax = 20000.0;
constexpr double kRhoMin = 0.8;
constexpr double kRhoMax = 2.4;
constexpr int64_t kBw = 32;
constexpr int64_t kBh = 32;
constexpr int64_t kFw = 3;
constexpr int64_t kFh = 3;
constexpr double kFthresh = 0.0;

constexpr float kGlobalTolerance = 0.5f;
constexpr float kMapMaxTolerance = 2.0f;
constexpr float kMapMeanTolerance = 0.15f;

using sep_background_fn =
    int (*)(const sep_image *, int64_t, int64_t, int64_t, int64_t, double, sep_bkg **);
using sep_bkg_array_fn = int (*)(const sep_bkg *, void *, int);
using sep_bkg_rmsarray_fn = int (*)(const sep_bkg *, void *, int);
using sep_bkg_global_fn = float (*)(const sep_bkg *);
using sep_bkg_free_fn = void (*)(sep_bkg *);
using sep_get_errmsg_fn = void (*)(int, char *);
using sep_get_errdetail_fn = void (*)(char *);

struct sep_api {
  void *handle;
  sep_background_fn background;
  sep_bkg_array_fn bkg_array;
  sep_bkg_rmsarray_fn rms_array;
  sep_bkg_global_fn global;
  sep_bkg_global_fn globalrms;
  sep_bkg_free_fn bkg_free;
  sep_get_errmsg_fn get_errmsg;
  sep_get_errdetail_fn get_errdetail;
};

struct DiffStats {
  double mean_abs;
  double rms;
  double max_abs;
  size_t max_index;
};

struct CompareResult {
  std::string dtype_name;
  double gpu_background_ms;
  double cpu_background_ms;
  double gpu_expand_ms;
  double cpu_expand_ms;
  float gpu_global;
  float cpu_global;
  float gpu_globalrms;
  float cpu_globalrms;
  DiffStats back_stats;
  DiffStats rms_stats;
};

void load_symbol(void **out, void *handle, const char *name) {
  *out = dlsym(handle, name);
  if (*out == nullptr) {
    std::fprintf(stderr, "failed to resolve symbol %s: %s\n", name, dlerror());
    std::exit(1);
  }
}

sep_api load_api(const char *path) {
  sep_api api{};
  api.handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
  if (api.handle == nullptr) {
    std::fprintf(stderr, "failed to open %s: %s\n", path, dlerror());
    std::exit(1);
  }

  load_symbol(reinterpret_cast<void **>(&api.background), api.handle, "sep_background");
  load_symbol(reinterpret_cast<void **>(&api.bkg_array), api.handle, "sep_bkg_array");
  load_symbol(reinterpret_cast<void **>(&api.rms_array), api.handle, "sep_bkg_rmsarray");
  load_symbol(reinterpret_cast<void **>(&api.global), api.handle, "sep_bkg_global");
  load_symbol(reinterpret_cast<void **>(&api.globalrms), api.handle, "sep_bkg_globalrms");
  load_symbol(reinterpret_cast<void **>(&api.bkg_free), api.handle, "sep_bkg_free");
  load_symbol(reinterpret_cast<void **>(&api.get_errmsg), api.handle, "sep_get_errmsg");
  load_symbol(reinterpret_cast<void **>(&api.get_errdetail), api.handle, "sep_get_errdetail");
  return api;
}

void check_status(const sep_api *api, int status, const char *label) {
  if (status != RETURN_OK) {
    char err[128];
    char detail[512];
    api->get_errmsg(status, err);
    api->get_errdetail(detail);
    if (status == SEP_CUDA_UNAVAILABLE || status == SEP_CUDA_RUNTIME_ERROR) {
      std::fprintf(stderr, "%s skipped: %s (%s)\n", label, err, detail);
      std::exit(77);
    }
    std::fprintf(stderr, "%s failed: %s (%s)\n", label, err, detail);
    std::exit(1);
  }
}

void convsurf(std::vector<float> &image, double x, double y, double g, double rho) {
  const double scaled_rho = rho * std::sqrt(2.0);
  const int x0 = std::max(0, static_cast<int>(std::llround(x - 2.0 * scaled_rho)));
  const int x1 = std::min(kWidth - 1, static_cast<int>(std::llround(x + 2.0 * scaled_rho)));
  const int y0 = std::max(0, static_cast<int>(std::llround(y - 2.0 * scaled_rho)));
  const int y1 = std::min(kHeight - 1, static_cast<int>(std::llround(y + 2.0 * scaled_rho)));

  const double g_peak =
      (std::erf(0.5 / scaled_rho) - std::erf(-0.5 / scaled_rho)) *
      (std::erf(0.5 / scaled_rho) - std::erf(-0.5 / scaled_rho));
  const double coeff = g / g_peak;

  for (int iy = y0; iy <= y1; ++iy) {
    for (int ix = x0; ix <= x1; ++ix) {
      const double response =
          (std::erf((ix + 0.5 - x) / scaled_rho) - std::erf((ix - 0.5 - x) / scaled_rho)) *
          (std::erf((iy + 0.5 - y) / scaled_rho) - std::erf((iy - 0.5 - y) / scaled_rho));
      const float candidate = static_cast<float>(coeff * response);
      float &pixel = image[static_cast<size_t>(iy) * kWidth + ix];
      pixel = std::max(pixel, candidate);
    }
  }
}

std::vector<float> make_scene() {
  std::vector<float> image(static_cast<size_t>(kWidth) * kHeight);
  std::mt19937 rng(kSeed);
  std::normal_distribution<float> background_dist(
      static_cast<float>(kBackgroundMean),
      static_cast<float>(kBackgroundSigma));
  std::uniform_real_distribution<double> x_dist(0.0, static_cast<double>(kWidth - 1));
  std::uniform_real_distribution<double> y_dist(0.0, static_cast<double>(kHeight - 1));
  std::uniform_real_distribution<double> flux_dist(kFluxMin, kFluxMax);
  std::uniform_real_distribution<double> rho_dist(kRhoMin, kRhoMax);

  for (float &pixel : image) {
    pixel = background_dist(rng);
  }

  for (int i = 0; i < kNumSources; ++i) {
    convsurf(image, x_dist(rng), y_dist(rng), flux_dist(rng), rho_dist(rng));
  }

  return image;
}

template <typename T>
std::vector<T> cast_vector(const std::vector<float> &source) {
  std::vector<T> out(source.size());
  for (size_t i = 0; i < source.size(); ++i) {
    out[i] = static_cast<T>(source[i]);
  }
  return out;
}

DiffStats compute_diff_stats(const std::vector<float> &lhs, const std::vector<float> &rhs) {
  DiffStats stats{};
  double sum_abs = 0.0;
  double sum_sq = 0.0;
  stats.max_abs = -1.0;
  stats.max_index = 0;

  for (size_t i = 0; i < lhs.size(); ++i) {
    const double diff = std::fabs(static_cast<double>(lhs[i]) - static_cast<double>(rhs[i]));
    sum_abs += diff;
    sum_sq += diff * diff;
    if (diff > stats.max_abs) {
      stats.max_abs = diff;
      stats.max_index = i;
    }
  }

  stats.mean_abs = sum_abs / static_cast<double>(lhs.size());
  stats.rms = std::sqrt(sum_sq / static_cast<double>(lhs.size()));
  return stats;
}

template <typename Func>
double measure_ms(Func &&func) {
  const auto start = std::chrono::steady_clock::now();
  func();
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

CompareResult compare_case(
    const sep_api *gpu_api,
    const sep_api *cpu_api,
    const char *dtype_name,
    const void *data_ptr,
    int dtype_code) {
  sep_image image{};
  image.data = data_ptr;
  image.dtype = dtype_code;
  image.w = kWidth;
  image.h = kHeight;

  sep_bkg *gpu_bkg = nullptr;
  sep_bkg *cpu_bkg = nullptr;

  CompareResult result{};
  result.dtype_name = dtype_name;

  result.gpu_background_ms = measure_ms([&]() {
    check_status(
        gpu_api,
        gpu_api->background(&image, kBw, kBh, kFw, kFh, kFthresh, &gpu_bkg),
        "gpu background");
  });

  result.cpu_background_ms = measure_ms([&]() {
    check_status(
        cpu_api,
        cpu_api->background(&image, kBw, kBh, kFw, kFh, kFthresh, &cpu_bkg),
        "cpu background");
  });

  std::vector<float> gpu_back(static_cast<size_t>(kWidth) * kHeight);
  std::vector<float> cpu_back(static_cast<size_t>(kWidth) * kHeight);
  std::vector<float> gpu_rms(static_cast<size_t>(kWidth) * kHeight);
  std::vector<float> cpu_rms(static_cast<size_t>(kWidth) * kHeight);

  result.gpu_expand_ms = measure_ms([&]() {
    check_status(gpu_api, gpu_api->bkg_array(gpu_bkg, gpu_back.data(), SEP_TFLOAT), "gpu back");
    check_status(gpu_api, gpu_api->rms_array(gpu_bkg, gpu_rms.data(), SEP_TFLOAT), "gpu rms");
  });

  result.cpu_expand_ms = measure_ms([&]() {
    check_status(cpu_api, cpu_api->bkg_array(cpu_bkg, cpu_back.data(), SEP_TFLOAT), "cpu back");
    check_status(cpu_api, cpu_api->rms_array(cpu_bkg, cpu_rms.data(), SEP_TFLOAT), "cpu rms");
  });

  result.gpu_global = gpu_api->global(gpu_bkg);
  result.cpu_global = cpu_api->global(cpu_bkg);
  result.gpu_globalrms = gpu_api->globalrms(gpu_bkg);
  result.cpu_globalrms = cpu_api->globalrms(cpu_bkg);
  result.back_stats = compute_diff_stats(gpu_back, cpu_back);
  result.rms_stats = compute_diff_stats(gpu_rms, cpu_rms);

  gpu_api->bkg_free(gpu_bkg);
  cpu_api->bkg_free(cpu_bkg);
  return result;
}

void print_result(const CompareResult &result) {
  const size_t back_y = result.back_stats.max_index / kWidth;
  const size_t back_x = result.back_stats.max_index % kWidth;
  const size_t rms_y = result.rms_stats.max_index / kWidth;
  const size_t rms_x = result.rms_stats.max_index % kWidth;

  std::printf(
      "[%s]\n"
      "  gpu background time: %.2f ms\n"
      "  cpu background time: %.2f ms\n"
      "  gpu expand time: %.2f ms\n"
      "  cpu expand time: %.2f ms\n"
      "  global background: gpu=%.6f cpu=%.6f diff=%.6f\n"
      "  global rms: gpu=%.6f cpu=%.6f diff=%.6f\n"
      "  background diff: mean_abs=%.6f rms=%.6f max_abs=%.6f at (%zu,%zu)\n"
      "  rms diff: mean_abs=%.6f rms=%.6f max_abs=%.6f at (%zu,%zu)\n",
      result.dtype_name.c_str(),
      result.gpu_background_ms,
      result.cpu_background_ms,
      result.gpu_expand_ms,
      result.cpu_expand_ms,
      result.gpu_global,
      result.cpu_global,
      std::fabs(result.gpu_global - result.cpu_global),
      result.gpu_globalrms,
      result.cpu_globalrms,
      std::fabs(result.gpu_globalrms - result.cpu_globalrms),
      result.back_stats.mean_abs,
      result.back_stats.rms,
      result.back_stats.max_abs,
      back_x,
      back_y,
      result.rms_stats.mean_abs,
      result.rms_stats.rms,
      result.rms_stats.max_abs,
      rms_x,
      rms_y);
}

void validate_result(const CompareResult &result) {
  if (std::fabs(result.gpu_global - result.cpu_global) > kGlobalTolerance) {
    std::fprintf(
        stderr,
        "[%s] global background mismatch exceeds tolerance: %.6f\n",
        result.dtype_name.c_str(),
        std::fabs(result.gpu_global - result.cpu_global));
    std::exit(1);
  }
  if (std::fabs(result.gpu_globalrms - result.cpu_globalrms) > kGlobalTolerance) {
    std::fprintf(
        stderr,
        "[%s] global rms mismatch exceeds tolerance: %.6f\n",
        result.dtype_name.c_str(),
        std::fabs(result.gpu_globalrms - result.cpu_globalrms));
    std::exit(1);
  }
  if (result.back_stats.max_abs > kMapMaxTolerance || result.back_stats.mean_abs > kMapMeanTolerance) {
    std::fprintf(
        stderr,
        "[%s] background map diff exceeds tolerance: mean_abs=%.6f max_abs=%.6f\n",
        result.dtype_name.c_str(),
        result.back_stats.mean_abs,
        result.back_stats.max_abs);
    std::exit(1);
  }
  if (result.rms_stats.max_abs > kMapMaxTolerance || result.rms_stats.mean_abs > kMapMeanTolerance) {
    std::fprintf(
        stderr,
        "[%s] rms map diff exceeds tolerance: mean_abs=%.6f max_abs=%.6f\n",
        result.dtype_name.c_str(),
        result.rms_stats.mean_abs,
        result.rms_stats.max_abs);
    std::exit(1);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <lib_sep_cuda.so> <libsep.so>\n", argv[0]);
    return 2;
  }

  const sep_api gpu_api = load_api(argv[1]);
  const sep_api cpu_api = load_api(argv[2]);

  std::printf(
      "Generating test scene: %dx%d, gaussian background(mean=%.1f, sigma=%.1f), "
      "N=%d point sources, seed=%u, bw=%lld bh=%lld fw=%lld fh=%lld\n",
      kWidth,
      kHeight,
      kBackgroundMean,
      kBackgroundSigma,
      kNumSources,
      kSeed,
      static_cast<long long>(kBw),
      static_cast<long long>(kBh),
      static_cast<long long>(kFw),
      static_cast<long long>(kFh));

  const std::vector<float> float_scene = make_scene();
  const std::vector<int> int_scene = cast_vector<int>(float_scene);
  const std::vector<double> double_scene = cast_vector<double>(float_scene);

  const CompareResult float_result =
      compare_case(&gpu_api, &cpu_api, "float", float_scene.data(), SEP_TFLOAT);
  print_result(float_result);
  validate_result(float_result);

  const CompareResult int_result =
      compare_case(&gpu_api, &cpu_api, "int", int_scene.data(), SEP_TINT);
  print_result(int_result);
  validate_result(int_result);

  const CompareResult double_result =
      compare_case(&gpu_api, &cpu_api, "double", double_scene.data(), SEP_TDOUBLE);
  print_result(double_result);
  validate_result(double_result);

  dlclose(gpu_api.handle);
  dlclose(cpu_api.handle);
  return 0;
}
