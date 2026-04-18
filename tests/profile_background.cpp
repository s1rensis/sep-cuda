#include <dlfcn.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "sep_cuda.h"
#include "sep_internal.h"

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
using sep_profile_get_last_background_fn = int (*)(sep_cuda_background_profile *);
using sep_profile_reset_runtime_state_fn = void (*)();

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
  sep_profile_get_last_background_fn get_last_background_profile;
  sep_profile_reset_runtime_state_fn reset_runtime_state;
};

struct DiffStats {
  double mean_abs;
  double rms;
  double max_abs;
  size_t max_index;
};

struct CompareResult {
  std::string dtype_name;
  int run_index;
  bool warmup;
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
  sep_cuda_background_profile profile;
  int has_profile;
};

struct StatsSummary {
  double mean;
  double p50;
  double p90;
  double min;
  double max;
};

void load_symbol(void **out, void *handle, const char *name, bool required = true) {
  *out = dlsym(handle, name);
  if (required && *out == nullptr) {
    std::fprintf(stderr, "failed to resolve symbol %s: %s\n", name, dlerror());
    std::exit(1);
  }
}

sep_api load_api(const char *path, bool is_gpu_library) {
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
  if (is_gpu_library) {
    load_symbol(
        reinterpret_cast<void **>(&api.get_last_background_profile),
        api.handle,
        "sep_cuda_profile_get_last_background",
        false);
    load_symbol(
        reinterpret_cast<void **>(&api.reset_runtime_state),
        api.handle,
        "sepcuda_profile_reset_runtime_state",
        false);
  }
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

template <typename T>
CompareResult compare_case(
    const sep_api *gpu_api,
    const sep_api *cpu_api,
    const char *dtype_name,
    const std::vector<T> &data,
    int dtype_code,
    int run_index,
    bool warmup) {
  sep_image image{};
  image.data = data.data();
  image.dtype = dtype_code;
  image.w = kWidth;
  image.h = kHeight;

  sep_bkg *gpu_bkg = nullptr;
  sep_bkg *cpu_bkg = nullptr;

  CompareResult result{};
  result.dtype_name = dtype_name;
  result.run_index = run_index;
  result.warmup = warmup;
  memset(&result.profile, 0, sizeof(result.profile));

  result.gpu_background_ms = measure_ms([&]() {
    check_status(
        gpu_api,
        gpu_api->background(&image, kBw, kBh, kFw, kFh, kFthresh, &gpu_bkg),
        "gpu background");
  });

  if (gpu_api->get_last_background_profile != nullptr) {
    result.has_profile = gpu_api->get_last_background_profile(&result.profile);
  }

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

void reset_cuda_device(const sep_api *gpu_api) {
  cudaError_t err = cudaDeviceReset();
  if (err == cudaErrorNoDevice) {
    return;
  }
  if (err != cudaSuccess) {
    std::fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(err));
  }
  if (gpu_api->reset_runtime_state != nullptr) {
    gpu_api->reset_runtime_state();
  }
}

StatsSummary summarize(const std::vector<double> &values) {
  StatsSummary summary{};
  if (values.empty()) {
    return summary;
  }

  std::vector<double> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  double sum = 0.0;
  for (double value : sorted) {
    sum += value;
  }

  auto percentile = [&](double p) {
    if (sorted.size() == 1) {
      return sorted.front();
    }
    const double pos = p * static_cast<double>(sorted.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(pos));
    const size_t hi = static_cast<size_t>(std::ceil(pos));
    const double t = pos - static_cast<double>(lo);
    return sorted[lo] * (1.0 - t) + sorted[hi] * t;
  };

  summary.mean = sum / static_cast<double>(sorted.size());
  summary.p50 = percentile(0.50);
  summary.p90 = percentile(0.90);
  summary.min = sorted.front();
  summary.max = sorted.back();
  return summary;
}

void print_run_line(const CompareResult &result) {
  const char *kind = result.warmup ? "warmup" : "measured";
  std::printf(
      "PROFILE_RUN dtype=%s run=%d kind=%s gpu_bg_ms=%.3f cpu_bg_ms=%.3f gpu_expand_ms=%.3f "
      "cpu_expand_ms=%.3f stage_ms=%.3f mesh_ms=%.3f filter_ms=%.3f spline_back_ms=%.3f "
      "spline_sigma_ms=%.3f runtime_init_ms=%.3f runtime_init=%d cuda_get_device_count_ms=%.3f "
      "cuda_malloc_ms=%.3f h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f cuda_free_ms=%.3f "
      "global_diff=%.6f globalrms_diff=%.6f back_mean_abs=%.6f back_max_abs=%.6f "
      "rms_mean_abs=%.6f rms_max_abs=%.6f\n",
      result.dtype_name.c_str(),
      result.run_index,
      kind,
      result.gpu_background_ms,
      result.cpu_background_ms,
      result.gpu_expand_ms,
      result.cpu_expand_ms,
      result.profile.staging_ms,
      result.profile.mesh_total_ms,
      result.profile.filter_ms,
      result.profile.spline_back_ms,
      result.profile.spline_sigma_ms,
      result.profile.runtime_init_ms,
      result.profile.runtime_init_performed,
      result.profile.cuda_get_device_count_ms,
      result.profile.cuda_malloc_ms,
      result.profile.h2d_ms,
      result.profile.kernel_ms,
      result.profile.d2h_ms,
      result.profile.cuda_free_ms,
      std::fabs(result.gpu_global - result.cpu_global),
      std::fabs(result.gpu_globalrms - result.cpu_globalrms),
      result.back_stats.mean_abs,
      result.back_stats.max_abs,
      result.rms_stats.mean_abs,
      result.rms_stats.max_abs);
}

void print_summary_block(const char *dtype_name, const CompareResult &warmup, const std::vector<CompareResult> &measured) {
  std::vector<double> gpu_bg;
  std::vector<double> cpu_bg;
  std::vector<double> gpu_expand;
  std::vector<double> cpu_expand;
  std::vector<double> staging;
  std::vector<double> mesh;
  std::vector<double> filter;
  std::vector<double> spline_back;
  std::vector<double> spline_sigma;
  std::vector<double> back_mean_abs;
  std::vector<double> rms_mean_abs;
  double total_phase_sum = 0.0;
  double staging_sum = 0.0;
  double mesh_sum = 0.0;
  double filter_sum = 0.0;
  double spline_back_sum = 0.0;
  double spline_sigma_sum = 0.0;

  for (const CompareResult &result : measured) {
    gpu_bg.push_back(result.gpu_background_ms);
    cpu_bg.push_back(result.cpu_background_ms);
    gpu_expand.push_back(result.gpu_expand_ms);
    cpu_expand.push_back(result.cpu_expand_ms);
    staging.push_back(result.profile.staging_ms);
    mesh.push_back(result.profile.mesh_total_ms);
    filter.push_back(result.profile.filter_ms);
    spline_back.push_back(result.profile.spline_back_ms);
    spline_sigma.push_back(result.profile.spline_sigma_ms);
    back_mean_abs.push_back(result.back_stats.mean_abs);
    rms_mean_abs.push_back(result.rms_stats.mean_abs);
    total_phase_sum += result.gpu_background_ms;
    staging_sum += result.profile.staging_ms;
    mesh_sum += result.profile.mesh_total_ms;
    filter_sum += result.profile.filter_ms;
    spline_back_sum += result.profile.spline_back_ms;
    spline_sigma_sum += result.profile.spline_sigma_ms;
  }

  const StatsSummary gpu_bg_stats = summarize(gpu_bg);
  const StatsSummary cpu_bg_stats = summarize(cpu_bg);
  const StatsSummary gpu_expand_stats = summarize(gpu_expand);
  const StatsSummary cpu_expand_stats = summarize(cpu_expand);
  const StatsSummary stage_stats = summarize(staging);
  const StatsSummary mesh_stats = summarize(mesh);
  const StatsSummary filter_stats = summarize(filter);
  const StatsSummary spline_back_stats = summarize(spline_back);
  const StatsSummary spline_sigma_stats = summarize(spline_sigma);
  const StatsSummary back_diff_stats = summarize(back_mean_abs);
  const StatsSummary rms_diff_stats = summarize(rms_mean_abs);

  const double inv_total_phase = total_phase_sum > 0.0 ? 100.0 / total_phase_sum : 0.0;

  std::printf(
      "PROFILE_COLD dtype=%s gpu_bg_ms=%.3f cpu_bg_ms=%.3f gpu_expand_ms=%.3f cpu_expand_ms=%.3f "
      "stage_ms=%.3f mesh_ms=%.3f filter_ms=%.3f spline_back_ms=%.3f spline_sigma_ms=%.3f "
      "runtime_init_ms=%.3f cuda_malloc_ms=%.3f h2d_ms=%.3f kernel_ms=%.3f d2h_ms=%.3f "
      "cuda_free_ms=%.3f\n",
      dtype_name,
      warmup.gpu_background_ms,
      warmup.cpu_background_ms,
      warmup.gpu_expand_ms,
      warmup.cpu_expand_ms,
      warmup.profile.staging_ms,
      warmup.profile.mesh_total_ms,
      warmup.profile.filter_ms,
      warmup.profile.spline_back_ms,
      warmup.profile.spline_sigma_ms,
      warmup.profile.runtime_init_ms,
      warmup.profile.cuda_malloc_ms,
      warmup.profile.h2d_ms,
      warmup.profile.kernel_ms,
      warmup.profile.d2h_ms,
      warmup.profile.cuda_free_ms);

  std::printf(
      "PROFILE_SUMMARY dtype=%s measured_runs=%zu gpu_bg_mean_ms=%.3f gpu_bg_p50_ms=%.3f "
      "gpu_bg_p90_ms=%.3f gpu_bg_min_ms=%.3f gpu_bg_max_ms=%.3f cpu_bg_mean_ms=%.3f "
      "gpu_expand_mean_ms=%.3f gpu_expand_p50_ms=%.3f gpu_expand_p90_ms=%.3f "
      "cpu_expand_mean_ms=%.3f stage_mean_ms=%.3f mesh_mean_ms=%.3f filter_mean_ms=%.3f "
      "spline_back_mean_ms=%.3f spline_sigma_mean_ms=%.3f stage_pct=%.2f mesh_pct=%.2f "
      "filter_pct=%.2f spline_back_pct=%.2f spline_sigma_pct=%.2f back_mean_abs=%.6f "
      "rms_mean_abs=%.6f\n",
      dtype_name,
      measured.size(),
      gpu_bg_stats.mean,
      gpu_bg_stats.p50,
      gpu_bg_stats.p90,
      gpu_bg_stats.min,
      gpu_bg_stats.max,
      cpu_bg_stats.mean,
      gpu_expand_stats.mean,
      gpu_expand_stats.p50,
      gpu_expand_stats.p90,
      cpu_expand_stats.mean,
      stage_stats.mean,
      mesh_stats.mean,
      filter_stats.mean,
      spline_back_stats.mean,
      spline_sigma_stats.mean,
      staging_sum * inv_total_phase,
      mesh_sum * inv_total_phase,
      filter_sum * inv_total_phase,
      spline_back_sum * inv_total_phase,
      spline_sigma_sum * inv_total_phase,
      back_diff_stats.mean,
      rms_diff_stats.mean);
}

int parse_int_arg(char **begin, char **end, const char *flag, int default_value) {
  for (char **it = begin; it != end; ++it) {
    if (std::strcmp(*it, flag) == 0 && (it + 1) != end) {
      return std::atoi(*(it + 1));
    }
  }
  return default_value;
}

std::string parse_string_arg(char **begin, char **end, const char *flag, const char *default_value) {
  for (char **it = begin; it != end; ++it) {
    if (std::strcmp(*it, flag) == 0 && (it + 1) != end) {
      return *(it + 1);
    }
  }
  return default_value;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 3) {
    std::fprintf(
        stderr,
        "usage: %s <lib_sep_cuda.so> <libsep.so> [--warmup N] [--runs N] [--dtype all|float|int|double]\n",
        argv[0]);
    return 2;
  }

  const int warmup_runs = parse_int_arg(argv + 3, argv + argc, "--warmup", 1);
  const int measured_runs = parse_int_arg(argv + 3, argv + argc, "--runs", 10);
  const std::string dtype_filter = parse_string_arg(argv + 3, argv + argc, "--dtype", "all");

  if (getenv("SEPCUDA_PROFILE") == nullptr) {
    std::fprintf(stderr, "warning: SEPCUDA_PROFILE is not set; detailed phase profiling will be zeroed\n");
  }

  const sep_api gpu_api = load_api(argv[1], true);
  const sep_api cpu_api = load_api(argv[2], false);

  std::printf(
      "PROFILE_SCENE width=%d height=%d background_mean=%.1f background_sigma=%.1f "
      "num_sources=%d seed=%u bw=%lld bh=%lld fw=%lld fh=%lld fthresh=%.1f warmup=%d runs=%d\n",
      kWidth,
      kHeight,
      kBackgroundMean,
      kBackgroundSigma,
      kNumSources,
      kSeed,
      static_cast<long long>(kBw),
      static_cast<long long>(kBh),
      static_cast<long long>(kFw),
      static_cast<long long>(kFh),
      kFthresh,
      warmup_runs,
      measured_runs);

  const std::vector<float> float_scene = make_scene();
  const std::vector<int> int_scene = cast_vector<int>(float_scene);
  const std::vector<double> double_scene = cast_vector<double>(float_scene);

  auto run_dtype = [&](const char *dtype_name, const auto &scene, int dtype_code) {
    if (dtype_filter != "all" && dtype_filter != dtype_name) {
      return;
    }

    reset_cuda_device(&gpu_api);

    CompareResult warmup_result{};
    bool have_warmup = false;
    std::vector<CompareResult> measured;
    measured.reserve(static_cast<size_t>(measured_runs));

    for (int run = 0; run < warmup_runs + measured_runs; ++run) {
      const bool warmup = run < warmup_runs;
      const CompareResult result =
          compare_case(&gpu_api, &cpu_api, dtype_name, scene, dtype_code, run, warmup);
      validate_result(result);
      print_run_line(result);
      if (warmup && !have_warmup) {
        warmup_result = result;
        have_warmup = true;
      }
      if (!warmup) {
        measured.push_back(result);
      }
    }

    if (!have_warmup && !measured.empty()) {
      warmup_result = measured.front();
    }
    print_summary_block(dtype_name, warmup_result, measured);
  };

  run_dtype("float", float_scene, SEP_TFLOAT);
  run_dtype("int", int_scene, SEP_TINT);
  run_dtype("double", double_scene, SEP_TDOUBLE);

  dlclose(gpu_api.handle);
  dlclose(cpu_api.handle);
  return 0;
}
