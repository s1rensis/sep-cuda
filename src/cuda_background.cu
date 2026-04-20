#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <atomic>

#include "sep_cuda.h"
#include "sep_internal.h"
#include "sepcore.h"

namespace {

constexpr int kThreadsPerBlock = 256;
constexpr int kMaxLevels = 4096;
constexpr float kBackMinGoodFrac = 0.5f;
constexpr float kQuantifNSigma = 5.0f;
constexpr float kQuantifAMin = 4.0f;
constexpr float kEps = 1.0e-4f;
constexpr float kBigFloat = 1.0e30f;
std::atomic<bool> g_runtime_initialized{false};

struct MeshWorkspace {
  float *d_image = nullptr;
  float *d_mask = nullptr;
  float *d_back = nullptr;
  float *d_sigma = nullptr;
  float *d_coeff_back = nullptr;
  float *d_coeff_dback = nullptr;
  float *d_coeff_sigma = nullptr;
  float *d_coeff_dsigma = nullptr;
  uint16_t *d_src_u16 = nullptr;
  uint16_t *d_dst_sub_u16 = nullptr;
  uint16_t *d_dst_rms_u16 = nullptr;
  size_t image_capacity_bytes = 0;
  size_t mask_capacity_bytes = 0;
  size_t back_capacity_bytes = 0;
  size_t sigma_capacity_bytes = 0;
  size_t coeff_back_capacity_bytes = 0;
  size_t coeff_dback_capacity_bytes = 0;
  size_t coeff_sigma_capacity_bytes = 0;
  size_t coeff_dsigma_capacity_bytes = 0;
  size_t src_u16_capacity_bytes = 0;
  size_t dst_sub_u16_capacity_bytes = 0;
  size_t dst_rms_u16_capacity_bytes = 0;
};

thread_local MeshWorkspace g_mesh_workspace{};

double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

inline int set_cuda_error(cudaError_t err, const char *context) {
  char detail[256];
  std::snprintf(
      detail,
      sizeof(detail),
      "%s failed: %s",
      context,
      cudaGetErrorString(err));
  put_errdetail(detail);
  return (err == cudaErrorNoDevice) ? SEP_CUDA_UNAVAILABLE : SEP_CUDA_RUNTIME_ERROR;
}

template <typename T>
inline int ensure_device_buffer(
    T **buffer,
    size_t *capacity_bytes,
    size_t required_bytes,
    const char *label,
    sep_cuda_background_profile *profile) {
  const int profile_enabled = profile != nullptr && profile->enabled;
  double phase_start_ms = 0.0;
  T *replacement = nullptr;
  cudaError_t err;

  if (required_bytes == 0 || (*buffer != nullptr && *capacity_bytes >= required_bytes)) {
    return RETURN_OK;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  err = cudaMalloc(reinterpret_cast<void **>(&replacement), required_bytes);
  if (profile_enabled) {
    profile->cuda_malloc_ms += now_ms() - phase_start_ms;
  }
  if (err != cudaSuccess) {
    char context[128];
    std::snprintf(context, sizeof(context), "cudaMalloc(%s)", label);
    return set_cuda_error(err, context);
  }

  if (*buffer != nullptr) {
    phase_start_ms = profile_enabled ? now_ms() : 0.0;
    err = cudaFree(*buffer);
    if (profile_enabled) {
      profile->cuda_free_ms += now_ms() - phase_start_ms;
    }
    if (err != cudaSuccess) {
      cudaFree(replacement);
      char context[128];
      std::snprintf(context, sizeof(context), "cudaFree(%s)", label);
      return set_cuda_error(err, context);
    }
  }

  *buffer = replacement;
  *capacity_bytes = required_bytes;
  return RETURN_OK;
}

__device__ void prepare_row_spline(
    const float *values,
    const float *dvalues,
    int64_t nx,
    int64_t ny,
    int64_t bh,
    int64_t y,
    float *node,
    float *dnode,
    float *u) {
  if (ny > 1) {
    float dy = static_cast<float>(y) / static_cast<float>(bh) - 0.5f;
    int64_t yl = static_cast<int64_t>(dy);
    dy -= static_cast<float>(yl);
    if (yl < 0) {
      yl = 0;
      dy -= 1.0f;
    } else if (yl >= ny - 1) {
      yl = ny < 2 ? 0 : ny - 2;
      dy += 1.0f;
    }

    const float cdy = 1.0f - dy;
    const float dy3 = dy * dy * dy - dy;
    const float cdy3 = cdy * cdy * cdy - cdy;
    const int64_t ystep = nx * yl;
    const float *blo = values + ystep;
    const float *bhi = blo + nx;
    const float *dblo = dvalues + ystep;
    const float *dbhi = dblo + nx;

    for (int64_t x = 0; x < nx; ++x) {
      node[x] = cdy * blo[x] + dy * bhi[x] + cdy3 * dblo[x] + dy3 * dbhi[x];
    }

    if (nx > 1) {
      dnode[0] = 0.0f;
      u[0] = 0.0f;
      for (int64_t x = 1; x < nx - 1; ++x) {
        float temp = -1.0f / (dnode[x - 1] + 4.0f);
        dnode[x] = temp;
        temp *= u[x - 1] - 6.0f * (node[x + 1] + node[x - 1] - 2.0f * node[x]);
        u[x] = temp;
      }
      dnode[nx - 1] = 0.0f;
      for (int64_t x = nx - 2; x > 0; --x) {
        dnode[x] = (dnode[x] * dnode[x + 1] + u[x]) / 6.0f;
      }
    } else {
      dnode[0] = 0.0f;
    }
    return;
  }

  for (int64_t x = 0; x < nx; ++x) {
    node[x] = values[x];
    dnode[x] = dvalues[x];
  }
}

__device__ float evaluate_row_spline(
    const float *node, const float *dnode, int64_t nx, int64_t bw, int64_t x) {
  if (nx <= 1) {
    return node[0];
  }

  float dx = (static_cast<float>(x) + 0.5f) / static_cast<float>(bw) - 0.5f;
  int64_t xl = static_cast<int64_t>(dx);
  dx -= static_cast<float>(xl);

  if (xl < 0) {
    xl = 0;
    dx -= 1.0f;
  } else if (xl >= nx - 1) {
    xl = nx < 2 ? 0 : nx - 2;
    dx += 1.0f;
  }

  const float cdx = 1.0f - dx;
  return cdx * (node[xl] + (cdx * cdx - 1.0f) * dnode[xl]) +
         dx * (node[xl + 1] + (dx * dx - 1.0f) * dnode[xl + 1]);
}

__device__ float backguess_device(
    const int *histo, int nlevels, float sigma_in, float qscale, float qzero, float *sigma_out) {
  unsigned long long lowsum, highsum, sum;
  double ftemp, mea, sig, sig1, med, dpix;
  int i, n, lcut, hcut, nlevelsm1, pix;
  const int *hilow, *hihigh, *histot;

  hcut = nlevelsm1 = nlevels - 1;
  lcut = 0;
  sig = 10.0 * nlevelsm1;
  sig1 = 1.0;
  mea = med = 0.0;

  for (n = 100; n-- && (sig >= 0.1) && (fabs(sig / sig1 - 1.0) > kEps);) {
    sig1 = sig;
    sum = 0;
    mea = 0.0;
    sig = 0.0;
    lowsum = 0;
    highsum = 0;
    histot = hilow = histo + lcut;
    hihigh = histo + hcut;

    for (i = lcut; i <= hcut; i++) {
      if (lowsum < highsum) {
        lowsum += static_cast<unsigned long long>(*(hilow++));
      } else {
        highsum += static_cast<unsigned long long>(*(hihigh--));
      }
      sum += static_cast<unsigned long long>(pix = *(histot++));
      mea += (dpix = static_cast<double>(pix) * i);
      sig += dpix * i;
    }

    med = hihigh >= histo
              ? ((hihigh - histo) + 0.5
                 + (static_cast<double>(highsum) - static_cast<double>(lowsum))
                       / (2.0 * (*hilow > *hihigh ? *hilow : *hihigh)))
              : 0.0;
    if (sum) {
      mea /= static_cast<double>(sum);
      sig = sig / static_cast<double>(sum) - mea * mea;
    }

    sig = sig > 0.0 ? sqrt(sig) : 0.0;
    lcut = (ftemp = med - 3.0 * sig) > 0.0 ? static_cast<int>(ftemp + 0.5) : 0;
    hcut = (ftemp = med + 3.0 * sig) < nlevelsm1 ? static_cast<int>(ftemp + 0.5) : nlevelsm1;
  }

  *sigma_out = static_cast<float>(sig * qscale);
  if (fabs(sig) > 0.0) {
    if (fabs(sigma_in / (*sigma_out) - 1.0f) < 0.0f) {
      return qzero + static_cast<float>(mea * qscale);
    }
    if (fabs((mea - med) / sig) < 0.3) {
      return qzero + static_cast<float>((2.5 * med - 1.5 * mea) * qscale);
    }
    return qzero + static_cast<float>(med * qscale);
  }

  return qzero + static_cast<float>(mea * qscale);
}

__global__ void compute_mesh_kernel(
    const float *image,
    const float *mask,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t nx,
    int64_t ny,
    float maskthresh,
    float *back,
    float *sigma) {
  __shared__ double s_sum[kThreadsPerBlock];
  __shared__ double s_sumsq[kThreadsPerBlock];
  __shared__ int s_count[kThreadsPerBlock];
  __shared__ int s_hist[kMaxLevels];
  __shared__ float s_lcut;
  __shared__ float s_hcut;
  __shared__ float s_qscale;
  __shared__ float s_qzero;
  __shared__ float s_sigma_input;
  __shared__ int s_valid_count;
  __shared__ int s_nlevels;
  __shared__ int s_bad_tile;

  const int tile_x = blockIdx.x;
  const int tile_y = blockIdx.y;
  const int tile_index = tile_y * static_cast<int>(nx) + tile_x;
  const int local_tid = threadIdx.x;

  if (tile_x >= nx || tile_y >= ny) {
    return;
  }

  const int64_t x0 = static_cast<int64_t>(tile_x) * bw;
  const int64_t y0 = static_cast<int64_t>(tile_y) * bh;
  const int tile_w = static_cast<int>((bw < (width - x0)) ? bw : (width - x0));
  const int tile_h = static_cast<int>((bh < (height - y0)) ? bh : (height - y0));
  const int tile_pixels = tile_w * tile_h;
  const float step = sqrtf(2.0f / static_cast<float>(M_PI)) * kQuantifNSigma / kQuantifAMin;

  double sum = 0.0;
  double sumsq = 0.0;
  int count = 0;

  for (int i = local_tid; i < tile_pixels; i += blockDim.x) {
    const int local_x = i % tile_w;
    const int local_y = i / tile_w;
    const int64_t index = (y0 + local_y) * width + (x0 + local_x);
    const float pixel = image[index];
    const float mask_value = mask ? mask[index] : 0.0f;
    const bool good = (!mask || mask_value <= maskthresh) && pixel > -kBigFloat && isfinite(pixel);
    if (good) {
      sum += pixel;
      sumsq += pixel * pixel;
      count++;
    }
  }

  s_sum[local_tid] = sum;
  s_sumsq[local_tid] = sumsq;
  s_count[local_tid] = count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (local_tid < stride) {
      s_sum[local_tid] += s_sum[local_tid + stride];
      s_sumsq[local_tid] += s_sumsq[local_tid + stride];
      s_count[local_tid] += s_count[local_tid + stride];
    }
    __syncthreads();
  }

  if (local_tid == 0) {
    s_valid_count = s_count[0];
    s_bad_tile = (static_cast<float>(s_valid_count) < static_cast<float>(tile_pixels) * kBackMinGoodFrac);

    if (s_bad_tile) {
      back[tile_index] = -kBigFloat;
      sigma[tile_index] = -kBigFloat;
      s_lcut = 0.0f;
      s_hcut = 0.0f;
      s_qscale = 1.0f;
      s_qzero = 0.0f;
      s_sigma_input = 0.0f;
      s_nlevels = 1;
    } else {
      const double mean = s_sum[0] / static_cast<double>(s_valid_count);
      const double sig = s_sumsq[0] / static_cast<double>(s_valid_count) - mean * mean;
      const double sigma1 = sig > 0.0 ? sqrt(sig) : 0.0;

      s_lcut = static_cast<float>(mean - 2.0 * sigma1);
      s_hcut = static_cast<float>(mean + 2.0 * sigma1);
    }
  }
  __syncthreads();

  if (s_bad_tile) {
    return;
  }

  sum = 0.0;
  sumsq = 0.0;
  count = 0;
  for (int i = local_tid; i < tile_pixels; i += blockDim.x) {
    const int local_x = i % tile_w;
    const int local_y = i / tile_w;
    const int64_t index = (y0 + local_y) * width + (x0 + local_x);
    const float pixel = image[index];
    const float mask_value = mask ? mask[index] : 0.0f;
    const bool good = (!mask || mask_value <= maskthresh) && pixel > -kBigFloat && isfinite(pixel);
    if (good && pixel >= s_lcut && pixel <= s_hcut) {
      sum += pixel;
      sumsq += pixel * pixel;
      count++;
    }
  }

  s_sum[local_tid] = sum;
  s_sumsq[local_tid] = sumsq;
  s_count[local_tid] = count;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (local_tid < stride) {
      s_sum[local_tid] += s_sum[local_tid + stride];
      s_sumsq[local_tid] += s_sumsq[local_tid + stride];
      s_count[local_tid] += s_count[local_tid + stride];
    }
    __syncthreads();
  }

  if (local_tid == 0) {
    const double mean = s_sum[0] / static_cast<double>(s_count[0]);
    const double sig = s_sumsq[0] / static_cast<double>(s_count[0]) - mean * mean;
    const float sigma2 = static_cast<float>(sig > 0.0 ? sqrt(sig) : 0.0);

    s_valid_count = s_count[0];
    s_sigma_input = sigma2;
    s_nlevels = static_cast<int>(step * s_valid_count + 1.0f);
    if (s_nlevels > kMaxLevels) {
      s_nlevels = kMaxLevels;
    }
    if (s_nlevels < 1) {
      s_nlevels = 1;
    }
    s_qscale = sigma2 > 0.0f ? 2.0f * kQuantifNSigma * sigma2 / static_cast<float>(s_nlevels) : 1.0f;
    s_qzero = static_cast<float>(mean) - kQuantifNSigma * sigma2;
  }
  __syncthreads();

  for (int i = local_tid; i < kMaxLevels; i += blockDim.x) {
    s_hist[i] = 0;
  }
  __syncthreads();

  const float cste = 0.499999f - s_qzero / s_qscale;
  for (int i = local_tid; i < tile_pixels; i += blockDim.x) {
    const int local_x = i % tile_w;
    const int local_y = i / tile_w;
    const int64_t index = (y0 + local_y) * width + (x0 + local_x);
    const float pixel = image[index];
    const float mask_value = mask ? mask[index] : 0.0f;
    const bool good = (!mask || mask_value <= maskthresh) && pixel > -kBigFloat && isfinite(pixel);
    if (good) {
      const int bin = static_cast<int>(pixel / s_qscale + cste);
      if (bin >= 0 && bin < s_nlevels) {
        atomicAdd(&s_hist[bin], 1);
      }
    }
  }
  __syncthreads();

  if (local_tid == 0) {
    float out_sigma = 0.0f;
    back[tile_index] = backguess_device(
        s_hist, s_nlevels, s_sigma_input, s_qscale, s_qzero, &out_sigma);
    sigma[tile_index] = out_sigma;
  }
}

__global__ void subtract_background_u16_kernel(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t nx,
    int64_t ny,
    const float *back,
    const float *dback,
    const float *sigma,
    const float *dsigma,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  extern __shared__ float shared[];
  float *back_node = shared;
  float *back_dnode = back_node + nx;
  float *back_u = back_dnode + nx;
  float *sigma_node = back_u + nx;
  float *sigma_dnode = sigma_node + nx;
  float *sigma_u = sigma_dnode + nx;

  const int64_t y = static_cast<int64_t>(blockIdx.x);
  const int64_t tid = static_cast<int64_t>(threadIdx.x);

  if (y >= height) {
    return;
  }

  if (tid == 0) {
    prepare_row_spline(back, dback, nx, ny, bh, y, back_node, back_dnode, back_u);
    prepare_row_spline(sigma, dsigma, nx, ny, bh, y, sigma_node, sigma_dnode, sigma_u);
  }
  __syncthreads();

  const int64_t row_offset = y * width;
  for (int64_t x = tid; x < width; x += blockDim.x) {
    const float background = evaluate_row_spline(back_node, back_dnode, nx, bw, x);
    const float rms = evaluate_row_spline(sigma_node, sigma_dnode, nx, bw, x);
    const int background_i = static_cast<int>(background + 0.5f);
    const int rms_i = static_cast<int>(rms + 0.5f);
    const int corrected = static_cast<int>(src[row_offset + x]) - background_i;

    dst_subtracted[row_offset + x] = static_cast<uint16_t>(corrected > 0 ? corrected : 0);
    dst_rms[row_offset + x] = static_cast<uint16_t>(rms_i > 0 ? rms_i : 0);
  }
}

__global__ void subtract_background_only_u16_kernel(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t nx,
    int64_t ny,
    const float *back,
    const float *dback,
    uint16_t *dst_subtracted) {
  extern __shared__ float shared[];
  float *back_node = shared;
  float *back_dnode = back_node + nx;
  float *back_u = back_dnode + nx;

  const int64_t y = static_cast<int64_t>(blockIdx.x);
  const int64_t tid = static_cast<int64_t>(threadIdx.x);

  if (y >= height) {
    return;
  }

  if (tid == 0) {
    prepare_row_spline(back, dback, nx, ny, bh, y, back_node, back_dnode, back_u);
  }
  __syncthreads();

  const int64_t row_offset = y * width;
  for (int64_t x = tid; x < width; x += blockDim.x) {
    const float background = evaluate_row_spline(back_node, back_dnode, nx, bw, x);
    const int background_i = static_cast<int>(background + 0.5f);
    const int corrected = static_cast<int>(src[row_offset + x]) - background_i;

    dst_subtracted[row_offset + x] = static_cast<uint16_t>(corrected > 0 ? corrected : 0);
  }
}

}  // namespace

extern "C" void sepcuda_profile_reset_runtime_state(void) {
  g_runtime_initialized.store(false, std::memory_order_release);
  g_mesh_workspace = MeshWorkspace{};
}

extern "C" int sep_cuda_compute_meshes(
    const float *image,
    const float *mask,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    float maskthresh,
    float *back,
    float *sigma,
    sep_cuda_background_profile *profile) {
  int status;
  cudaError_t err;
  int device_count;
  int64_t nx, ny, mesh_count;
  size_t image_bytes;
  size_t mesh_bytes;
  double phase_start_ms;
  const int profile_enabled = profile != NULL && profile->enabled;
  MeshWorkspace &workspace = g_mesh_workspace;

  if (profile_enabled && !g_runtime_initialized.exchange(true, std::memory_order_acq_rel)) {
    phase_start_ms = now_ms();
    err = cudaFree(nullptr);
    profile->runtime_init_ms = now_ms() - phase_start_ms;
    profile->runtime_init_performed = 1;
    if (err != cudaSuccess) {
      return set_cuda_error(err, "cudaFree(nullptr)");
    }
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  err = cudaGetDeviceCount(&device_count);
  if (profile_enabled) {
    profile->cuda_get_device_count_ms = now_ms() - phase_start_ms;
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaGetDeviceCount");
  }
  if (device_count <= 0) {
    put_errdetail("no CUDA device available");
    return SEP_CUDA_UNAVAILABLE;
  }

  nx = (width - 1) / bw + 1;
  ny = (height - 1) / bh + 1;
  mesh_count = nx * ny;
  image_bytes = static_cast<size_t>(width * height) * sizeof(float);
  mesh_bytes = static_cast<size_t>(mesh_count) * sizeof(float);

  status = ensure_device_buffer(
      &workspace.d_image, &workspace.image_capacity_bytes, image_bytes, "image", profile);
  if (status != RETURN_OK) {
    return status;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  err = cudaMemcpy(workspace.d_image, image, image_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    if (profile_enabled) {
      profile->h2d_ms += now_ms() - phase_start_ms;
    }
    return set_cuda_error(err, "cudaMemcpy(image)");
  }
  if (profile_enabled) {
    profile->h2d_ms += now_ms() - phase_start_ms;
  }

  if (mask != nullptr) {
    status = ensure_device_buffer(
        &workspace.d_mask, &workspace.mask_capacity_bytes, image_bytes, "mask", profile);
    if (status != RETURN_OK) {
      return status;
    }
    phase_start_ms = profile_enabled ? now_ms() : 0.0;
    err = cudaMemcpy(workspace.d_mask, mask, image_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      if (profile_enabled) {
        profile->h2d_ms += now_ms() - phase_start_ms;
      }
      return set_cuda_error(err, "cudaMemcpy(mask)");
    }
    if (profile_enabled) {
      profile->h2d_ms += now_ms() - phase_start_ms;
    }
  }

  status = ensure_device_buffer(
      &workspace.d_back, &workspace.back_capacity_bytes, mesh_bytes, "back", profile);
  if (status != RETURN_OK) {
    return status;
  }

  status = ensure_device_buffer(
      &workspace.d_sigma, &workspace.sigma_capacity_bytes, mesh_bytes, "sigma", profile);
  if (status != RETURN_OK) {
    return status;
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  compute_mesh_kernel<<<dim3(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny)), kThreadsPerBlock>>>(
      workspace.d_image,
      mask ? workspace.d_mask : nullptr,
      width,
      height,
      bw,
      bh,
      nx,
      ny,
      maskthresh,
      workspace.d_back,
      workspace.d_sigma);

  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (profile_enabled) {
    profile->kernel_ms += now_ms() - phase_start_ms;
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "compute_mesh_kernel");
  }

  phase_start_ms = profile_enabled ? now_ms() : 0.0;
  err = cudaMemcpy(back, workspace.d_back, mesh_bytes, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(sigma, workspace.d_sigma, mesh_bytes, cudaMemcpyDeviceToHost);
  }
  if (profile_enabled) {
    profile->d2h_ms += now_ms() - phase_start_ms;
  }

  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(results)");
  }

  return RETURN_OK;
}

extern "C" int sep_cuda_subtract_background_and_fill_rms_u16_fast(
    const sep_bkg *bkg,
    const uint16_t *src,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  cudaError_t err;
  MeshWorkspace &workspace = g_mesh_workspace;
  const int64_t width = bkg->w;
  const int64_t height = bkg->h;
  const int64_t mesh_count = bkg->n;
  const size_t image_bytes = static_cast<size_t>(width * height) * sizeof(uint16_t);
  const size_t coeff_bytes = static_cast<size_t>(mesh_count) * sizeof(float);
  const size_t shared_bytes = static_cast<size_t>(6 * bkg->nx) * sizeof(float);
  int status;

  if (bkg == nullptr || src == nullptr || dst_subtracted == nullptr || dst_rms == nullptr) {
    put_errdetail("u16 fast path received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = ensure_device_buffer(
      &workspace.d_src_u16,
      &workspace.src_u16_capacity_bytes,
      image_bytes,
      "src_u16",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_dst_sub_u16,
      &workspace.dst_sub_u16_capacity_bytes,
      image_bytes,
      "dst_sub_u16",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_dst_rms_u16,
      &workspace.dst_rms_u16_capacity_bytes,
      image_bytes,
      "dst_rms_u16",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_back,
      &workspace.coeff_back_capacity_bytes,
      coeff_bytes,
      "coeff_back",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_dback,
      &workspace.coeff_dback_capacity_bytes,
      coeff_bytes,
      "coeff_dback",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_sigma,
      &workspace.coeff_sigma_capacity_bytes,
      coeff_bytes,
      "coeff_sigma",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_dsigma,
      &workspace.coeff_dsigma_capacity_bytes,
      coeff_bytes,
      "coeff_dsigma",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }

  err = cudaMemcpy(workspace.d_src_u16, src, image_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(src_u16)");
  }
  err = cudaMemcpy(workspace.d_coeff_back, bkg->back, coeff_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(workspace.d_coeff_dback, bkg->dback, coeff_bytes, cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(workspace.d_coeff_sigma, bkg->sigma, coeff_bytes, cudaMemcpyHostToDevice);
  }
  if (err == cudaSuccess) {
    err = cudaMemcpy(workspace.d_coeff_dsigma, bkg->dsigma, coeff_bytes, cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(bkg_coefficients)");
  }

  subtract_background_u16_kernel<<<static_cast<unsigned int>(height), kThreadsPerBlock, shared_bytes>>>(
      workspace.d_src_u16,
      width,
      height,
      bkg->bw,
      bkg->bh,
      bkg->nx,
      bkg->ny,
      workspace.d_coeff_back,
      workspace.d_coeff_dback,
      workspace.d_coeff_sigma,
      workspace.d_coeff_dsigma,
      workspace.d_dst_sub_u16,
      workspace.d_dst_rms_u16);

  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "subtract_background_u16_kernel");
  }

  err = cudaMemcpy(dst_subtracted, workspace.d_dst_sub_u16, image_bytes, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(dst_rms, workspace.d_dst_rms_u16, image_bytes, cudaMemcpyDeviceToHost);
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(u16_outputs)");
  }

  return RETURN_OK;
}

extern "C" int sep_cuda_subtract_background_u16_fast(
    const sep_bkg *bkg,
    const uint16_t *src,
    uint16_t *dst_subtracted) {
  cudaError_t err;
  MeshWorkspace &workspace = g_mesh_workspace;
  const int64_t width = bkg->w;
  const int64_t height = bkg->h;
  const int64_t mesh_count = bkg->n;
  const size_t image_bytes = static_cast<size_t>(width * height) * sizeof(uint16_t);
  const size_t coeff_bytes = static_cast<size_t>(mesh_count) * sizeof(float);
  const size_t shared_bytes = static_cast<size_t>(3 * bkg->nx) * sizeof(float);
  int status;

  if (bkg == nullptr || src == nullptr || dst_subtracted == nullptr) {
    put_errdetail("u16 fast path received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = ensure_device_buffer(
      &workspace.d_src_u16,
      &workspace.src_u16_capacity_bytes,
      image_bytes,
      "src_u16",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_dst_sub_u16,
      &workspace.dst_sub_u16_capacity_bytes,
      image_bytes,
      "dst_sub_u16",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_back,
      &workspace.coeff_back_capacity_bytes,
      coeff_bytes,
      "coeff_back",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }
  status = ensure_device_buffer(
      &workspace.d_coeff_dback,
      &workspace.coeff_dback_capacity_bytes,
      coeff_bytes,
      "coeff_dback",
      nullptr);
  if (status != RETURN_OK) {
    return status;
  }

  err = cudaMemcpy(workspace.d_src_u16, src, image_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(src_u16)");
  }
  err = cudaMemcpy(workspace.d_coeff_back, bkg->back, coeff_bytes, cudaMemcpyHostToDevice);
  if (err == cudaSuccess) {
    err = cudaMemcpy(workspace.d_coeff_dback, bkg->dback, coeff_bytes, cudaMemcpyHostToDevice);
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(background_coefficients)");
  }

  subtract_background_only_u16_kernel<<<static_cast<unsigned int>(height), kThreadsPerBlock, shared_bytes>>>(
      workspace.d_src_u16,
      width,
      height,
      bkg->bw,
      bkg->bh,
      bkg->nx,
      bkg->ny,
      workspace.d_coeff_back,
      workspace.d_coeff_dback,
      workspace.d_dst_sub_u16);

  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) {
    return set_cuda_error(err, "subtract_background_only_u16_kernel");
  }

  err = cudaMemcpy(dst_subtracted, workspace.d_dst_sub_u16, image_bytes, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(dst_subtracted_u16)");
  }

  return RETURN_OK;
}
