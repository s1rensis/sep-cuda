#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

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

}  // namespace

extern "C" int sep_cuda_compute_meshes(
    const float *image,
    const float *mask,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    float maskthresh,
    float *back,
    float *sigma) {
  cudaError_t err;
  int device_count;
  int64_t nx, ny, mesh_count;
  size_t image_bytes;
  size_t mesh_bytes;
  float *d_image, *d_mask, *d_back, *d_sigma;

  d_image = nullptr;
  d_mask = nullptr;
  d_back = nullptr;
  d_sigma = nullptr;

  err = cudaGetDeviceCount(&device_count);
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

  err = cudaMalloc(reinterpret_cast<void **>(&d_image), image_bytes);
  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMalloc(image)");
  }

  err = cudaMemcpy(d_image, image, image_bytes, cudaMemcpyHostToDevice);
  if (err != cudaSuccess) {
    cudaFree(d_image);
    return set_cuda_error(err, "cudaMemcpy(image)");
  }

  if (mask != nullptr) {
    err = cudaMalloc(reinterpret_cast<void **>(&d_mask), image_bytes);
    if (err != cudaSuccess) {
      cudaFree(d_image);
      return set_cuda_error(err, "cudaMalloc(mask)");
    }
    err = cudaMemcpy(d_mask, mask, image_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
      cudaFree(d_mask);
      cudaFree(d_image);
      return set_cuda_error(err, "cudaMemcpy(mask)");
    }
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_back), mesh_bytes);
  if (err != cudaSuccess) {
    cudaFree(d_mask);
    cudaFree(d_image);
    return set_cuda_error(err, "cudaMalloc(back)");
  }

  err = cudaMalloc(reinterpret_cast<void **>(&d_sigma), mesh_bytes);
  if (err != cudaSuccess) {
    cudaFree(d_back);
    cudaFree(d_mask);
    cudaFree(d_image);
    return set_cuda_error(err, "cudaMalloc(sigma)");
  }

  compute_mesh_kernel<<<dim3(static_cast<unsigned int>(nx), static_cast<unsigned int>(ny)), kThreadsPerBlock>>>(
      d_image, d_mask, width, height, bw, bh, nx, ny, maskthresh, d_back, d_sigma);

  err = cudaGetLastError();
  if (err == cudaSuccess) {
    err = cudaDeviceSynchronize();
  }
  if (err != cudaSuccess) {
    cudaFree(d_sigma);
    cudaFree(d_back);
    cudaFree(d_mask);
    cudaFree(d_image);
    return set_cuda_error(err, "compute_mesh_kernel");
  }

  err = cudaMemcpy(back, d_back, mesh_bytes, cudaMemcpyDeviceToHost);
  if (err == cudaSuccess) {
    err = cudaMemcpy(sigma, d_sigma, mesh_bytes, cudaMemcpyDeviceToHost);
  }

  cudaFree(d_sigma);
  cudaFree(d_back);
  cudaFree(d_mask);
  cudaFree(d_image);

  if (err != cudaSuccess) {
    return set_cuda_error(err, "cudaMemcpy(results)");
  }

  return RETURN_OK;
}
