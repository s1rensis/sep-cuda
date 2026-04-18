#ifndef SEPCUDA_INTERNAL_H
#define SEPCUDA_INTERNAL_H

#include <stdint.h>

#include "sep_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  int enabled;
  int runtime_init_performed;
  double total_background_ms;
  double staging_ms;
  double mesh_total_ms;
  double filter_ms;
  double spline_back_ms;
  double spline_sigma_ms;
  double runtime_init_ms;
  double cuda_get_device_count_ms;
  double cuda_malloc_ms;
  double h2d_ms;
  double kernel_ms;
  double d2h_ms;
  double cuda_free_ms;
} sep_cuda_background_profile;

int sep_cuda_compute_meshes(
    const float *image,
    const float *mask,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    float maskthresh,
    float *back,
    float *sigma,
    sep_cuda_background_profile *profile);

int sepcuda_profile_enabled(void);
void sepcuda_profile_reset_background(sep_cuda_background_profile *profile);
void sepcuda_profile_commit_background(const sep_cuda_background_profile *profile);
void sepcuda_profile_reset_runtime_state(void);

SEP_API int sep_cuda_profile_get_last_background(sep_cuda_background_profile *profile);

SEP_API int sep_cuda_subtract_background_and_fill_rms_u16_reference(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms);

int sep_cuda_subtract_background_and_fill_rms_u16_fast(
    const sep_bkg *bkg,
    const uint16_t *src,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms);

#ifdef __cplusplus
}
#endif

#endif
