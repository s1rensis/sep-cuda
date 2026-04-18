#ifndef SEPCUDA_INTERNAL_H
#define SEPCUDA_INTERNAL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int sep_cuda_compute_meshes(
    const float *image,
    const float *mask,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    float maskthresh,
    float *back,
    float *sigma);

#ifdef __cplusplus
}
#endif

#endif
