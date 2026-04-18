#ifndef SEPCUDA_ADDON_H
#define SEPCUDA_ADDON_H

#include "sep_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

SEP_API int sep_cuda_subtract_background_and_fill_rms_u16(
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

#ifdef __cplusplus
}

namespace sepcuda {

inline void subtract_background_and_fill_rms_u16(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms,
    BackgroundOptions options = {}) {
  detail::throw_on_error(
      sep_cuda_subtract_background_and_fill_rms_u16(
          src,
          width,
          height,
          options.bw,
          options.bh,
          options.fw,
          options.fh,
          options.fthresh,
          dst_subtracted,
          dst_rms));
}

}  // namespace sepcuda

#endif

#endif
