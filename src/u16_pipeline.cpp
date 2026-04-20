#include <stdint.h>

#include <algorithm>
#include <vector>

#include "sep_cuda.h"
#include "sep_internal.h"
#include "sepcore.h"

namespace {

int build_background(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    sep_bkg **bkg_out) {
  sep_image image{};

  if (src == NULL || bkg_out == NULL) {
    put_errdetail("u16 pipeline received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }
  if (width <= 0 || height <= 0 || bw <= 0 || bh <= 0 || fw <= 0 || fh <= 0) {
    put_errdetail("u16 pipeline received non-positive dimensions");
    return ILLEGAL_APER_PARAMS;
  }

  image.data = src;
  image.dtype = SEP_TUSHORT;
  image.w = width;
  image.h = height;
  image.noise_type = SEP_NOISE_NONE;
  image.maskthresh = 0.0;
  image.gain = 1.0;

  return sep_background(&image, bw, bh, fw, fh, fthresh, bkg_out);
}

int run_reference_subtract_background(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted) {
  int status = RETURN_OK;
  sep_bkg *bkg = NULL;
  std::vector<int> back;
  const int64_t count = width * height;

  if (src == NULL || dst_subtracted == NULL) {
    put_errdetail("u16 pipeline received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = build_background(src, width, height, bw, bh, fw, fh, fthresh, &bkg);
  if (status != RETURN_OK) {
    return status;
  }

  back.resize(static_cast<size_t>(count));
  status = sep_bkg_array(bkg, back.data(), SEP_TINT);
  if (status != RETURN_OK) {
    sep_bkg_free(bkg);
    return status;
  }

  for (int64_t i = 0; i < count; ++i) {
    const int corrected =
        static_cast<int>(src[static_cast<size_t>(i)]) - back[static_cast<size_t>(i)];
    dst_subtracted[static_cast<size_t>(i)] =
        static_cast<uint16_t>(std::max(corrected, 0));
  }

  sep_bkg_free(bkg);
  return RETURN_OK;
}

int run_reference_subtract_background_and_fill_rms(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  int status = RETURN_OK;
  sep_bkg *bkg = NULL;
  std::vector<int> back;
  std::vector<int> rms;
  const int64_t count = width * height;

  if (src == NULL || dst_subtracted == NULL || dst_rms == NULL) {
    put_errdetail("u16 pipeline received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = build_background(src, width, height, bw, bh, fw, fh, fthresh, &bkg);
  if (status != RETURN_OK) {
    return status;
  }

  back.resize(static_cast<size_t>(count));
  rms.resize(static_cast<size_t>(count));

  status = sep_bkg_array(bkg, back.data(), SEP_TINT);
  if (status != RETURN_OK) {
    sep_bkg_free(bkg);
    return status;
  }

  status = sep_bkg_rmsarray(bkg, rms.data(), SEP_TINT);
  if (status != RETURN_OK) {
    sep_bkg_free(bkg);
    return status;
  }

  for (int64_t i = 0; i < count; ++i) {
    const int corrected =
        static_cast<int>(src[static_cast<size_t>(i)]) - back[static_cast<size_t>(i)];
    dst_subtracted[static_cast<size_t>(i)] =
        static_cast<uint16_t>(std::max(corrected, 0));
    dst_rms[static_cast<size_t>(i)] =
        static_cast<uint16_t>(std::max(rms[static_cast<size_t>(i)], 0));
  }

  sep_bkg_free(bkg);
  return RETURN_OK;
}

int run_fast_subtract_background(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted) {
  int status = RETURN_OK;
  sep_bkg *bkg = NULL;

  if (src == NULL || dst_subtracted == NULL) {
    put_errdetail("u16 pipeline received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = build_background(src, width, height, bw, bh, fw, fh, fthresh, &bkg);
  if (status != RETURN_OK) {
    return status;
  }

  status = sep_cuda_subtract_background_u16_fast(bkg, src, dst_subtracted);
  sep_bkg_free(bkg);
  return status;
}

int run_fast_subtract_background_and_fill_rms(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  int status = RETURN_OK;
  sep_bkg *bkg = NULL;

  if (src == NULL || dst_subtracted == NULL || dst_rms == NULL) {
    put_errdetail("u16 pipeline received a null pointer");
    return ILLEGAL_APER_PARAMS;
  }

  status = build_background(src, width, height, bw, bh, fw, fh, fthresh, &bkg);
  if (status != RETURN_OK) {
    return status;
  }

  status = sep_cuda_subtract_background_and_fill_rms_u16_fast(
      bkg, src, dst_subtracted, dst_rms);
  sep_bkg_free(bkg);
  return status;
}

}  // namespace

extern "C" SEP_API int sep_cuda_subtract_background_u16_reference(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted) {
  return run_reference_subtract_background(
      src,
      width,
      height,
      bw,
      bh,
      fw,
      fh,
      fthresh,
      dst_subtracted);
}

extern "C" SEP_API int sep_cuda_subtract_background_and_fill_rms_u16_reference(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  return run_reference_subtract_background_and_fill_rms(
      src,
      width,
      height,
      bw,
      bh,
      fw,
      fh,
      fthresh,
      dst_subtracted,
      dst_rms);
}

extern "C" SEP_API int sep_cuda_subtract_background_u16(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted) {
  return run_fast_subtract_background(
      src,
      width,
      height,
      bw,
      bh,
      fw,
      fh,
      fthresh,
      dst_subtracted);
}

extern "C" SEP_API int sep_cuda_subtract_background_and_fill_rms_u16(
    const uint16_t *src,
    int64_t width,
    int64_t height,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    uint16_t *dst_subtracted,
    uint16_t *dst_rms) {
  return run_fast_subtract_background_and_fill_rms(
      src,
      width,
      height,
      bw,
      bh,
      fw,
      fh,
      fthresh,
      dst_subtracted,
      dst_rms);
}
