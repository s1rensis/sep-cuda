#ifndef SEPCUDA_SEP_H
#define SEPCUDA_SEP_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _MSC_VER
#define SEP_API __declspec(dllexport)
#else
#define SEP_API __attribute__((visibility("default")))
#endif

#define SEP_TBYTE 11
#define SEP_TINT 31
#define SEP_TFLOAT 42
#define SEP_TDOUBLE 82

#define SEP_NOISE_NONE 0
#define SEP_NOISE_STDDEV 1
#define SEP_NOISE_VAR 2

#define RETURN_OK 0
#define MEMORY_ALLOC_ERROR 1
#define PIXSTACK_FULL 2
#define ILLEGAL_DTYPE 3
#define ILLEGAL_SUBPIX 4
#define NON_ELLIPSE_PARAMS 5
#define ILLEGAL_APER_PARAMS 6
#define DEBLEND_OVERFLOW 7
#define LINE_NOT_IN_BUF 8
#define RELTHRESH_NO_NOISE 9
#define UNKNOWN_NOISE_TYPE 10
#define SEP_CUDA_UNAVAILABLE 100
#define SEP_CUDA_RUNTIME_ERROR 101

typedef struct {
  const void *data;
  const void *noise;
  const void *mask;
  const void *segmap;
  int dtype;
  int ndtype;
  int mdtype;
  int sdtype;
  int64_t *segids;
  int64_t *idcounts;
  int64_t numids;
  int64_t w;
  int64_t h;
  double noiseval;
  short noise_type;
  double gain;
  double maskthresh;
} sep_image;

typedef struct {
  int64_t w, h;
  int64_t bw, bh;
  int64_t nx, ny;
  int64_t n;
  float global;
  float globalrms;
  float *back;
  float *dback;
  float *sigma;
  float *dsigma;
} sep_bkg;

SEP_API int sep_background(
    const sep_image *image,
    int64_t bw,
    int64_t bh,
    int64_t fw,
    int64_t fh,
    double fthresh,
    sep_bkg **bkg);

SEP_API float sep_bkg_global(const sep_bkg *bkg);
SEP_API float sep_bkg_globalrms(const sep_bkg *bkg);
SEP_API float sep_bkg_pix(const sep_bkg *bkg, int64_t x, int64_t y);

SEP_API int sep_bkg_line(const sep_bkg *bkg, int64_t y, void *line, int dtype);
SEP_API int sep_bkg_subline(const sep_bkg *bkg, int64_t y, void *line, int dtype);
SEP_API int sep_bkg_rmsline(const sep_bkg *bkg, int64_t y, void *line, int dtype);

SEP_API int sep_bkg_array(const sep_bkg *bkg, void *arr, int dtype);
SEP_API int sep_bkg_subarray(const sep_bkg *bkg, void *arr, int dtype);
SEP_API int sep_bkg_rmsarray(const sep_bkg *bkg, void *arr, int dtype);

SEP_API void sep_bkg_free(sep_bkg *bkg);

SEP_API void sep_get_errmsg(int status, char *errtext);
SEP_API void sep_get_errdetail(char *errtext);

#ifdef __cplusplus
}

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace sepcuda {

struct BackgroundOptions {
  int64_t bw = 64;
  int64_t bh = 64;
  int64_t fw = 3;
  int64_t fh = 3;
  double fthresh = 0.0;
};

class Error : public std::runtime_error {
 public:
  Error(int status, std::string message)
      : std::runtime_error(std::move(message)), status_(status) {}

  int status() const noexcept {
    return status_;
  }

 private:
  int status_;
};

struct ImageView {
  const void *data = nullptr;
  int dtype = 0;
  int64_t width = 0;
  int64_t height = 0;
  const void *mask = nullptr;
  int mask_dtype = 0;
  double mask_threshold = 0.0;

  sep_image to_sep_image() const noexcept {
    sep_image image{};
    image.data = data;
    image.noise = nullptr;
    image.mask = mask;
    image.segmap = nullptr;
    image.dtype = dtype;
    image.ndtype = 0;
    image.mdtype = mask ? mask_dtype : 0;
    image.sdtype = 0;
    image.segids = nullptr;
    image.idcounts = nullptr;
    image.numids = 0;
    image.w = width;
    image.h = height;
    image.noiseval = 0.0;
    image.noise_type = SEP_NOISE_NONE;
    image.gain = 1.0;
    image.maskthresh = mask_threshold;
    return image;
  }
};

namespace detail {

template <typename T>
using remove_cvref_t = std::remove_cv_t<std::remove_reference_t<T>>;

template <typename T>
inline constexpr bool always_false_v = false;

template <typename T>
inline constexpr bool is_byte_v =
    std::is_same_v<remove_cvref_t<T>, uint8_t> ||
    std::is_same_v<remove_cvref_t<T>, unsigned char>;

template <typename T>
inline constexpr bool is_native_sep_dtype_v =
    is_byte_v<T> ||
    std::is_same_v<remove_cvref_t<T>, int> ||
    std::is_same_v<remove_cvref_t<T>, float> ||
    std::is_same_v<remove_cvref_t<T>, double>;

template <typename T>
inline int native_sep_dtype() {
  using Bare = remove_cvref_t<T>;
  if constexpr (is_byte_v<Bare>) {
    return SEP_TBYTE;
  } else if constexpr (std::is_same_v<Bare, int>) {
    return SEP_TINT;
  } else if constexpr (std::is_same_v<Bare, float>) {
    return SEP_TFLOAT;
  } else if constexpr (std::is_same_v<Bare, double>) {
    return SEP_TDOUBLE;
  } else {
    static_assert(always_false_v<T>, "Unsupported native SEP dtype");
  }
}

template <typename T>
inline int output_sep_dtype() {
  using Bare = remove_cvref_t<T>;
  if constexpr (std::is_same_v<Bare, int>) {
    return SEP_TINT;
  } else if constexpr (std::is_same_v<Bare, float>) {
    return SEP_TFLOAT;
  } else if constexpr (std::is_same_v<Bare, double>) {
    return SEP_TDOUBLE;
  } else {
    static_assert(always_false_v<T>, "Output type must be int, float, or double");
  }
}

inline void require_valid_shape(int64_t width, int64_t height) {
  if (width <= 0 || height <= 0) {
    throw std::invalid_argument("image dimensions must be positive");
  }
}

inline void require_non_null(const void *ptr, const char *label) {
  if (ptr == nullptr) {
    throw std::invalid_argument(std::string(label) + " must not be null");
  }
}

inline void require_expected_size(int64_t actual, int64_t expected, const char *label) {
  if (actual != expected) {
    throw std::invalid_argument(std::string(label) + " size does not match image dimensions");
  }
}

inline std::string error_message(int status) {
  char errtext[128];
  char detail[512];
  sep_get_errmsg(status, errtext);
  sep_get_errdetail(detail);
  std::string message(errtext);
  if (detail[0] != '\0') {
    message += ": ";
    message += detail;
  }
  return message;
}

inline void throw_on_error(int status) {
  if (status != RETURN_OK) {
    throw Error(status, error_message(status));
  }
}

template <typename SrcT, typename DstT>
inline std::vector<DstT> cast_copy(const SrcT *source, int64_t count) {
  std::vector<DstT> out(static_cast<size_t>(count));
  for (int64_t i = 0; i < count; ++i) {
    out[static_cast<size_t>(i)] = static_cast<DstT>(source[i]);
  }
  return out;
}

}  // namespace detail

class HostImage {
 public:
  HostImage() = default;

  template <typename PixelT>
  HostImage(const PixelT *data, int64_t width, int64_t height) {
    set_pixels(data, width, height);
  }

  template <typename PixelT, typename MaskT>
  HostImage(
      const PixelT *data,
      int64_t width,
      int64_t height,
      const MaskT *mask,
      double mask_threshold = 0.0) {
    set_pixels(data, width, height);
    set_mask(mask, mask_threshold);
  }

  template <typename PixelT>
  void set_pixels(const PixelT *data, int64_t width, int64_t height) {
    detail::require_non_null(data, "data");
    detail::require_valid_shape(width, height);

    view_.width = width;
    view_.height = height;

    if constexpr (detail::is_native_sep_dtype_v<PixelT>) {
      owned_pixels_i32_.clear();
      view_.data = data;
      view_.dtype = detail::native_sep_dtype<PixelT>();
    } else if constexpr (
        std::is_integral_v<detail::remove_cvref_t<PixelT>> &&
        sizeof(detail::remove_cvref_t<PixelT>) <= sizeof(int)) {
      owned_pixels_i32_ = detail::cast_copy<PixelT, int>(data, width * height);
      view_.data = owned_pixels_i32_.data();
      view_.dtype = SEP_TINT;
    } else {
      static_assert(detail::always_false_v<PixelT>, "Unsupported pixel type");
    }
  }

  template <typename MaskT>
  void set_mask(const MaskT *mask, double mask_threshold = 0.0) {
    detail::require_non_null(mask, "mask");
    const int64_t count = view_.width * view_.height;

    view_.mask_threshold = mask_threshold;

    if constexpr (detail::is_native_sep_dtype_v<MaskT>) {
      owned_mask_i32_.clear();
      owned_mask_u8_.clear();
      view_.mask = mask;
      view_.mask_dtype = detail::native_sep_dtype<MaskT>();
    } else if constexpr (std::is_same_v<detail::remove_cvref_t<MaskT>, bool>) {
      owned_mask_i32_.clear();
      owned_mask_u8_.resize(static_cast<size_t>(count));
      for (int64_t i = 0; i < count; ++i) {
        owned_mask_u8_[static_cast<size_t>(i)] =
            mask[i] ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0);
      }
      view_.mask = owned_mask_u8_.data();
      view_.mask_dtype = SEP_TBYTE;
    } else if constexpr (
        std::is_integral_v<detail::remove_cvref_t<MaskT>> &&
        sizeof(detail::remove_cvref_t<MaskT>) <= sizeof(int)) {
      owned_mask_u8_.clear();
      owned_mask_i32_ = detail::cast_copy<MaskT, int>(mask, count);
      view_.mask = owned_mask_i32_.data();
      view_.mask_dtype = SEP_TINT;
    } else {
      static_assert(detail::always_false_v<MaskT>, "Unsupported mask type");
    }
  }

  void clear_mask() noexcept {
    owned_mask_i32_.clear();
    owned_mask_u8_.clear();
    view_.mask = nullptr;
    view_.mask_dtype = 0;
    view_.mask_threshold = 0.0;
  }

  const ImageView &view() const noexcept {
    return view_;
  }

 private:
  ImageView view_{};
  std::vector<int> owned_pixels_i32_;
  std::vector<int> owned_mask_i32_;
  std::vector<uint8_t> owned_mask_u8_;
};

class Background {
 public:
  Background() noexcept : handle_(nullptr, &sep_bkg_free) {}

  explicit Background(const HostImage &image, BackgroundOptions options = {})
      : Background() {
    reset(image, options);
  }

  explicit Background(const ImageView &image, BackgroundOptions options = {})
      : Background() {
    reset(image, options);
  }

  Background(Background &&) noexcept = default;
  Background &operator=(Background &&) noexcept = default;

  Background(const Background &) = delete;
  Background &operator=(const Background &) = delete;

  template <typename PixelT>
  static Background from_pixels(
      const PixelT *data,
      int64_t width,
      int64_t height,
      BackgroundOptions options = {}) {
    HostImage image(data, width, height);
    return Background(image, options);
  }

  template <typename PixelT, typename MaskT>
  static Background from_pixels(
      const PixelT *data,
      int64_t width,
      int64_t height,
      const MaskT *mask,
      double mask_threshold = 0.0,
      BackgroundOptions options = {}) {
    HostImage image(data, width, height, mask, mask_threshold);
    return Background(image, options);
  }

  void reset() noexcept {
    handle_.reset();
  }

  void reset(const HostImage &image, BackgroundOptions options = {}) {
    reset(image.view(), options);
  }

  void reset(const ImageView &image, BackgroundOptions options = {}) {
    sep_image low_level = image.to_sep_image();
    sep_bkg *result = nullptr;
    detail::throw_on_error(
        sep_background(
            &low_level,
            options.bw,
            options.bh,
            options.fw,
            options.fh,
            options.fthresh,
            &result));
    handle_.reset(result);
  }

  bool empty() const noexcept {
    return handle_.get() == nullptr;
  }

  explicit operator bool() const noexcept {
    return !empty();
  }

  int64_t width() const {
    ensure_ready();
    return handle_->w;
  }

  int64_t height() const {
    ensure_ready();
    return handle_->h;
  }

  int64_t size() const {
    ensure_ready();
    return handle_->w * handle_->h;
  }

  float global() const {
    ensure_ready();
    return sep_bkg_global(handle_.get());
  }

  float global_rms() const {
    ensure_ready();
    return sep_bkg_globalrms(handle_.get());
  }

  float at(int64_t x, int64_t y) const {
    ensure_ready();
    return sep_bkg_pix(handle_.get(), x, y);
  }

  template <typename OutputT = float>
  std::vector<OutputT> back() const {
    std::vector<OutputT> out(static_cast<size_t>(size()));
    fill_back(out.data(), static_cast<int64_t>(out.size()));
    return out;
  }

  template <typename OutputT = float>
  std::vector<OutputT> rms() const {
    std::vector<OutputT> out(static_cast<size_t>(size()));
    fill_rms(out.data(), static_cast<int64_t>(out.size()));
    return out;
  }

  template <typename OutputT>
  void fill_back(OutputT *data, int64_t count) const {
    ensure_ready();
    detail::require_non_null(data, "output buffer");
    detail::require_expected_size(count, size(), "output buffer");
    detail::throw_on_error(
        sep_bkg_array(handle_.get(), data, detail::output_sep_dtype<OutputT>()));
  }

  template <typename OutputT>
  void fill_rms(OutputT *data, int64_t count) const {
    ensure_ready();
    detail::require_non_null(data, "output buffer");
    detail::require_expected_size(count, size(), "output buffer");
    detail::throw_on_error(
        sep_bkg_rmsarray(handle_.get(), data, detail::output_sep_dtype<OutputT>()));
  }

  template <typename T>
  void subtract_from(T *data, int64_t count) const {
    ensure_ready();
    detail::require_non_null(data, "data");
    detail::require_expected_size(count, size(), "data");
    detail::throw_on_error(
        sep_bkg_subarray(handle_.get(), data, detail::output_sep_dtype<T>()));
  }

  template <typename T>
  void subtract_from(std::vector<T> &data) const {
    subtract_from(data.data(), static_cast<int64_t>(data.size()));
  }

  const sep_bkg *raw() const noexcept {
    return handle_.get();
  }

 private:
  void ensure_ready() const {
    if (handle_.get() == nullptr) {
      throw std::logic_error("sepcuda::Background is empty");
    }
  }

  std::unique_ptr<sep_bkg, decltype(&sep_bkg_free)> handle_;
};

}  // namespace sepcuda

#endif

#endif
