#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "sep_cuda.h"

namespace cv {

constexpr int CV_16UC1 = 0;

// dummy cv::Mat for demo
struct Mat {
  int rows = 0;
  int cols = 0;
  int mat_type = CV_16UC1;
  std::vector<uint16_t> buffer;

  Mat() = default;

  Mat(int rows_in, int cols_in, int type_in)
      : rows(rows_in),
        cols(cols_in),
        mat_type(type_in),
        buffer(static_cast<size_t>(rows_in) * static_cast<size_t>(cols_in), 0) {}

  bool empty() const {
    return buffer.empty();
  }

  bool isContinuous() const {
    return true;
  }

  int type() const {
    return mat_type;
  }

  uint16_t *data() {
    return buffer.data();
  }

  const uint16_t *data() const {
    return buffer.data();
  }

  uint16_t &at(int row, int col) {
    return buffer[static_cast<size_t>(row) * cols + col];
  }

  const uint16_t &at(int row, int col) const {
    return buffer[static_cast<size_t>(row) * cols + col];
  }

  template <typename T>
  T *ptr(int row = 0) {
    return reinterpret_cast<T *>(buffer.data() + static_cast<size_t>(row) * cols);
  }

  template <typename T>
  const T *ptr(int row = 0) const {
    return reinterpret_cast<const T *>(buffer.data() + static_cast<size_t>(row) * cols);
  }
};

}  // namespace cv


static void assert_mat_16uc1(const cv::Mat &src) {
  if (src.empty() || !src.isContinuous() || src.type() != cv::CV_16UC1) {
    throw std::runtime_error("dummy cv::Mat must be continuous CV_16UC1");
  }
}

static void subtract_background_in_place(cv::Mat &src) {
  assert_mat_16uc1(src);
  const auto input = sepcuda::HostImage(src.ptr<uint16_t>(0), src.cols, src.rows);
  const auto background = sepcuda::Background(
      input,
      sepcuda::BackgroundOptions{32, 32, 3, 3, 0.0});

  std::vector<int> bg = background.back<int>();
  uint16_t *src_ptr = src.ptr<uint16_t>(0);
  for (size_t i = 0; i < src.buffer.size(); ++i) {
    const int corrected = static_cast<int>(src_ptr[i]) - bg[i];
    src_ptr[i] = static_cast<uint16_t>(std::max(corrected, 0));
  }
}

static void subtract_background_and_fill_rms(cv::Mat &src, cv::Mat &imgRms) {
  assert_mat_16uc1(src);
  assert_mat_16uc1(imgRms);
  const auto input = sepcuda::HostImage(src.ptr<uint16_t>(0), src.cols, src.rows);
  const auto background = sepcuda::Background(
      input,
      sepcuda::BackgroundOptions{32, 32, 3, 3, 0.0});

  std::vector<int> bg = background.back<int>();
  std::vector<int> rms = background.rms<int>();

  uint16_t *src_ptr = src.ptr<uint16_t>(0);
  uint16_t *rms_ptr = imgRms.ptr<uint16_t>(0);
  for (size_t i = 0; i < src.buffer.size(); ++i) {
    const int corrected = static_cast<int>(src_ptr[i]) - bg[i];
    src_ptr[i] = static_cast<uint16_t>(std::max(corrected, 0));
    rms_ptr[i] = static_cast<uint16_t>(std::max(rms[i], 0));
  }
}

static void subtract_background_preserve_dynamic_range(cv::Mat &src, cv::Mat &imgRms) {
  assert_mat_16uc1(src);
  assert_mat_16uc1(imgRms);
  const auto input = sepcuda::HostImage(src.ptr<uint16_t>(0), src.cols, src.rows);
  const auto background = sepcuda::Background(
      input,
      sepcuda::BackgroundOptions{32, 32, 3, 3, 0.0});

  std::vector<int> bg = background.back<int>();
  std::vector<int> rms = background.rms<int>();

  int min_value = 0;
  bool first = true;
  std::vector<int> shifted(src.buffer.size(), 0);
  uint16_t *src_ptr = src.ptr<uint16_t>(0);
  uint16_t *rms_ptr = imgRms.ptr<uint16_t>(0);

  for (size_t i = 0; i < src.buffer.size(); ++i) {
    shifted[i] = static_cast<int>(src_ptr[i]) - bg[i];
    if (first || shifted[i] < min_value) {
      min_value = shifted[i];
      first = false;
    }
  }

  for (size_t i = 0; i < src.buffer.size(); ++i) {
    src_ptr[i] = static_cast<uint16_t>(shifted[i] - min_value);
    rms_ptr[i] = static_cast<uint16_t>(std::max((rms[i] * 2) - min_value, 0));
  }
}

static void printMat(const cv::Mat &mat, const char *label) {
  std::printf("%s\n", label);
  for (int row = 0; row < mat.rows; ++row) {
    for (int col = 0; col < mat.cols; ++col) {
      std::printf("%5u ", static_cast<unsigned>(mat.at(row, col)));
    }
    std::printf("\n");
  }
}

int main() {
  cv::Mat src(6, 6, cv::CV_16UC1);
  for (int row = 0; row < src.rows; ++row) {
    for (int col = 0; col < src.cols; ++col) {
      src.at(row, col) = static_cast<uint16_t>(1000 + row * 10 + col * 3);
    }
  }
  src.at(2, 3) = 1800;
  src.at(4, 1) = 1500;

  cv::Mat rms(src.rows, src.cols, cv::CV_16UC1);
  cv::Mat src_keep_dynamic_range = src;
  cv::Mat rms_keep_dynamic_range(src.rows, src.cols, cv::CV_16UC1);

  try {
    printMat(src, "original src:");

    subtract_background_and_fill_rms(src, rms);
    printMat(src, "after subtract_background_and_fill_rms(src, rms), src becomes background-subtracted and clipped:");
    printMat(rms, "rms map:");

    subtract_background_preserve_dynamic_range(src_keep_dynamic_range, rms_keep_dynamic_range);
    printMat(
        src_keep_dynamic_range,
        "after subtract_background_preserve_dynamic_range(src, rms), src is shifted so all pixels stay non-negative:");
    printMat(rms_keep_dynamic_range, "adjusted rms map:");
  } catch (const sepcuda::Error &error) {
    std::fprintf(stderr, "background estimation failed: %s\n", error.what());
    return 1;
  }

  return 0;
}
