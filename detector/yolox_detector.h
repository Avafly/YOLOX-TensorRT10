#ifndef YOLOX_DETECTOR_H_
#define YOLOX_DETECTOR_H_

#include <string>
#include <vector>
#include <memory>
#include <utility>

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <cuda_runtime_api.h>

#include <opencv2/opencv.hpp>

#include "logging.h"

namespace Infer
{

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

struct Anchor
{
    int x;
    int y;
    int stride;
};

class YOLOXDetector
{
public:
    explicit YOLOXDetector(const char *model_path, const float conf_thres,
        const float nms_thres, const int target_size, const int num_class,
        const int max_batch_size = 0);
    ~YOLOXDetector();

    YOLOXDetector(YOLOXDetector &&other) noexcept;
    YOLOXDetector & operator = (YOLOXDetector &&other) noexcept;

    YOLOXDetector(const YOLOXDetector &) = delete;
    YOLOXDetector & operator = (const YOLOXDetector &) = delete;

    std::vector<Object> Detect(const cv::Mat &image) const;
    std::vector<std::vector<Object>> Detect(const std::vector<cv::Mat> &images) const;

    static bool DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
        const std::vector<std::string> &labels, bool is_silent);

    inline bool IsInited() const noexcept;

    inline int GetMaxBatchSize() const noexcept;

private:
    bool is_inited_{false};
    float conf_thres_{};
    float nms_thres_{};
    int target_size_{};
    int num_class_{};
    int max_batch_size_{};

    std::unique_ptr<cudaStream_t> stream_{};
    std::unique_ptr<nvinfer1::IRuntime> runtime_{};
    std::unique_ptr<nvinfer1::ICudaEngine> engine_{};
    std::unique_ptr<nvinfer1::IExecutionContext> context_{};
    std::pair<int, std::string> in_tensor_info_{};
    std::pair<int, std::string> out_tensor_info_{};
    std::vector<void *> buffers_{};
    float *pinned_in_host_{};
    float *pinned_out_host_{};

    static inline const std::vector<int> strides_{8, 16, 32};

    inline void MakeContinuous(cv::Mat &mat) const;

    void GetLetterboxDimensions(const int img_rows, const int img_cols, const bool is_dynamic,
        int &resize_rows, int &resize_cols, int &pad_rows, int &pad_cols, float &scale) const;

    void GetBatchLetterboxDimensions(const std::vector<std::pair<int, int>> &image_sizes,
        int &out_rows, int &out_cols, std::vector<int> &resize_rows_vec, std::vector<int> &resize_cols_vec,
        std::vector<int> &pad_rows_vec, std::vector<int> &pad_cols_vec, std::vector<float> &scale_vec) const;

    void GetAnchors(const int rows, const int cols, std::vector<Anchor> &anchors) const;

    void GenerateProposals(const float *blob, const std::vector<int> nhwc_shape,
        const std::vector<Anchor> &anchors, std::vector<Object> &proposals) const;

    void NMS(std::vector<Object> &proposals, std::vector<Object> &objects,
        const int orig_h, const int orig_w,
        const float dh, const float dw,
        const float ratio_h, const float ratio_w) const;

    void Cleanup() noexcept;
};

}   // namespace Infer

#include "yolox_detector.inl"

#endif  // YOLOX_DETECTOR_H_
