#include <iostream>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <algorithm>

#include "yolox_detector.h"

#define CUDA_ASSERT(call)                                                     \
    do                                                                        \
    {                                                                         \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess)                                        \
        {                                                                     \
            printf("CUDA_CHECK Error:\n");                                    \
            printf("    File:       %s\n", __FILE__);                         \
            printf("    Line:       %d\n", __LINE__);                         \
            printf("    Error code: %d\n", error_code);                       \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        const cudaError_t error_code = call;                                  \
        if (error_code != cudaSuccess)                                        \
        {                                                                     \
            printf("CUDA_CHECK Error:\n");                                    \
            printf("    File:       %s\n", __FILE__);                         \
            printf("    Line:       %d\n", __LINE__);                         \
            printf("    Error code: %d\n", error_code);                       \
            printf("    Error text: %s\n", cudaGetErrorString(error_code));   \
        }                                                                     \
    } while (0)

namespace Infer
{

static Logger logger;

YOLOXDetector::YOLOXDetector(const char *model_path, const float conf_thres,
    const float nms_thres, const int target_size, const int num_class,
    const int max_batch_size)
    : conf_thres_(conf_thres)
    , nms_thres_(nms_thres)
    , target_size_(target_size)
    , num_class_(num_class)
    , max_batch_size_(max_batch_size)
{
    // --- Check settings
    if (target_size_ % strides_.back() != 0 || target_size_ < 32)
        return;

    // --- Init TRT
    // load model data
    std::ifstream engine_file(model_path, std::ios::binary);
    if (!engine_file)
    {
        std::cerr << "Failed to open engine file\n";
        return;
    }
    engine_file.seekg(0, engine_file.end);
    std::streamsize engine_size = engine_file.tellg();
    engine_file.seekg(0, engine_file.beg);
    std::unique_ptr<char[]> engine_data{std::make_unique<char[]>(engine_size)};
    if (!engine_file.read(engine_data.get(), engine_size))
    {
        std::cerr << "Failed to read engine file\n";
        return;
    }
    engine_file.close();

    // create trt runtime
    runtime_.reset(nvinfer1::createInferRuntime(logger));
    if (!runtime_)
    {
        std::cerr << "Failed to create runtime\n";
        return;
    }

    // deserialize engine
    engine_.reset(runtime_->deserializeCudaEngine(engine_data.get(), engine_size));
    if (!engine_)
    {
        std::cerr << "Failed to deserialize engine\n";
        return;
    }

    // create context
    context_.reset(engine_->createExecutionContext());
    if (!context_)
    {
        std::cerr << "Failed to create contexts\n";
        return;
    }

    // create stream
    stream_ = std::make_unique<cudaStream_t>();
    CUDA_ASSERT(cudaStreamCreate(stream_.get()));

    // get model info
    for (int i = 0; i < engine_->getNbIOTensors(); ++i)
    {
        const char *tensor_name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode io_mode = engine_->getTensorIOMode(tensor_name);
        if (io_mode == nvinfer1::TensorIOMode::kINPUT)
            in_tensor_info_ = {i, std::string(tensor_name)};
        else if (io_mode == nvinfer1::TensorIOMode::kOUTPUT)
            out_tensor_info_ = {i, std::string(tensor_name)};
    }
    // get max batch size
    int batch_size = engine_->getProfileShape(in_tensor_info_.second.c_str(), 0,
        nvinfer1::OptProfileSelector::kMAX).d[0];
    if (max_batch_size_ <= 0)
        max_batch_size_ = batch_size;

    // --- Create resources
    // create pinned host memory
    int max_in_size_byte = 3 * target_size_ * target_size_ * static_cast<int>(sizeof(float)) * max_batch_size_;
    int out0_hw = target_size_ / strides_[0], out1_hw = target_size_ / strides_[1], out2_hw = target_size_ / strides_[2];
    int max_out_size_byte = (out0_hw * out0_hw + out1_hw * out1_hw + out2_hw * out2_hw) * (num_class_ + 5) *
        static_cast<int>(sizeof(float)) * max_batch_size_;
    CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&pinned_in_host_), max_in_size_byte));
    CUDA_ASSERT(cudaMallocHost(reinterpret_cast<void **>(&pinned_out_host_), max_out_size_byte));
    // create device memory
    buffers_.resize(engine_->getNbIOTensors());
    CUDA_ASSERT(cudaMalloc(&buffers_[in_tensor_info_.first], max_in_size_byte));
    CUDA_ASSERT(cudaMalloc(&buffers_[out_tensor_info_.first], max_out_size_byte));

    // set in/out tensor addresses
    context_->setInputTensorAddress(in_tensor_info_.second.c_str(), buffers_[0]);
    context_->setOutputTensorAddress(out_tensor_info_.second.c_str(), buffers_[1]);

    is_inited_ = true;
}

YOLOXDetector::~YOLOXDetector()
{
    Cleanup();
}

YOLOXDetector::YOLOXDetector(YOLOXDetector &&other) noexcept
    : is_inited_(std::exchange(other.is_inited_, false))
    , conf_thres_(std::exchange(other.conf_thres_, {}))
    , nms_thres_(std::exchange(other.nms_thres_, {}))
    , target_size_(std::exchange(other.target_size_, {}))
    , num_class_(std::exchange(other.num_class_, {}))
    , max_batch_size_(std::exchange(other.max_batch_size_, {}))
    , stream_(std::move(other.stream_))
    , runtime_(std::move(other.runtime_))
    , engine_(std::move(other.engine_))
    , context_(std::move(other.context_))
    , in_tensor_info_(std::move(other.in_tensor_info_))
    , out_tensor_info_(std::move(other.out_tensor_info_))
    , buffers_(std::move(other.buffers_))
    , pinned_in_host_(std::exchange(other.pinned_in_host_, {}))
    , pinned_out_host_(std::exchange(other.pinned_out_host_, {}))
{

}

YOLOXDetector & YOLOXDetector::operator = (YOLOXDetector &&other) noexcept
{
    if (this != &other)
    {
        Cleanup();

        is_inited_ = std::exchange(other.is_inited_, false);
        conf_thres_ = std::exchange(other.conf_thres_, {});
        nms_thres_ = std::exchange(other.nms_thres_, {});
        target_size_ = std::exchange(other.target_size_, {});
        num_class_ = std::exchange(other.num_class_, {});
        max_batch_size_ = std::exchange(other.max_batch_size_, {});
        stream_ = std::move(other.stream_);
        runtime_ = std::move(other.runtime_);
        engine_ = std::move(other.engine_);
        context_ = std::move(other.context_);
        in_tensor_info_ = std::move(other.in_tensor_info_);
        out_tensor_info_ = std::move(other.out_tensor_info_);
        buffers_ = std::move(other.buffers_);
        pinned_in_host_ = std::exchange(other.pinned_in_host_, {});
        pinned_out_host_ = std::exchange(other.pinned_out_host_, {});
    }
    return *this;
}

std::vector<Object> YOLOXDetector::Detect(const cv::Mat &image) const
{
    if (!IsInited())
        return {};

    // --- Preprocessing
    int img_rows = image.rows;
    int img_cols = image.cols;
    float scale;
    int resize_rows, resize_cols, pad_rows, pad_cols;
    GetLetterboxDimensions(
        img_rows, img_cols, true,
        resize_rows, resize_cols, pad_rows, pad_cols, scale
    );
    cv::Mat letterbox{}, blob{};
    cv::resize(image, letterbox, cv::Size(resize_cols, resize_rows), 0, 0, cv::INTER_AREA);
    cv::copyMakeBorder(
        letterbox, letterbox,
        pad_rows / 2, pad_rows - pad_rows / 2,
        pad_cols / 2, pad_cols - pad_cols / 2,
        cv::BORDER_CONSTANT, cv::Scalar(114.0, 114.0, 114.0)
    );
    cv::dnn::blobFromImage(letterbox, blob, 1.0f, cv::Size(letterbox.cols, letterbox.rows), cv::Scalar(0, 0, 0), false, false, CV_32F);
    MakeContinuous(blob);

    // --- Inference
    // set input shape
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = 1;
    trt_in_dims.d[1] = 3;
    trt_in_dims.d[2] = letterbox.rows;
    trt_in_dims.d[3] = letterbox.cols;
    context_->setInputShape(in_tensor_info_.second.c_str(), trt_in_dims);
    // compute in/out size for dynamic shape input
    const auto out_dims = context_->getTensorShape(out_tensor_info_.second.c_str());
    const size_t in_size_byte = 3 * letterbox.rows * letterbox.cols * static_cast<int>(sizeof(float));
    const size_t out_size_byte = static_cast<int>(sizeof(float)) * out_dims.d[0] * out_dims.d[1] * out_dims.d[2];

    memcpy(pinned_in_host_, blob.data, in_size_byte);

    // execute
    CUDA_ASSERT(cudaMemcpyAsync(buffers_[0], pinned_in_host_, in_size_byte, cudaMemcpyHostToDevice, *stream_));

    context_->enqueueV3(*stream_);

    CUDA_ASSERT(cudaMemcpyAsync(pinned_out_host_, buffers_[1], out_size_byte, cudaMemcpyDeviceToHost, *stream_));
    CUDA_ASSERT(cudaStreamSynchronize(*stream_));

    // --- Postprocessing
    std::vector<Object> proposals, objects;
    std::vector<Anchor> anchors;

    const int num_anchors = out_dims.d[1];
    const int walk = out_dims.d[2];

    GetAnchors(letterbox.rows, letterbox.cols, anchors);
    GenerateProposals(pinned_out_host_, {1, num_anchors, walk}, anchors, proposals);
    NMS(proposals, objects, img_rows, img_cols, pad_rows / 2, pad_cols / 2, scale, scale);

    return objects;
}

std::vector<std::vector<Object>> YOLOXDetector::Detect(const std::vector<cv::Mat> &images) const
{
    if (!IsInited())
        return {};

    const int batch_size = static_cast<int>(images.size());
    if (batch_size == 0)
    {
        std::cerr << "Batch size == 0\n";
        return {};
    }
    else if (batch_size > max_batch_size_)
    {
        std::cerr << "Batch size " << batch_size << " exceeds max batch size "
            << max_batch_size_ << "\n";
        return {};
    }

    // --- Preprocessing
    std::vector<std::pair<int, int>> image_sizes(batch_size);
    for (int i = 0; i < batch_size; ++i)
        image_sizes[i] = {images[i].rows, images[i].cols};

    int out_rows, out_cols;
    std::vector<int> resize_rows_vec, resize_cols_vec;
    std::vector<int> pad_rows_vec, pad_cols_vec;
    std::vector<float> scale_vec;
    GetBatchLetterboxDimensions(
        image_sizes, out_rows, out_cols,
        resize_rows_vec, resize_cols_vec,
        pad_rows_vec, pad_cols_vec, scale_vec
    );

    const size_t single_blob_size = 3 * out_rows * out_cols * sizeof(float);

    for (int i = 0; i < batch_size; ++i)
    {
        cv::Mat resized;
        cv::resize(images[i], resized, cv::Size(resize_cols_vec[i], resize_rows_vec[i]),
            0, 0, cv::INTER_AREA);

        cv::Mat letterbox;
        cv::copyMakeBorder(resized, letterbox,
            pad_rows_vec[i] / 2, pad_rows_vec[i] - pad_rows_vec[i] / 2,
            pad_cols_vec[i] / 2, pad_cols_vec[i] - pad_cols_vec[i] / 2,
            cv::BORDER_CONSTANT, cv::Scalar(114, 114, 114));

        cv::Mat blob;
        cv::dnn::blobFromImage(letterbox, blob, 1.0f, cv::Size(letterbox.cols, letterbox.rows),
            cv::Scalar(0, 0, 0), false, false, CV_32F);
        MakeContinuous(blob);
        memcpy(pinned_in_host_ + i * 3 * out_rows * out_cols, blob.data, single_blob_size);
    }

    // --- Inference
    // set input shape
    nvinfer1::Dims trt_in_dims{};
    trt_in_dims.nbDims = 4;
    trt_in_dims.d[0] = batch_size;
    trt_in_dims.d[1] = 3;
    trt_in_dims.d[2] = out_rows;
    trt_in_dims.d[3] = out_cols;
    context_->setInputShape(in_tensor_info_.second.c_str(), trt_in_dims);
    // compute in/out size for dynamic shape input
    const auto out_dims = context_->getTensorShape(out_tensor_info_.second.c_str());
    const size_t in_size_byte = batch_size * 3 * out_rows * out_cols * static_cast<int>(sizeof(float));
    const size_t out_size_byte = static_cast<int>(sizeof(float)) * out_dims.d[0] * out_dims.d[1] * out_dims.d[2];

    // execute
    CUDA_ASSERT(cudaMemcpyAsync(buffers_[0], pinned_in_host_, in_size_byte, cudaMemcpyHostToDevice, *stream_));

    context_->enqueueV3(*stream_);

    CUDA_ASSERT(cudaMemcpyAsync(pinned_out_host_, buffers_[1], out_size_byte, cudaMemcpyDeviceToHost, *stream_));
    CUDA_ASSERT(cudaStreamSynchronize(*stream_));

    // --- Postprocessing
    std::vector<std::vector<Object>> batch_objects(batch_size);
    std::vector<Anchor> anchors;
    GetAnchors(out_rows, out_cols, anchors);

    const int num_anchors = out_dims.d[1];
    const int walk = out_dims.d[2];
    const size_t single_out_size = num_anchors * walk;

    for (int i = 0; i < batch_size; ++i)
    {
        const float *out_ptr = pinned_out_host_ + i * single_out_size;
        std::vector<Object> proposals;
        GenerateProposals(out_ptr, {1, num_anchors, walk}, anchors, proposals);
        NMS(proposals, batch_objects[i], image_sizes[i].first, image_sizes[i].second,
            pad_rows_vec[i] / 2, pad_cols_vec[i] / 2, scale_vec[i], scale_vec[i]);
    }

    return batch_objects;
}

bool YOLOXDetector::DrawObjects(cv::Mat &image, const std::vector<Object> &objects,
    const std::vector<std::string> &labels, bool is_silent)
{
    for (auto obj : objects)
    {
        if (obj.label >= static_cast<int>(labels.size()))
            return false;

        if (is_silent != true)
            std::printf("%s = %.2f%% at (%.1f, %.1f)  %.1f x %.1f\n", labels[obj.label].c_str(), obj.prob * 100.0f,
                obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        char text[256];
        std::snprintf(text, sizeof(text), "%s %.1f%%", labels[obj.label].c_str(), obj.prob * 100.0f);

        auto scalar = cv::Scalar(255, 255, 255);
        cv::rectangle(image, obj.rect, scalar, 2);

        int base_line = 5;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.75, 1, &base_line);

        int x = obj.rect.x - 1;
        int y = obj.rect.y - label_size.height - base_line;
        y = std::max(0, y);
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + base_line)),
            scalar, -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height + base_line / 2),
            cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 0, 0), 2);
    }

    return true;
}

void YOLOXDetector::GetLetterboxDimensions(const int img_rows, const int img_cols, const bool is_dynamic,
    int &resize_rows, int &resize_cols, int &pad_rows, int &pad_cols, float &scale) const
{
    scale = static_cast<float>(target_size_) / std::max(img_rows, img_cols);
    resize_rows = static_cast<int>(std::round(img_rows * scale));
    resize_cols = static_cast<int>(std::round(img_cols * scale));

    if (is_dynamic)
    {
        pad_rows = (resize_rows + strides_.back() - 1) / strides_.back() * strides_.back() - resize_rows;
        pad_cols = (resize_cols + strides_.back() - 1) / strides_.back() * strides_.back() - resize_cols;
    }
    else
    {
        pad_rows = target_size_ - resize_rows;
        pad_cols = target_size_ - resize_cols;
    }
}

void YOLOXDetector::GetBatchLetterboxDimensions(
    const std::vector<std::pair<int, int>> &image_sizes, int &out_rows, int &out_cols,
    std::vector<int> &resize_rows_vec, std::vector<int> &resize_cols_vec,
    std::vector<int> &pad_rows_vec, std::vector<int> &pad_cols_vec, std::vector<float> &scale_vec) const
{
    const size_t batch_size = image_sizes.size();
    resize_rows_vec.resize(batch_size);
    resize_cols_vec.resize(batch_size);
    pad_rows_vec.resize(batch_size);
    pad_cols_vec.resize(batch_size);
    scale_vec.resize(batch_size);

    int max_resize_rows = 0, max_resize_cols = 0;

    // get max resize size of each image
    for (size_t i = 0; i < batch_size; ++i)
    {
        int img_rows = image_sizes[i].first;
        int img_cols = image_sizes[i].second;

        float scale = static_cast<float>(target_size_) / std::max(img_rows, img_cols);
        int resize_rows = static_cast<int>(std::round(img_rows * scale));
        int resize_cols = static_cast<int>(std::round(img_cols * scale));

        resize_rows_vec[i] = resize_rows;
        resize_cols_vec[i] = resize_cols;
        scale_vec[i] = scale;

        max_resize_rows = std::max(max_resize_rows, resize_rows);
        max_resize_cols = std::max(max_resize_cols, resize_cols);
    }
    out_rows = (max_resize_rows + strides_.back() - 1) / strides_.back() * strides_.back();
    out_cols = (max_resize_cols + strides_.back() - 1) / strides_.back() * strides_.back();

    // get pad size
    for (size_t i = 0; i < batch_size; ++i)
    {
        pad_rows_vec[i] = out_rows - resize_rows_vec[i];
        pad_cols_vec[i] = out_cols - resize_cols_vec[i];
    }
}

void YOLOXDetector::GetAnchors(const int rows, const int cols, std::vector<Anchor> &anchors) const
{
    for (const auto &stride : strides_)
    {
        int grid_rows = rows / stride;
        int grid_cols = cols / stride;
        for (int gh = 0; gh < grid_rows; ++gh)
        {
            for (int gw = 0; gw < grid_cols; ++gw)
            {
                Anchor anchor;
                anchor.x = gw;
                anchor.y = gh;
                anchor.stride = stride;
                anchors.emplace_back(anchor);
            }
        }
    }
}

void YOLOXDetector::GenerateProposals(const float *blob, const std::vector<int> nhwc_shape,
    const std::vector<Anchor> &anchors, std::vector<Object> &proposals) const
{
    const int num_anchor = nhwc_shape[1];
    const int walk = nhwc_shape[2];
    const int num_class = walk - 5;
    for (int i = 0; i < num_anchor; ++i)
    {
        const float *box_ptr = blob + i * walk;
        const float *cls_ptr = box_ptr + 5;
        const float *pred_ptr = std::max_element(cls_ptr, cls_ptr + num_class);
        float objness = box_ptr[4];
        float conf = *pred_ptr * objness;
        int label = pred_ptr - cls_ptr;
        if (conf >= conf_thres_)
        {
            float pb_cx = (box_ptr[0] + anchors[i].x) * anchors[i].stride;
            float pb_cy = (box_ptr[1] + anchors[i].y) * anchors[i].stride;

            float w = std::exp(box_ptr[2]) * anchors[i].stride;
            float h = std::exp(box_ptr[3]) * anchors[i].stride;
            float x0 = pb_cx - w * 0.5f;
            float y0 = pb_cy - h * 0.5f;

            Object object;
            object.rect.x = x0;
            object.rect.y = y0;
            object.rect.width = w;
            object.rect.height = h;
            object.label = label;
            object.prob = conf;
            proposals.emplace_back(object);
        }
    }
}

void YOLOXDetector::NMS(std::vector<Object> &proposals, std::vector<Object> &objects,
    const int orig_h, const int orig_w,
    const float dh, const float dw,
    const float ratio_h, const float ratio_w) const
{
    std::vector<cv::Rect> boxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    for (const auto &prop : proposals)
    {
        boxes.emplace_back(prop.rect);
        scores.emplace_back(prop.prob);
        labels.emplace_back(prop.label);
    }

    cv::dnn::NMSBoxes(boxes, scores, conf_thres_, nms_thres_, indices);

    objects.resize(indices.size());
    for (size_t i = 0; i < indices.size(); ++i)
    {
        const float &score = proposals[indices[i]].prob;
        const int &label = proposals[indices[i]].label;
        const cv::Rect_<float> &box = proposals[indices[i]].rect;
        float x0 = box.x;
        float y0 = box.y;
        float x1 = box.x + box.width;
        float y1 = box.y + box.height;

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = std::clamp(x0, 0.0f, static_cast<float>(orig_w));
        y0 = std::clamp(y0, 0.0f, static_cast<float>(orig_h));
        x1 = std::clamp(x1, x0, static_cast<float>(orig_w));
        y1 = std::clamp(y1, y0, static_cast<float>(orig_h));

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
        objects[i].prob = score;
        objects[i].label = label;
    }
}

void YOLOXDetector::Cleanup() noexcept
{
    for (const auto &buffer : buffers_)
        CUDA_CHECK(cudaFree(buffer));
    buffers_.clear();

    CUDA_CHECK(cudaFreeHost(pinned_in_host_));
    CUDA_CHECK(cudaFreeHost(pinned_out_host_));
    pinned_in_host_ = nullptr;
    pinned_out_host_ = nullptr;

    if (stream_ && *stream_)
        CUDA_CHECK(cudaStreamDestroy(*stream_));
    stream_.reset();
}

}   // namespace Infer
