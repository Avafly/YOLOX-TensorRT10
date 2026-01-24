namespace Infer
{

inline bool YOLOXDetector::IsInited() const noexcept
{
    return is_inited_;
}

inline int YOLOXDetector::GetMaxBatchSize() const noexcept
{
    return max_batch_size_;
}

inline void YOLOXDetector::MakeContinuous(cv::Mat &mat) const
{
    if (!mat.isContinuous())
        mat = mat.clone();
}

}   // namespace Infer
