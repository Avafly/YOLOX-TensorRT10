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

}   // namespace Infer
