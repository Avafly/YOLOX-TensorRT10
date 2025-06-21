namespace Infer
{

inline bool YOLOXDetector::IsInited() const noexcept
{
    return is_inited_;
}

template <typename T>
inline T YOLOXDetector::Clamp(T x, T min_x, T max_x) const
{
    return x > min_x ? (x < max_x ? x : max_x) : min_x;
}

inline void YOLOXDetector::MakeContinuous(cv::Mat &mat) const
{
    if (!mat.isContinuous())
        mat = mat.clone();
}
    
}   // namespace Infer
