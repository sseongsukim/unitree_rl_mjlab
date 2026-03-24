#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <yaml-cpp/yaml.h>

namespace go2_camera
{

class RealSenseSource
{
public:
    struct DepthFrame
    {
        int width = 0;
        int height = 0;
        std::vector<uint16_t> data;

        bool valid() const
        {
            return width > 0 && height > 0 && data.size() == static_cast<size_t>(width * height);
        }
    };

    explicit RealSenseSource(const YAML::Node& cfg);
    ~RealSenseSource();

    void start();
    void stop();

    std::vector<float> latest_features() const;
    DepthFrame latest_depth_frame() const;
    bool enabled() const { return enabled_; }
    int feature_dim() const { return grid_rows_ * grid_cols_; }

private:
    void capture_loop();

    bool enabled_ = false;
    int width_ = 424;
    int height_ = 240;
    int fps_ = 30;
    int grid_rows_ = 8;
    int grid_cols_ = 8;
    int frame_timeout_ms_ = 100;
    float min_depth_m_ = 0.15f;
    float max_depth_m_ = 3.0f;
    std::string serial_;

    mutable std::mutex mutex_;
    std::vector<float> latest_features_;
    DepthFrame latest_depth_frame_;
    std::thread worker_;
    std::atomic<bool> running_{false};
};

}  // namespace go2_camera
