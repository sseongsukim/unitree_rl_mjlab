#include "RealSenseSource.h"

#include <algorithm>
#include <chrono>

#include <spdlog/spdlog.h>

#ifdef GO2_CAMERA_HAS_REALSENSE
#include <librealsense2/rs.hpp>
#endif

namespace go2_camera
{

RealSenseSource::RealSenseSource(const YAML::Node& cfg)
{
    if(cfg.IsNull())
    {
        latest_features_.assign(feature_dim(), 0.0f);
        return;
    }

    enabled_ = cfg["enabled"].as<bool>(true);
    width_ = cfg["width"].as<int>(width_);
    height_ = cfg["height"].as<int>(height_);
    fps_ = cfg["fps"].as<int>(fps_);
    grid_rows_ = cfg["grid_rows"].as<int>(grid_rows_);
    grid_cols_ = cfg["grid_cols"].as<int>(grid_cols_);
    frame_timeout_ms_ = cfg["frame_timeout_ms"].as<int>(frame_timeout_ms_);
    min_depth_m_ = cfg["min_depth_m"].as<float>(min_depth_m_);
    max_depth_m_ = cfg["max_depth_m"].as<float>(max_depth_m_);
    serial_ = cfg["serial"].as<std::string>("");

    latest_features_.assign(feature_dim(), 0.0f);
}

RealSenseSource::~RealSenseSource()
{
    stop();
}

void RealSenseSource::start()
{
    if(!enabled_ || running_)
    {
        return;
    }

#ifndef GO2_CAMERA_HAS_REALSENSE
    spdlog::warn("go2_camera built without librealsense2. Camera observation will stay zero.");
    return;
#else
    running_ = true;
    worker_ = std::thread(&RealSenseSource::capture_loop, this);
#endif
}

void RealSenseSource::stop()
{
    running_ = false;
    if(worker_.joinable())
    {
        worker_.join();
    }
}

std::vector<float> RealSenseSource::latest_features() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_features_;
}

RealSenseSource::DepthFrame RealSenseSource::latest_depth_frame() const
{
    std::lock_guard<std::mutex> lock(mutex_);
    return latest_depth_frame_;
}

void RealSenseSource::capture_loop()
{
#ifdef GO2_CAMERA_HAS_REALSENSE
    try
    {
        rs2::pipeline pipe;
        rs2::config cfg;
        if(!serial_.empty())
        {
            cfg.enable_device(serial_);
        }
        cfg.enable_stream(RS2_STREAM_DEPTH, width_, height_, RS2_FORMAT_Z16, fps_);
        pipe.start(cfg);
        spdlog::info("RealSense started for go2_camera: {}x{} @ {} FPS", width_, height_, fps_);

        while(running_)
        {
            rs2::frameset frames;
            if(!pipe.poll_for_frames(&frames))
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(frame_timeout_ms_));
                continue;
            }

            auto depth = frames.get_depth_frame();
            if(!depth)
            {
                continue;
            }

            DepthFrame depth_frame;
            depth_frame.width = depth.get_width();
            depth_frame.height = depth.get_height();

            const auto* raw_data = static_cast<const uint16_t*>(depth.get_data());
            const size_t num_pixels = static_cast<size_t>(depth_frame.width) * static_cast<size_t>(depth_frame.height);
            depth_frame.data.assign(raw_data, raw_data + num_pixels);

            std::vector<float> features(feature_dim(), 0.0f);
            const int tile_h = std::max(1, height_ / grid_rows_);
            const int tile_w = std::max(1, width_ / grid_cols_);
            const float depth_range = std::max(1e-6f, max_depth_m_ - min_depth_m_);

            for(int r = 0; r < grid_rows_; ++r)
            {
                for(int c = 0; c < grid_cols_; ++c)
                {
                    const int py = std::min(height_ - 1, r * tile_h + tile_h / 2);
                    const int px = std::min(width_ - 1, c * tile_w + tile_w / 2);
                    float d = depth.get_distance(px, py);

                    if(!(d > 0.0f))
                    {
                        features[r * grid_cols_ + c] = 0.0f;
                        continue;
                    }

                    d = std::clamp(d, min_depth_m_, max_depth_m_);
                    const float normalized = (d - min_depth_m_) / depth_range;
                    features[r * grid_cols_ + c] = 1.0f - normalized;
                }
            }

            {
                std::lock_guard<std::mutex> lock(mutex_);
                latest_features_ = std::move(features);
                latest_depth_frame_ = std::move(depth_frame);
            }
        }

        pipe.stop();
    }
    catch(const std::exception& e)
    {
        spdlog::error("RealSense capture loop terminated: {}", e.what());
        std::lock_guard<std::mutex> lock(mutex_);
        std::fill(latest_features_.begin(), latest_features_.end(), 0.0f);
        latest_depth_frame_ = DepthFrame{};
        running_ = false;
    }
#endif
}

}  // namespace go2_camera
