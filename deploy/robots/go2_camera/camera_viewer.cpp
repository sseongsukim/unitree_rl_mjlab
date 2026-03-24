#include "RealSenseSource.h"

#include <chrono>
#include <iostream>
#include <thread>

#include <boost/program_options.hpp>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#ifdef GO2_CAMERA_HAS_OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

int main(int argc, char** argv)
{
    namespace po = boost::program_options;

    po::options_description desc("go2_camera_viewer options");
    desc.add_options()
        ("help,h", "show help")
        ("config,c", po::value<std::string>()->default_value("config/policy/velocity_camera/v0/params/deploy.yaml"),
         "path to deploy yaml");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if(vm.count("help"))
    {
        std::cout << desc << std::endl;
        return 0;
    }

#ifndef GO2_CAMERA_HAS_REALSENSE
    spdlog::error("go2_camera_viewer was built without librealsense2.");
    return 1;
#elif !defined(GO2_CAMERA_HAS_OPENCV)
    spdlog::error("go2_camera_viewer was built without OpenCV highgui/imgproc support.");
    return 1;
#else
    const auto config_path = vm["config"].as<std::string>();
    auto deploy_cfg = YAML::LoadFile(config_path);
    auto camera_cfg = deploy_cfg["camera"];

    go2_camera::RealSenseSource realsense(camera_cfg);
    realsense.start();

    const std::string window_name = camera_cfg["window_name"].as<std::string>("Go2 RealSense Depth");
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    spdlog::info("Starting RealSense viewer with config: {}", config_path);
    spdlog::info("Press 'q' in the preview window to quit.");

    while(true)
    {
        auto frame = realsense.latest_depth_frame();
        if(frame.valid())
        {
            cv::Mat depth_raw(frame.height, frame.width, CV_16UC1, frame.data.data());
            cv::Mat depth_raw_copy = depth_raw.clone();

            double max_value = 0.0;
            cv::minMaxLoc(depth_raw_copy, nullptr, &max_value);
            if(max_value > 0.0)
            {
                cv::Mat depth_u8;
                depth_raw_copy.convertTo(depth_u8, CV_8UC1, 255.0 / max_value);

                cv::Mat depth_colormap;
                cv::applyColorMap(depth_u8, depth_colormap, cv::COLORMAP_JET);
                cv::imshow(window_name, depth_colormap);
            }
        }

        const int key = cv::waitKey(1);
        if(key == 'q' || key == 'Q')
        {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    realsense.stop();
    cv::destroyWindow(window_name);
    return 0;
#endif
}
