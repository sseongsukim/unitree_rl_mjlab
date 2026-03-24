#include "FSM/State_RLBase.h"
#include "RealSenseSource.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "unitree_articulation.h"

#include <algorithm>
#include <memory>

#ifdef GO2_CAMERA_HAS_OPENCV
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#endif

namespace
{
std::shared_ptr<go2_camera::RealSenseSource> g_realsense;
bool g_show_camera_window = true;
std::string g_camera_window_name = "Go2 RealSense Depth";

#ifdef GO2_CAMERA_HAS_OPENCV
void update_camera_window()
{
    if(!g_show_camera_window || !g_realsense)
    {
        return;
    }

    auto frame = g_realsense->latest_depth_frame();
    if(!frame.valid())
    {
        cv::waitKey(1);
        return;
    }

    cv::Mat depth_raw(frame.height, frame.width, CV_16UC1, frame.data.data());
    cv::Mat depth_raw_copy = depth_raw.clone();

    double min_value = 0.0;
    double max_value = 0.0;
    cv::minMaxLoc(depth_raw_copy, &min_value, &max_value);
    if(max_value <= 0.0)
    {
        cv::waitKey(1);
        return;
    }

    cv::Mat depth_u8;
    depth_raw_copy.convertTo(depth_u8, CV_8UC1, 255.0 / max_value);

    cv::Mat depth_colormap;
    cv::applyColorMap(depth_u8, depth_colormap, cv::COLORMAP_JET);

    cv::imshow(g_camera_window_name, depth_colormap);
    cv::waitKey(1);
}
#else
void update_camera_window()
{
}
#endif
}

namespace isaaclab
{
namespace mdp
{

REGISTER_OBSERVATION(realsense_depth_features)
{
    const int feature_dim = params["feature_dim"].as<int>(64);
    if(!g_realsense)
    {
        return std::vector<float>(feature_dim, 0.0f);
    }

    auto features = g_realsense->latest_features();
    if(static_cast<int>(features.size()) < feature_dim)
    {
        features.resize(feature_dim, 0.0f);
    }
    else if(static_cast<int>(features.size()) > feature_dim)
    {
        features.resize(feature_dim);
    }
    return features;
}

REGISTER_OBSERVATION(realsense_depth_valid)
{
    if(!g_realsense)
    {
        return std::vector<float>{0.0f};
    }

    auto features = g_realsense->latest_features();
    const bool any_valid = std::any_of(
        features.begin(),
        features.end(),
        [](float v) { return v > 0.0f; }
    );
    return std::vector<float>{any_valid ? 1.0f : 0.0f};
}

}  // namespace mdp
}  // namespace isaaclab

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string)
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");

    g_realsense = std::make_shared<go2_camera::RealSenseSource>(deploy_cfg["camera"]);
    g_realsense->start();
    g_show_camera_window = deploy_cfg["camera"]["show_window"].as<bool>(true);
    g_camera_window_name = deploy_cfg["camera"]["window_name"].as<std::string>("Go2 RealSense Depth");

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    this->registered_checks.emplace_back(
        std::make_pair(
            [&]()->bool{ return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    update_camera_window();

    auto action = env->action_manager->processed_actions();
    for(int i(0); i < env->robot->data.joint_ids_map.size(); i++) {
        lowcmd->msg_.motor_cmd()[env->robot->data.joint_ids_map[i]].q() = action[i];
    }
}
