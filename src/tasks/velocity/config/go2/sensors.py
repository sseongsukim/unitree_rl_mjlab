from __future__ import annotations

from dataclasses import dataclass
from math import cos, sin

from mjlab.sensor import CameraSensorCfg


@dataclass(frozen=True)
class ForwardCameraSpec:
    """Go2 forward camera spec kept local to the task config.

    Notes:
    - ``position``/``rotation`` retain the original richer schema so we can extend
      this later without touching mjlab internals.
    - Current CameraSensorCfg supports a fixed pose only, so this helper uses the
      mean position and the midpoint of the Euler rotation range.
    - Cropping, resizing, and latency fields are stored here for future task-side
      processing, but are not applied by CameraSensorCfg directly today.
    """

    obs_components: tuple[str, ...] = ("forward_depth",)
    resolution: tuple[int, int] = (120, 160)
    position_mean: tuple[float, float, float] = (0.24, -0.0175, 0.12)
    position_std: tuple[float, float, float] = (0.01, 0.0025, 0.03)
    rotation_lower: tuple[float, float, float] = (-0.1, 0.37, -0.1)
    rotation_upper: tuple[float, float, float] = (0.1, 0.43, 0.1)
    resized_resolution: tuple[int, int] = (48, 64)
    output_resolution: tuple[int, int] = (48, 64)
    horizontal_fov: tuple[float, float] = (86.0, 90.0)
    crop_top_bottom: tuple[int, int] = (12, 0)
    crop_left_right: tuple[int, int] = (7, 9)
    near_plane: float = 0.15  # <- 0.05
    depth_range: tuple[float, float] = (0.15, 5.0)  # <- (0.0, 3.0)
    latency_range: tuple[float, float] = (0.08, 0.142)
    latency_resampling_time: float = 5.0
    refresh_duration: float = 0.1

    # noise
    stereo_min_distance = 0.175  # when using (480, 640) resolution
    stereo_far_distance = 3.0  # <- 1.2
    stereo_far_noise_std = 0.08
    stereo_near_noise_std = 0.02
    stereo_full_block_artifacts_prob = 0.008
    stereo_full_block_values = [0.0, 0.25, 0.5, 1.0, 3.0]
    stereo_full_block_height_mean_std = [62, 1.5]
    stereo_full_block_width_mean_std = [3, 0.01]
    stereo_half_block_spark_prob = 0.02
    stereo_half_block_value = 3000
    sky_artifacts_prob = 0.0001
    sky_artifacts_far_distance = 2.0
    sky_artifacts_values = [0.6, 1.0, 1.2, 1.5, 1.8]
    sky_artifacts_height_mean_std = [2, 3.2]
    sky_artifacts_width_mean_std = [2, 3.2]


DEFAULT_FORWARD_CAMERA_SPEC = ForwardCameraSpec()


def _quat_from_euler_xyz(
    roll: float, pitch: float, yaw: float
) -> tuple[float, float, float, float]:
    """Convert XYZ Euler angles to MuJoCo quaternion order (w, x, y, z)."""
    cr = cos(roll * 0.5)
    sr = sin(roll * 0.5)
    cp = cos(pitch * 0.5)
    sp = sin(pitch * 0.5)
    cy = cos(yaw * 0.5)
    sy = sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def make_forward_camera_sensor_cfg(
    spec: ForwardCameraSpec = DEFAULT_FORWARD_CAMERA_SPEC,
    name: str = "forward_camera",
    parent_body: str = "robot/base_link",
) -> CameraSensorCfg:
    """Create a CameraSensorCfg for the Go2 forward depth camera."""
    mid_rotation = tuple(
        0.5 * (lo + hi) for lo, hi in zip(spec.rotation_lower, spec.rotation_upper)
    )
    quat = _quat_from_euler_xyz(*mid_rotation)
    fovy = 0.5 * (spec.horizontal_fov[0] + spec.horizontal_fov[1])
    return CameraSensorCfg(
        name=name,
        parent_body=parent_body,
        pos=spec.position_mean,
        quat=quat,
        fovy=fovy,
        width=spec.resolution[1],
        height=spec.resolution[0],
        data_types=("depth",),
        use_textures=False,
        use_shadows=False,
        enabled_geom_groups=(0, 1, 2),
    )
