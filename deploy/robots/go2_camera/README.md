# go2_camera

`go2_camera` is a deploy target for Go2 that augments the deploy-time observation
vector with RealSense depth features.

## What it does

- keeps the existing Go2 low-level control flow
- adds a `realsense_depth_features` observation term at deploy time
- reads a RealSense depth stream and compresses it into an `grid_rows x grid_cols`
  feature vector using center-sampled depth values
- can optionally display the latest depth frame in a desktop window during deploy
- falls back to a zero vector if `librealsense2` is not installed

## Important constraint

The deploy observation layout must match the policy that was trained/exported.
If you add `realsense_depth_features` in `params/deploy.yaml`, you must train and
export a policy with the same observation term and dimension.

## Build

```bash
cd deploy/robots/go2_camera
mkdir -p build && cd build
cmake .. && make
```

If `librealsense2` is found, camera support is enabled automatically.
If not, the binary still builds, but camera observations will remain zeros.

If OpenCV is found, the controller can also show a depth preview window.
If not, the binary still builds, but no preview window is created.

## Run

```bash
cd deploy/robots/go2_camera/build
./go2_camera_ctrl --network=enp5s0
```

The preview window is controlled from `params/deploy.yaml`:

```yaml
camera:
  show_window: true
  window_name: "Go2 RealSense Depth"
```

If you only want to test the RealSense stream without running the policy/controller:

```bash
cd deploy/robots/go2_camera/build
./go2_camera_viewer --config ../config/policy/velocity_camera/v0/params/deploy.yaml
```
