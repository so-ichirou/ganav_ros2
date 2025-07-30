# ganav_ros2

## realsenseノード
`cd Desktop/realsense_ws`\
`source install/setup.bash`\
`ros2 launch realsense2_camera rs_launch.py`\

## 推論エンジン
`conda activate ganav_env_final`\
`cd Desktop/traversability_ws`\
`python run_inference_engine.py`\

## ros2ラッパーノード
`cd Desktop/traversability_ws/`\
`source install/setup.bash`\
`ros2 run semantic_segmentation_ros2 inference_node`\

rosbag
`ros2 bag record -o my_segmentation_bag /camera/camera/color/image_raw /segmentation/image /segmentation/overlay`
