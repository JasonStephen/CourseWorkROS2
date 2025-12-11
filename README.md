ros2 launch turtlebot3_gazebo cwk_world.launch.py

ros2 launch turtlebot3_navigation2 navigation2.launch.py map:=/home/ros2_ws/src/maps/cwk_map.yaml use_sim_time:=true

colcon build --symlink-install

source install/setup.bash

ros2 run nav_package nav_node

ros2 run eye_package eye_node

Repo: glpat-A7psfYDkeJr-9sFqqShB1G86MQp1Ojhkdgk.01.0z0947jgc
glft-c2Z4kCCL7nHPE-sDshrL
