from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nav_package',
            executable='nav_node',
            name='nav_node',
            output='screen'
        ),
        Node(
            package='eye_package',
            executable='eye_node',
            name='eye_node',
            output='screen'
        ),
    ])
