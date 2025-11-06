import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import math


class Minitask3Smooth(Node):
    def __init__(self):
        super().__init__('minitask3_smooth_left_follow')

        # LIDAR angles (left-side wall following)
        # 0° front, 60° front-left, 90° left, 120° back-left
        self.front = 0.0
        self.left_front = 0.0
        self.left = 0.0
        self.left_back = 0.0

        # === Parameter settings ===
        self.wall_follow_dist = 0.3   # Ideal distance to left wall
        self.buffer = 0.1             # Distance buffer zone
        self.angle_bias = 0.03        # Geometric compensation
        self.angle_deadzone = 0.05    # Dead zone to prevent oscillation
        self.state = 'follow_wall'    # Start directly with wall-following

        # === ROS interfaces ===
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        """Read LIDAR data from key angles"""
        self.front = msg.ranges[0]         # Front 0°
        self.left_front = msg.ranges[60]   # Front-left 60°
        self.left = msg.ranges[90]         # Left 90°
        self.left_back = msg.ranges[120]   # Back-left 120°

    def timer_callback(self):
        twist = Twist()

        # Compute errors
        dist_error = self.wall_follow_dist - self.left
        # Add small bias to cancel natural negative offset
        angle_error = (self.left_front - self.left_back) + self.angle_bias

        # Debug output
        self.get_logger().info(
            f"L:{self.left:.2f} | LF:{self.left_front:.2f} | LB:{self.left_back:.2f} | "
            f"dist_err:{dist_error:.3f} | ang_err:{angle_error:.3f}"
        )

        # Base motion
        twist.linear.x = 0.20
        twist.angular.z = 0.0
        speed = 0.02

        # --- Distance control ---
        if self.left < (self.wall_follow_dist - self.buffer):
            twist.linear.x = speed
            twist.angular.z = -0.25
            self.get_logger().info("Too close to left wall → smooth right turn")

        elif self.left > (self.wall_follow_dist + self.buffer):
            twist.linear.x = speed
            twist.angular.z = 0.25
            self.get_logger().info("Too far from left wall → smooth left turn")

        # --- Angle correction (with deadzone) ---
        elif angle_error > self.angle_deadzone:
            twist.linear.x = speed
            twist.angular.z = -0.15
            self.get_logger().info("Inner corner → smooth right turn")

        elif angle_error < -self.angle_deadzone:
            twist.linear.x = speed
            twist.angular.z = 0.15
            self.get_logger().info("Outer corner → smooth left turn")

        # --- Stable & parallel ---
        else:
            twist.linear.x = 0.15
            # proportional angular correction with small gain
            twist.angular.z = -0.8 * dist_error + 0.1 * angle_error
            self.get_logger().info("Parallel to wall → steady forward")

        # Publish velocity command
        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = Minitask3Smooth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
