import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan


class Minitask3Smooth(Node):
    def __init__(self):
        super().__init__('minitask3_smooth')

        # LIDAR angles: 0° front, 60° front-right, 90° right, 120° back-right
        self.front = 0.0
        self.right_front = 0.0
        self.right = 0.0
        self.right_back = 0.0

        # Parameter settings
        self.safe_front = 0.45        # Safe distance ahead
        self.wall_follow_dist = 0.5   # Ideal wall-following distance
        self.buffer = 0.45            # Distance buffer zone
        self.clear_threshold = 1.0    # Threshold to determine clear path
        self.state = 'forward'        # Initial state

        # ROS interfaces
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        """Read LIDAR data from key angles"""
        self.front = msg.ranges[0]         # Front 0°
        self.right_front = msg.ranges[300] # Front-right 60°
        self.right = msg.ranges[270]       # Right 90°
        self.right_back = msg.ranges[240]  # Back-right 120°

    def timer_callback(self):
        twist = Twist()

        # Debug output
        self.get_logger().info(
            f"State: {self.state} | F:{self.front:.2f} | RF:{self.right_front:.2f} | R:{self.right:.2f} | RB:{self.right_back:.2f}"
        )

        # ========== State Machine Logic ==========
        if self.state == 'forward':
            if 0.0 < self.front < self.safe_front:
                self.state = 'turn_left'
                self.get_logger().warn("Obstacle ahead → switching to TURN_LEFT")
                return

            twist.linear.x = 0.25
            twist.angular.z = 0.0

        elif self.state == 'turn_left':
            twist.linear.x = 0.0
            twist.angular.z = 0.3
            self.get_logger().info("Turning left in place...")

            if self.front > self.clear_threshold:
                self.state = 'follow_wall'
                self.get_logger().info("Front clear → switching to FOLLOW_WALL")

        elif self.state == 'follow_wall':
            dist_error = self.wall_follow_dist - self.right
            angle_error = self.right_front - self.right_back

            # Obstacle ahead → turn left to avoid
            if 0.0 < self.front < self.safe_front:
                self.state = 'turn_left'
                self.get_logger().warn("Obstacle ahead during wall follow → turning left")
                return

            # Default forward speed
            twist.linear.x = 0.22
            twist.angular.z = 0.0

            # Too close to right wall → smooth left turn
            speed = 0.02
            
            if self.right < (self.wall_follow_dist - self.buffer):
                twist.linear.x = speed
                twist.angular.z = 0.25
                self.get_logger().info("Too close to right wall → smooth left turn")

            # Too far from right wall → smooth right turn
            elif self.right > (self.wall_follow_dist + self.buffer):
                twist.linear.x = speed
                twist.angular.z = -0.25
                self.get_logger().info("Too far from right wall → smooth right turn")

            # Outer corner → smooth right turn to approach wall
            elif angle_error > 0.02:
                twist.linear.x = speed
                twist.angular.z = -0.15
                self.get_logger().info("Outer corner → smooth right turn")

            # Inner corner → smooth left turn to move away from wall
            elif angle_error < -0.02:
                twist.linear.x = speed
                twist.angular.z = 0.15
                self.get_logger().info("Inner corner → smooth left turn")

            # Parallel to wall → maintain straight motion
            else:
                twist.linear.x = 0.15
                twist.angular.z = dist_error * 0.8

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
