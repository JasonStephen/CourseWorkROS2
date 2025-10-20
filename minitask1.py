import rclpy
from rclpy.node import Node
import math
from geometry_msgs.msg import Twist, Vector3
from nav_msgs.msg import Odometry
from tf_transformations import euler_from_quaternion  # standard ROS utility

class Minitask1(Node):

    def __init__(self):
        super().__init__('minitask1')
        self.publisher_ = self.create_publisher(Twist, 'cmd_vel', 10)
        timer_period = 0.1  # slightly faster update for smoother feedback
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # create odometry subscriber
        self.subscription = self.create_subscription(
            Odometry,
            'odom',
            self.odom_callback,
            10)
        self.subscription  

        # --- state variables ---
        self.state = "INIT_MOVE"     # initial state: move forward briefly
        self.move_start_time = self.get_clock().now()
        self.side_count = 0          # completed sides (square = 4)
        self.position = None         # latest odom position (x, y)
        self.yaw = 0.0               # current heading (radians)

        ### NEW: store start pose for closed-loop control
        self.start_x = None
        self.start_y = None
        self.start_yaw = None

    def timer_callback(self):
        msg = Twist()

        # If odometry not yet available, wait
        if self.position is None:
            self.get_logger().info('⏳ Waiting for odometry...')
            self.publisher_.publish(msg)
            return

        now = self.get_clock().now()
        elapsed = (now - self.move_start_time).nanoseconds / 1e9

        # -------------------------------
        # Initial forward movement phase
        # -------------------------------
        if self.state == "INIT_MOVE":
            if elapsed < 1.0:
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().info('🚀 Initial forward (1.0s)...')
            else:
                self.state = "MOVE"
                self.move_start_time = now
                self.start_x, self.start_y = self.position
                self.start_yaw = self.yaw
                self.get_logger().info('✅ Initial move finished, start square loop.')

        # -------------------------------
        # Move forward using odometry feedback
        # -------------------------------
        elif self.state == "MOVE":
            # compute distance from start point
            dx = self.position[0] - self.start_x
            dy = self.position[1] - self.start_y
            dist = math.sqrt(dx**2 + dy**2)

            if dist < 1.0:  # target distance ≈ 1 m
                msg.linear.x = 0.25
                msg.angular.z = 0.0
                self.get_logger().info(f'🚗 Moving forward... {dist:.2f} m')
            else:
                self.state = "STOP"
                self.move_start_time = now
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().info(f'⏸ Reached target distance ({dist:.2f} m), stop before turning...')

        elif self.state == "STOP":
            if elapsed < 0.5:  # pause 0.5 s
                msg.linear.x = 0.0
                msg.angular.z = 0.0
            else:
                self.state = "TURN"
                self.move_start_time = now
                self.start_yaw = self.yaw
                self.get_logger().info('↩ Turning 90 degrees...')

        # -------------------------------
        # Turn using odometry feedback
        # -------------------------------
        elif self.state == "TURN":
            # compute absolute rotation since start
            yaw_diff = self._angle_diff(self.yaw, self.start_yaw)
            target_angle = math.pi / 2  # 90 degrees in radians

            if abs(yaw_diff) < target_angle:
                msg.linear.x = 0.0
                msg.angular.z = 0.25
                self.get_logger().info(f'🔄 Rotating... {math.degrees(yaw_diff):.1f}°')
            else:
                # finished one side
                self.state = "MOVE"
                self.move_start_time = now
                self.start_x, self.start_y = self.position
                self.side_count = (self.side_count + 1) % 4
                msg.linear.x = 0.0
                msg.angular.z = 0.0
                self.get_logger().info(f'✅ Finished side {self.side_count + 1}')

        # publish the message
        self.publisher_.publish(msg)

    def _angle_diff(self, current, start):
        """Return minimal signed difference between two angles."""
        diff = current - start
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi
        return diff

    def odom_callback(self, msg):
        # extract position
        pos = msg.pose.pose.position
        self.position = (pos.x, pos.y)

        # convert quaternion → Euler yaw
        q = msg.pose.pose.orientation
        quaternion = [q.x, q.y, q.z, q.w]
        _, _, yaw = euler_from_quaternion(quaternion)
        self.yaw = yaw
        # (optional debug output)
        # self.get_logger().info(f'📍 x={pos.x:.2f}, y={pos.y:.2f}, yaw={math.degrees(yaw):.1f}°')


def main(args=None):
    rclpy.init(args=args)
    node = Minitask1()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
