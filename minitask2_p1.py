import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan



class Minitask3Smooth(Node):
    def __init__(self):
        super().__init__('minitask3_smooth')

        # 激光雷达角度：0°前方，60°右前，90°右，120°右后
        self.front = 0.0
        self.right_front = 0.0
        self.right = 0.0
        self.right_back = 0.0

        # 参数设置
        self.safe_front = 0.45        # 前方安全距离
        self.wall_follow_dist = 0.5   # 理想贴边距离
        self.buffer = 0.45             # 距离缓冲区
        self.clear_threshold = 1.0    # 前方畅通判定
        self.state = 'forward'        # 初始状态

        # ROS接口
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        self.subscription = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        """读取关键角度传感器数据"""
        self.front = msg.ranges[0]         # 前 0°
        self.right_front = msg.ranges[300] # 右前 60°
        self.right = msg.ranges[270]       # 右 90°
        self.right_back = msg.ranges[240]  # 右后 120°

    def timer_callback(self):
        twist = Twist()

        # 调试输出
        self.get_logger().info(
            f"State: {self.state} | F:{self.front:.2f} | RF:{self.right_front:.2f} | R:{self.right:.2f} | RB:{self.right_back:.2f}"
        )

        # ========== 状态机逻辑 ==========
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

            # 前方障碍 → 左转避障
            if 0.0 < self.front < self.safe_front:
                self.state = 'turn_left'
                self.get_logger().warn("Obstacle ahead during wall follow → turning left")
                return

            # 默认前进速度
            twist.linear.x = 0.22
            twist.angular.z = 0.0


            # 右墙太近 → 缓慢左转
            speed = 0.02
            
            if self.right < (self.wall_follow_dist - self.buffer):
                twist.linear.x = speed
                twist.angular.z = 0.25
                self.get_logger().info("Too close to right wall → smooth left turn")

            # 右墙太远 → 缓慢右转
            elif self.right > (self.wall_follow_dist + self.buffer):
                twist.linear.x = speed
                twist.angular.z = -0.25
                self.get_logger().info("Too far from right wall → smooth right turn")

            # 外角 → 缓慢右转靠墙
            elif angle_error > 0.02:
                twist.linear.x = speed
                twist.angular.z = -0.15
                self.get_logger().info("Outer corner → smooth right turn")

            # 内角 → 缓慢左转离墙
            elif angle_error < -0.02:
                twist.linear.x = speed
                twist.angular.z = 0.15
                self.get_logger().info("Inner corner → smooth left turn")

            # 平行 → 稳定直行
            else:
                twist.linear.x = 0.15
                twist.angular.z = dist_error * 0.8

        # 发布控制指令
        self.publisher.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = Minitask3Smooth()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
