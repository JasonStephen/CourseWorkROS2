import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
import numpy as np
import time


class EightDirSmartGap(Node):
    def __init__(self):
        super().__init__('eight_dir_smart_gap_escape')

        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.timer = self.create_timer(0.1, self.control_loop)

        self.ranges = [float('inf')] * 360
        self.safe_dist = 0.5
        self.critical_dist = 0.3
        self.slow_dist = 1.0
        self.turn_prev = 0.0
        self.in_brake_zone = False

        # 脱困控制参数
        self.brake_start_time = None
        self.in_escape_mode = False
        self.escape_direction = 0.0
        self.escape_start_time = 0.0
        self.escape_duration = 1.5  # 随机旋转持续时间 (s)

        self.get_logger().info("✅ EightDirSmartGap (带脱困扫描) 节点已启动")

    def scan_callback(self, msg):
        self.ranges = msg.ranges

    def get_range(self, deg):
        idx = int(deg) % 360
        val = self.ranges[idx]
        if np.isnan(val) or np.isinf(val):
            val = 3.5
        return val

    def control_loop(self):
        vel = Twist()

        # === 1. 获取八方向距离 ===
        front = self.get_range(0)
        front_right = self.get_range(315)
        right = self.get_range(270)
        back_right = self.get_range(225)
        back = self.get_range(180)
        back_left = self.get_range(135)
        left = self.get_range(90)
        front_left = self.get_range(45)

        # === 2. 线速度控制 ===
        if front < self.critical_dist:
            vel.linear.x = 0.0
        elif front < self.safe_dist:
            vel.linear.x = max(0.0, (front - self.critical_dist) / (self.safe_dist - self.critical_dist) * 0.05)
        else:
            if front < self.slow_dist:
                vel.linear.x = 0.1 * (front - self.safe_dist) / (self.slow_dist - self.safe_dist)
                vel.linear.x = max(0.05, vel.linear.x)
            else:
                vel.linear.x = 0.15

        # === 3. 紧急制动滞后机制 ===
        enter_brake = 0.3
        exit_brake = 0.4

        if front < enter_brake:
            self.in_brake_zone = True
        elif front > exit_brake:
            self.in_brake_zone = False

        emergency_brake = self.in_brake_zone

        # === 4. 脱困逻辑触发检测 ===
        now = time.time()

        if emergency_brake:
            # 第一次进入紧急模式，记录时间
            if self.brake_start_time is None:
                self.brake_start_time = now

            # 若持续 3 秒仍在紧急状态，进入随机旋转模式
            if not self.in_escape_mode and now - self.brake_start_time > 3.0:
                self.in_escape_mode = True
                self.escape_start_time = now
                self.escape_direction = np.random.choice([-1.0, 1.0])  # -1右转, 1左转
                self.get_logger().warn("⚠️ 启动脱困旋转: 随机方向 = %s" %
                                       ("左" if self.escape_direction > 0 else "右"))

        else:
            # 离开紧急模式，清空计时
            self.brake_start_time = None
            self.in_escape_mode = False

        # === 5. 执行脱困行为 ===
        if self.in_escape_mode:
            vel.linear.x = 0.0
            vel.angular.z = 0.8 * self.escape_direction

            # 如果旋转时间超过设定值，结束脱困模式
            if now - self.escape_start_time > self.escape_duration:
                self.in_escape_mode = False
                self.get_logger().info("✅ 脱困扫描完成，恢复正常避障。")

            self.pub.publish(vel)
            return  # 跳过其余逻辑，专注于扫描

        # === 6. 紧急制动保护（稳定版） ===
        if emergency_brake:
            vel.linear.x = 0.0
            diff = right - left

            if abs(diff) < 0.05:
                diff = np.random.choice([-0.1, 0.1])

            target_turn = -0.6 * np.sign(diff)
            vel.angular.z = 0.7 * self.turn_prev + 0.3 * target_turn
            self.turn_prev = vel.angular.z

        # === 7. 安全距离保护 ===
        if not emergency_brake:
            if left < self.safe_dist:
                safety_factor = min(1.0, (self.safe_dist - left) / self.safe_dist)
                vel.angular.z -= 0.5 * safety_factor
            if right < self.safe_dist:
                safety_factor = min(1.0, (self.safe_dist - right) / self.safe_dist)
                vel.angular.z += 0.5 * safety_factor
            if front_left < self.safe_dist:
                safety_factor = min(1.0, (self.safe_dist - front_left) / self.safe_dist)
                vel.angular.z -= 0.3 * safety_factor
            if front_right < self.safe_dist:
                safety_factor = min(1.0, (self.safe_dist - front_right) / self.safe_dist)
                vel.angular.z += 0.3 * safety_factor

        # === 8. 趋宽避窄策略 ===
        if not emergency_brake and front < self.slow_dist:
            max_range = 1.5
            left_wider = right_wider = False
            if front_left < max_range and front_right < max_range:
                if front_left > front_right:
                    left_wider = True
                elif front_right > front_left:
                    right_wider = True
            elif front_left >= max_range and front_right < max_range:
                left_wider = True
            elif front_right >= max_range and front_left < max_range:
                right_wider = True

            if left_wider:
                vel.angular.z += 0.1
            elif right_wider:
                vel.angular.z -= 0.1

            vel.linear.x = min(vel.linear.x, 0.08)

        # === 9. 基础左右差控制 ===
        if not emergency_brake:
            diff_lr = right - left
            diff_frfl = front_right - front_left
            if abs(diff_lr) < 0.1:
                diff_lr = 0.0
            if abs(diff_frfl) < 0.1:
                diff_frfl = 0.0

            k1 = 0.4
            k2 = 0.2
            base_turn = - (k1 * diff_lr + k2 * diff_frfl)
            vel.angular.z += base_turn

            vel.angular.z = 0.7 * self.turn_prev + 0.3 * vel.angular.z
            self.turn_prev = vel.angular.z

        # === 10. 限制角速度范围 ===
        vel.angular.z = max(min(vel.angular.z, 1.0), -1.0)

        # === 11. 发布命令 ===
        self.pub.publish(vel)

        # 调试输出
        self.get_logger().info(
            f"Front={front:.2f}, FL={front_left:.2f}, FR={front_right:.2f}, "
            f"L={left:.2f}, R={right:.2f}, v={vel.linear.x:.2f}, turn={vel.angular.z:.2f}, "
            f"Emergency={emergency_brake}, Escape={self.in_escape_mode}"
        )


def main(args=None):
    rclpy.init(args=args)
    node = EightDirSmartGap()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
