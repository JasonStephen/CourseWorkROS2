import math
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

from nav2_msgs.action import NavigateToPose
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist, Point


# ==============================
# WAYPOINTS
# ==============================
WAYPOINTS = [
    (3.42, 0.30, 0.0),
    (0.29, 2.12, 0.0),
    (-1.10, 4.02, 0.0),
    (-1.80, 2.07, 0.0),
    (-0.63, 3.99, 0.0)
]


# ==============================
# MAIN NAV NODE
# ==============================
class NavNode(Node):
    def __init__(self):
        super().__init__("nav_node")

        # Nav2
        self._client = ActionClient(self, NavigateToPose, "/navigate_to_pose")
        self._goal_handle = None

        # Robot pose
        self.robot_x = None
        self.robot_y = None

        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.pose_callback,
            10
        )

        # Green detection
        self.green_x = None
        self.green_y = None
        self.green_area = None
        self.green_last = None

        self.create_subscription(
            Point,
            "/green_position",
            self.green_callback,
            10
        )

        self.cmd_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Flags
        self.beacon_used = False
        self.is_rotating = False
        self.in_beacon = False

        self.get_logger().info("Waiting for Nav2 action server...")
        self._client.wait_for_server()
        self.get_logger().info("Nav2 is ready!")

        time.sleep(1.0)
        self.navigate_all()

    # ======================================================
    # CALLBACKS
    # ======================================================

    def pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

    def green_callback(self, msg):
        self.green_last = time.time()
        self.green_x = msg.x
        self.green_y = msg.y
        self.green_area = msg.z

    # ======================================================
    # MAIN NAV LOOP
    # ======================================================

    def navigate_all(self):

        for idx, (x, y, _) in enumerate(WAYPOINTS):
            self.get_logger().info(f"--- Waypoint {idx+1} ---")

            self.send_goal(x, y)
            self.wait_reached(x, y)

            # Nav2 和旋转可能冲突，先取消
            self._cancel_current_nav()

            # 转圈寻找绿色物体
            if not self.beacon_used:
                self.rotate_360_and_search()

            # 旋转结束后再检查一次绿色（area > 60000）
            if not self.beacon_used:
                self.check_green_after_rotate()

        self.get_logger().info("All waypoints finished.")

    # ======================================================
    # NAVIGATION SEND / CANCEL
    # ======================================================

    def send_goal(self, x, y):
        goal = NavigateToPose.Goal()
        pose = PoseStamped()
        pose.header.frame_id = "map"
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.pose.position.x = x
        pose.pose.position.y = y
        pose.pose.orientation.w = 1.0

        goal.pose = pose

        future = self._client.send_goal_async(goal)
        future.add_done_callback(self._goal_callback)

    def _goal_callback(self, future):
        self._goal_handle = future.result()

    def _cancel_current_nav(self):
        if not self._goal_handle:
            return

        self.get_logger().info("Canceling Nav2 goal...")
        cancel_future = self._goal_handle.cancel_goal_async()
        rclpy.spin_until_future_complete(self, cancel_future)

        self._goal_handle = None

        # Stop any leftover velocity
        self.cmd_pub.publish(Twist())
        time.sleep(0.2)

    # ======================================================
    # WAIT UNTIL REACHED
    # ======================================================

    def wait_reached(self, gx, gy):
        tolerance = 0.03
        last = None
        steady = None

        while rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.robot_x is None:
                continue

            dist = math.dist([self.robot_x, self.robot_y], [gx, gy])
            if dist < tolerance:
                return

            # stuck detection
            if last is not None and abs(dist - last) < 0.008:
                if steady is None:
                    steady = time.time()
                elif time.time() - steady > 4:
                    return
            else:
                steady = None

            last = dist

    # ======================================================
    # ROTATE 360° (only trigger beacon when area > 60000)
    # ======================================================

    def rotate_360_and_search(self):
        self.get_logger().info("Rotating to search for green...")
        self.is_rotating = True

        start = time.time()
        DURATION = 6.0

        twist = Twist()
        twist.angular.z = 1.0

        while time.time() - start < DURATION:
            rclpy.spin_once(self, timeout_sec=0.05)

            # New condition: must have green_area > 60000
            if (
                not self.beacon_used and
                self.green_last and
                time.time() - self.green_last < 0.25 and
                self.green_area is not None and
                self.green_area > 60000
            ):
                self.get_logger().info(
                    f"Green detected during rotation (area={self.green_area:.0f}) → Beacon"
                )

                # Stop rotation
                self.cmd_pub.publish(Twist())
                time.sleep(0.15)

                self.is_rotating = False
                self.beacon()
                self.beacon_used = True
                return

            self.cmd_pub.publish(twist)

        self.cmd_pub.publish(Twist())
        time.sleep(0.15)

        self.is_rotating = False
        self.get_logger().info("Rotation finished.")

    # ======================================================
    # POST-ROTATE GREEN CHECK (already area > 60000)
    # ======================================================

    def check_green_after_rotate(self):
        start = time.time()

        while time.time() - start < 2.0:
            rclpy.spin_once(self, timeout_sec=0.05)

            if (
                self.green_last and
                time.time() - self.green_last < 0.6 and
                self.green_area > 60000
            ):
                self.get_logger().info(
                    f"Green detected after rotate (area={self.green_area:.0f}) → Beacon"
                )
                self.beacon()
                self.beacon_used = True
                return

    # ======================================================
    # BEACON — APPROACH TRASH BIN
    # ======================================================

    def beacon(self):
        self.get_logger().info("=== Beacon START ===")
        self.in_beacon = True

        START = time.time()
        LIMIT = 30.0

        lost_frames = 0
        start_x = self.robot_x
        start_y = self.robot_y

        while rclpy.ok() and time.time() - START < LIMIT:

            rclpy.spin_once(self, timeout_sec=0.03)

            # Green lost temporarily
            if time.time() - self.green_last > 0.4:
                lost_frames += 1
                self.cmd_pub.publish(Twist())
                if lost_frames > 20:
                    self.get_logger().info("Target lost → exiting beacon")
                    break
                continue
            else:
                lost_frames = 0

            gx = self.green_x
            area = self.green_area

            twist = Twist()

            # Angular control
            turn = -0.6 * gx
            turn = max(-0.4, min(0.4, turn))
            twist.angular.z = turn

            # ===== COMMENTED OUT: Distance-based exit =====
            # moved = math.dist([self.robot_x, self.robot_y], [start_x, start_y])
            # if moved > 0.40:
            #     self.get_logger().info("Max approach range reached → stop")
            #     break

            # Approach based on area only
            if area < 60000:
                twist.linear.x = 0.05
            elif area < 250000:
                twist.linear.x = 0.03
            else:
                self.get_logger().info("Reached object → stopping")
                break

            self.cmd_pub.publish(twist)

        self.cmd_pub.publish(Twist())
        time.sleep(0.3)

        self.in_beacon = False
        self.get_logger().info("=== Beacon END ===")


# ======================================================
# MAIN
# ======================================================
def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
