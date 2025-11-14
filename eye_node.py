import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from geometry_msgs.msg import Point

from cv_bridge import CvBridge
import cv2
import numpy as np
import time


class GreenDetector(Node):
    def __init__(self):
        super().__init__('eye_node')

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.listener_callback,
            10
        )
        self.publisher_ = self.create_publisher(Point, '/green_position', 10)

        self.br = CvBridge()

        self.last_process = 0.0
        self.process_interval = 0.18

        self.last_pub = 0.0
        self.pub_interval = 0.18

        self.get_logger().info("eye_node started (480p, rate limited).")

    def listener_callback(self, msg):
        now = time.time()
        if now - self.last_process < self.process_interval:
            return
        self.last_process = now

        frame = self.br.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        frame = cv2.resize(frame, (640, 480))

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h, w = frame.shape[:2]

        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)

            if area > 1500:
                M = cv2.moments(largest)
                if M['m00'] != 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])

                    norm_x = (cx - w / 2) / (w / 2)
                    norm_y = (cy - h / 2) / (h / 2)

                    if now - self.last_pub >= self.pub_interval:
                        p = Point()
                        p.x = float(norm_x)
                        p.y = float(norm_y)
                        p.z = float(area)
                        self.publisher_.publish(p)
                        self.last_pub = now

        cv2.imshow("camera", frame)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = GreenDetector()
    rclpy.spin(node)
    node.destroy_node()
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
