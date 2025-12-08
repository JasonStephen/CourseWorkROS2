import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Point, PoseWithCovarianceStamped, PointStamped
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import Bool, String, Float32
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge

import cv2
import numpy as np
import math
import time
from collections import deque


class ObjectDetector(Node):

    def __init__(self):
        super().__init__('eye_node')
  
        self.declare_parameter('min_area_threshold', 3000)
        self.declare_parameter('max_area_threshold', 250000)
        self.declare_parameter('duplicate_distance', 1.5)
        self.declare_parameter('process_interval', 0.07) # ~15 FPS
        self.declare_parameter('show_debug_window', True)
        self.declare_parameter('confirm_timeout', 2.0)
        self.declare_parameter('camera_fov_horizontal', 62.2)
        
        # Interval frame rate parameters
        self.declare_parameter('min_confirm_frames', 4) # Minimum frame rate
        self.declare_parameter('max_confirm_frames', 8) # Maximum frame rate
        
        # Graded stability thresholds
        self.declare_parameter('strict_std_threshold', 0.12) # Very stable (4 frames)
        self.declare_parameter('normal_std_threshold', 0.15) # Stable (5-7 frames)
        self.declare_parameter('loose_std_threshold', 0.20) # Basically stable (up to 8 frames)
        
        # Center region thresholds
        self.declare_parameter('center_threshold', 0.15) # Center ±15%
        self.declare_parameter('slowdown_threshold', 0.20) # Slowdown zone ±20%
        
        # Prepare detection zone threshold (center 40%, i.e., ±20%)
        self.declare_parameter('prepare_zone_threshold', 0.20)
        
        # Distance measurement parameters
        self.declare_parameter('min_valid_distance', 0.3)
        self.declare_parameter('max_valid_distance', 4.0)
        
        # Get parameter values
        self.min_area = self.get_parameter('min_area_threshold').value
        self.max_area = self.get_parameter('max_area_threshold').value
        self.duplicate_distance = self.get_parameter('duplicate_distance').value
        self.process_interval = self.get_parameter('process_interval').value
        self.show_debug = self.get_parameter('show_debug_window').value
        self.confirm_timeout = self.get_parameter('confirm_timeout').value
        self.camera_fov = math.radians(self.get_parameter('camera_fov_horizontal').value)
        
        # Interval frame rate
        self.min_frames = self.get_parameter('min_confirm_frames').value
        self.max_frames = self.get_parameter('max_confirm_frames').value
        
        # Graded thresholds
        self.strict_std = self.get_parameter('strict_std_threshold').value
        self.normal_std = self.get_parameter('normal_std_threshold').value
        self.loose_std = self.get_parameter('loose_std_threshold').value
        
        self.center_threshold = self.get_parameter('center_threshold').value # Center ±15%
        self.slowdown_threshold = self.get_parameter('slowdown_threshold').value # Prepare zone
        self.min_valid_distance = self.get_parameter('min_valid_distance').value
        self.max_valid_distance = self.get_parameter('max_valid_distance').value
        
        # HSV color ranges
        # Green trash bin
        self.green_lower = np.array([35, 40, 5])
        self.green_upper = np.array([85, 255, 255])
        
        # Red fire hydrant(strict range)
        self.red_lower_1 = np.array([0, 150, 50])
        self.red_upper_1 = np.array([6, 255, 255])
        self.red_lower_2 = np.array([175, 150, 50])
        self.red_upper_2 = np.array([180, 255, 255])
        
        # Shape filtering parameters
        self.min_solidity = 0.5
        self.min_extent = 0.3
        self.max_aspect_ratio = 3.5
        self.min_aspect_ratio = 0.25
        
        # Subscribers 
        self.image_sub = self.create_subscription(
            Image, '/camera_depth/image_raw', self.image_callback, 10)
        
        self.depth_sub = self.create_subscription(
            Image, '/camera_depth/depth/image_raw', self.depth_callback, 10)
        
        self.scan_sub = self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)
        
        self.pose_sub = self.create_subscription(
            PoseWithCovarianceStamped, '/amcl_pose', self.amcl_pose_callback, 10)
        
        self.odom_sub = self.create_subscription(
            Odometry, '/odom', self.odom_callback, 10)
        
        # Publishers
        self.detection_pub = self.create_publisher(Point, '/detected_object', 10)
        self.world_pos_pub = self.create_publisher(PointStamped, '/object_world_position', 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/found_objects', 10)
        self.detected_pub = self.create_publisher(Bool, '/object_detected', 10)
        
        # /object_offset: Object offset relative to the center of the frame (-1.0 to 1.0)
        #   Positive = Object is on the right side of the frame, robot needs to turn right
        #   Negative = Object is on the left side of the frame, robot needs to turn left
        #   0 = Object is centered
        self.offset_pub = self.create_publisher(Float32, '/object_offset', 10)
        
        # /object_status: Object detection status string
        #   "none" = No object detected
        #   "left" = Object is on the left side
        #   "right" = Object is on the right side  
        #   "centered" = Object is centered
        #   "locked" = Object is centered and being confirmed
        #   "confirmed" = Object confirmed and marked
        self.status_pub = self.create_publisher(String, '/object_status', 10)
        
        # Internal state
        self.bridge = CvBridge()
        self.current_image = None
        self.depth_image = None
        
        self.scan_data = None
        self.scan_angle_min = 0.0
        self.scan_angle_max = 0.0
        self.scan_angle_increment = 0.0
        self.scan_received = False
        
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.pose_received = False
        self.pose_source = 'none'
        
        self.image_received = False
        self.depth_received = False
        self.last_image_time = 0.0
        
        # Detected objects list
        self.found_objects = []
        self.marker_id = 0
        
        # Marked zones log rate limiting
        self.last_known_log_time = {'green': 0.0, 'red': 0.0}
        self.known_log_interval = 2.0  # At most one log every 2 seconds
        
        # Multi-frame confirmation - use max_frames as deque size
        self.candidates = {
            'green': {
                'positions': deque(maxlen=self.max_frames), 
                'last_seen': 0.0, 
                'count': 0,
                'distances': deque(maxlen=self.max_frames),
            },
            'red': {
                'positions': deque(maxlen=self.max_frames), 
                'last_seen': 0.0, 
                'count': 0,
                'distances': deque(maxlen=self.max_frames),
            }
        }
        
        # Marked zones
        self.marked_zones_green = []
        self.marked_zones_red = []
        
        # Timer
        self.timer = self.create_timer(self.process_interval, self.process_frame)
        
        self.get_logger().info('='*60)
        self.get_logger().info('Eye Node v3.9')
        self.get_logger().info('='*60)
        self.get_logger().info(f'Area divisions:')
        self.get_logger().info(f'- Slowdown zone (±{self.slowdown_threshold*100:.0f}%): Send slowdown signal upon entry')
        self.get_logger().info(f'- Center zone (±{self.center_threshold*100:.0f}%): Distance measurement marker')
        self.get_logger().info(f'Frame range: {self.min_frames}-{self.max_frames} frames')
        self.get_logger().info(f'Topics: /object_offset, /object_status')
        self.get_logger().info('='*60)

    # Callback functions
    
    def image_callback(self, msg):
        try:
            self.current_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image_received = True
            self.last_image_time = time.time()
        except Exception as e:
            self.get_logger().error(f'Image conversion failed: {e}')
    
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            self.depth_received = True
        except:
            pass
    
    def scan_callback(self, msg):
        self.scan_data = np.array(msg.ranges)
        self.scan_angle_min = msg.angle_min
        self.scan_angle_max = msg.angle_max
        self.scan_angle_increment = msg.angle_increment
        self.scan_received = True
    
    def amcl_pose_callback(self, msg):
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose_received = True
        self.pose_source = 'amcl'
    
    def odom_callback(self, msg):
        if self.pose_source != 'amcl':
            self.robot_x = msg.pose.pose.position.x
            self.robot_y = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            self.robot_yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
            self.pose_received = True
            self.pose_source = 'odom'

    # Distance measurement functions
    
    def get_distance_from_lidar(self):
        # Lidar distance measurement - only measure directly ahead
        if self.scan_data is None or len(self.scan_data) == 0:
            return None
        
        target_angle = 0.0
        
        while target_angle < self.scan_angle_min:
            target_angle += 2 * math.pi
        while target_angle > self.scan_angle_max:
            target_angle -= 2 * math.pi
        
        index = int((target_angle - self.scan_angle_min) / self.scan_angle_increment)
        
        if index < 0 or index >= len(self.scan_data):
            return None
        
        window = 2
        start_idx = max(0, index - window)
        end_idx = min(len(self.scan_data), index + window + 1)
        
        distances = self.scan_data[start_idx:end_idx]
        valid_distances = distances[
            (distances > self.min_valid_distance) & 
            (distances < self.max_valid_distance) & 
            np.isfinite(distances)
        ]
        
        if len(valid_distances) == 0:
            return None
        
        return float(np.median(valid_distances))
    
    def get_depth_at_center(self, cx, cy, window_size=15):
        # Depth camera distance measurement
        if self.depth_image is None:
            return None
        
        h, w = self.depth_image.shape[:2]
        x1 = max(0, cx - window_size)
        x2 = min(w, cx + window_size)
        y1 = max(0, cy - window_size)
        y2 = min(h, cy + window_size)
        
        depth_window = self.depth_image[y1:y2, x1:x2]
        valid_depths = depth_window[
            (depth_window > self.min_valid_distance) & 
            (depth_window < self.max_valid_distance) & 
            np.isfinite(depth_window)
        ]
        
        if len(valid_depths) == 0:
            return None
        
        return float(np.median(valid_depths))

    # Core detection logic
    
    def process_frame(self):
        if self.current_image is None or not self.pose_received:
            return
        
        frame = self.current_image.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        current_time = time.time()
        h, w = frame.shape[:2]
        
        # Draw area indicator lines
        # Center zone (yellow)
        center_left = int(w * (0.5 - self.center_threshold))
        center_right = int(w * (0.5 + self.center_threshold))
        cv2.line(frame, (center_left, 0), (center_left, h), (0, 255, 255), 2)
        cv2.line(frame, (center_right, 0), (center_right, h), (0, 255, 255), 2)
        
        # Slowdown zone (orange)
        slowdown_left = int(w * (0.5 - self.slowdown_threshold))
        slowdown_right = int(w * (0.5 + self.slowdown_threshold))
        cv2.line(frame, (slowdown_left, 0), (slowdown_left, h), (0, 165, 255), 1)
        cv2.line(frame, (slowdown_right, 0), (slowdown_right, h), (0, 165, 255), 1)
        
        # Area labels
        cv2.putText(frame, 'CENTER', (center_left + 5, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(frame, 'SLOW', (slowdown_left + 2, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1)
        
        # Detect green
        green_mask = cv2.inRange(hsv, self.green_lower, self.green_upper)
        green_result = self.detect_and_track(green_mask, frame, 'green', 'Trash Can', (0, 255, 0), current_time)
        
        # Detect red
        red_mask_1 = cv2.inRange(hsv, self.red_lower_1, self.red_upper_1)
        red_mask_2 = cv2.inRange(hsv, self.red_lower_2, self.red_upper_2)
        red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
        red_result = self.detect_and_track(red_mask, frame, 'red', 'Fire Hydrant', (0, 0, 255), current_time)
        
        # If neither object is detected, publish "none" status
        if green_result is None and red_result is None:
            status_msg = String()
            status_msg.data = "none"
            self.status_pub.publish(status_msg)
            
            offset_msg = Float32()
            offset_msg.data = 0.0
            self.offset_pub.publish(offset_msg)
        
        self.cleanup_candidates(current_time)
        
        if self.show_debug:
            self.show_debug_windows(frame, green_mask, red_mask)
    
    def detect_and_track(self, mask, frame, color_type, object_name, box_color, current_time):
        # Core detection logic - interval frame confirmation version
        
        # Morphological processing
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_large = np.ones((7, 7), np.uint8)
        mask_processed = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        mask_processed = cv2.morphologyEx(mask_processed, cv2.MORPH_CLOSE, kernel_large)
        
        contours, _ = cv2.findContours(mask_processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Collect valid contours and sort by area
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area or area > self.max_area:
                continue
            if not self.check_shape(contour, area):
                continue
            valid_contours.append((contour, area))
        
        if not valid_contours:
            return None
        
        # Only process the largest object
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        largest_contour, area = valid_contours[0]
        
        M = cv2.moments(largest_contour)
        if M['m00'] == 0:
            return None
        
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        h, w = frame.shape[:2]
        norm_x = (cx - w / 2) / (w / 2)
        
        # Draw detection box
        x, y, bw, bh = cv2.boundingRect(largest_contour)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), box_color, 2)
        cv2.circle(frame, (cx, cy), 5, box_color, -1)
        
        # Area judgment: slowdown zone (±20%) includes center zone (±15%)
        abs_norm_x = abs(norm_x)
        is_in_slowdown = abs_norm_x < self.slowdown_threshold      # Slowdown zone (±20%)
        is_centered = abs_norm_x < self.center_threshold           # Center zone (±15%)
        
        # Situation 1: Not in slowdown zone (>±20%) - send left or right direction signal
        if not is_in_slowdown:
            direction = "←Left" if norm_x > 0 else "Right→"
            cv2.putText(frame, f'{object_name} [{direction}]', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            cv2.putText(frame, f'Offset: {norm_x:.2f}', 
                       (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
            
            # Real-time log: left or right direction indication
            direction_log = "On the right, need to turn right" if norm_x > 0 else "On the left, need to turn left"
            self.get_logger().info(
                f'[{object_name}] {direction_log} (Offset: {norm_x:.2f})'
            )
            
            # Publish status: left or right
            offset_msg = Float32()
            offset_msg.data = float(norm_x)
            self.offset_pub.publish(offset_msg)
            
            status_msg = String()
            status_msg.data = "right" if norm_x > 0 else "left"
            self.status_pub.publish(status_msg)
            
            return {'status': 'outside', 'offset': norm_x}
        
        # Situation 2: In the slowdown zone (±20%) - send slowdown signal
        # Publish slowdown status (real-time)
        offset_msg = Float32()
        offset_msg.data = float(norm_x)
        self.offset_pub.publish(offset_msg)
        
        status_msg = String()
        status_msg.data = "slowdown"
        self.status_pub.publish(status_msg)
        
        # Real-time log: In slowdown zone
        self.get_logger().info(
            f'[{object_name}] In slowdown zone (offset: {norm_x:.2f})'
        )
        
        # Situation 2a: In the slowdown zone but not in the center zone (±15%~±20%) - slowdown only, no distance measurement
        if not is_centered:
            cv2.putText(frame, f'{object_name} [SLOWDOWN]', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)
            cv2.putText(frame, f'Offset: {norm_x:.2f}', 
                       (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
            
            return {'status': 'slowdown', 'offset': norm_x}
        
        # Situation 2b: In the center zone (±15%) - slowdown + distance measurement
        # Centered: perform distance measurement
        distance = self.get_centered_distance(cx, cy)
        
        if distance is None:
            cv2.putText(frame, f'{object_name} [SLOWDOWN-NO DIST]', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
            return {'status': 'no_distance'}
        
        # Calculate world coordinates
        angle_offset = norm_x * (self.camera_fov / 2)
        world_x, world_y = self.calculate_world_position(angle_offset, distance)
        
        # Check for duplicates
        if self.is_in_marked_zone(world_x, world_y, color_type):
            cv2.putText(frame, f'{object_name} [KNOWN]', 
                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 2)
            # Rate-limited log output (at most once every 2 seconds)
            if current_time - self.last_known_log_time[color_type] >= self.known_log_interval:
                self.get_logger().info(
                    f'[{object_name}] Located in known marked zone ({world_x:.2f},{world_y:.2f}) → Skipping duplicate marking'
                )
                self.last_known_log_time[color_type] = current_time
            return {'status': 'known'}
        
        # Multi-frame confirmation
        candidate = self.candidates[color_type]
        candidate['positions'].append((world_x, world_y))
        candidate['distances'].append(distance)
        candidate['last_seen'] = current_time
        candidate['count'] = len(candidate['positions'])
        
        frame_count = candidate['count']
        
        # Interval frame confirmation logic
        confirm_result = self.check_confirmation(candidate, object_name, color_type)
        
        # Display information
        status_text = f'{object_name} [{frame_count}/{self.min_frames}-{self.max_frames}]'
        if confirm_result == 'confirmed':
            status_text += ' LOCKED'
        elif frame_count >= self.min_frames:
            status_text += ' CHECKING'
        
        cv2.putText(frame, status_text, 
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'D:{distance:.2f}m', 
                   (x, y + bh + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        cv2.putText(frame, f'({world_x:.2f},{world_y:.2f})', 
                   (x, y + bh + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, box_color, 1)
        
        # Output log
        self.get_logger().info(
            f'[{object_name}] Centered [{frame_count}/{self.min_frames}-{self.max_frames}] '
            f'D:{distance:.2f}m Pos:({world_x:.2f},{world_y:.2f})'
        )
        
        return {'status': confirm_result or 'tracking', 'count': frame_count}
    
    def check_confirmation(self, candidate, object_name, color_type):
        """
        Interval frame confirmation logic
        Rules:
            ≥4 frames + very stable (std < 0.10) → confirm immediately
            ≥5 frames + stable (std < 0.12) → confirm immediately  
            =8 frames + basically stable (std < 0.15) → confirm
            =8 frames + unstable → abandon, reset
        Precise object coordinates reduce errors, and multiple status divisions avoid wasting time and improve efficiency.
        Position stability is evaluated by standard deviation, and objects are confirmed in stages.
        """
        frame_count = candidate['count']
        
        # Not reached minimum frames
        if frame_count < self.min_frames:
            return None
        
        # Calculate position stability
        positions = list(candidate['positions'])
        std_x = np.std([p[0] for p in positions])
        std_y = np.std([p[1] for p in positions])
        max_std = max(std_x, std_y)
        
        # Graded judgment
        should_confirm = False
        confirm_reason = ""
        
        if frame_count >= self.min_frames and max_std < self.strict_std:
            # Very stable, confirm immediately
            should_confirm = True
            confirm_reason = f"Very stable ({frame_count} frames, std={max_std:.3f})"
            
        elif frame_count >= 5 and max_std < self.normal_std:
            # Stable, confirm immediately
            should_confirm = True
            confirm_reason = f"Stable ({frame_count} frames, std={max_std:.3f})"
            
        elif frame_count >= self.max_frames:
            # Reached cap
            if max_std < self.loose_std:
                # Basically stable, confirm
                should_confirm = True
                confirm_reason = f"Capped confirmation ({frame_count} frames, std={max_std:.3f})"
            else:
                # Unstable, abandon
                self.get_logger().warn(
                    f'[{object_name}] Reached {self.max_frames} frames cap but unstable '
                    f'(std={max_std:.3f} > {self.loose_std}), abandoning'
                )
                self.clear_candidate(color_type)
                return 'abandoned'
        
        # Execute confirmation
        if should_confirm:
            final_x = np.mean([p[0] for p in positions])
            final_y = np.mean([p[1] for p in positions])
            
            if not self.is_in_marked_zone(final_x, final_y, color_type):
                self.get_logger().info(f'[{object_name}] {confirm_reason}')
                self.confirm_object(color_type, object_name, final_x, final_y)
                self.clear_candidate(color_type)
                return 'confirmed'
            else:
                self.clear_candidate(color_type)
                return 'duplicate'
        
        return None
    
    def get_centered_distance(self, cx, cy):
        # Get distance at the center
        depth_distance = self.get_depth_at_center(cx, cy)
        if depth_distance is not None:
            return depth_distance
        
        lidar_distance = self.get_distance_from_lidar()
        if lidar_distance is not None:
            return lidar_distance
        
        return None
    
    def clear_candidate(self, color_type):
        candidate = self.candidates[color_type]
        candidate['positions'].clear()
        candidate['distances'].clear()
        candidate['count'] = 0
    
    def calculate_world_position(self, angle_offset, distance):
        world_angle = self.robot_yaw + angle_offset
        world_x = self.robot_x + distance * math.cos(world_angle)
        world_y = self.robot_y + distance * math.sin(world_angle)
        return world_x, world_y
    
    def check_shape(self, contour, area):
        x, y, w, h = cv2.boundingRect(contour)
        
        aspect_ratio = float(w) / h if h > 0 else 0
        if aspect_ratio > self.max_aspect_ratio or aspect_ratio < self.min_aspect_ratio:
            return False
        
        rect_area = w * h
        extent = float(area) / rect_area if rect_area > 0 else 0
        if extent < self.min_extent:
            return False
        
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0
        if solidity < self.min_solidity:
            return False
        
        return True
    
    def confirm_object(self, obj_type, obj_name, world_x, world_y):
        # Confirm and register a new object
        if self.is_in_marked_zone(world_x, world_y, obj_type):
            return
        
        self.found_objects.append((obj_type, world_x, world_y, self.marker_id))
        
        if obj_type == 'green':
            self.marked_zones_green.append((world_x, world_y, self.duplicate_distance))
        else:
            self.marked_zones_red.append((world_x, world_y, self.duplicate_distance))
        
        self.publish_marker(obj_type, obj_name, world_x, world_y)
        self.publish_detection(world_x, world_y, obj_type)
        
        self.marker_id += 1
        
        green_count = sum(1 for obj in self.found_objects if obj[0] == 'green')
        red_count = sum(1 for obj in self.found_objects if obj[0] == 'red')
        
        self.get_logger().info('='*50)
        self.get_logger().info(f'Confirmed found {obj_name}!')
        self.get_logger().info(f'Position: ({world_x:.2f}, {world_y:.2f})')
        self.get_logger().info(f'Total: green {green_count} red {red_count}')
        self.get_logger().info('='*50)
    
    def is_in_marked_zone(self, world_x, world_y, obj_type):
        if obj_type == 'green':
            zones = self.marked_zones_green
        else:
            zones = self.marked_zones_red
        
        for zx, zy, radius in zones:
            dist = math.sqrt((world_x - zx) ** 2 + (world_y - zy) ** 2)
            if dist < radius:
                return True
        return False
    
    def cleanup_candidates(self, current_time):
        for color_type in self.candidates:
            candidate = self.candidates[color_type]
            if candidate['count'] > 0:
                if current_time - candidate['last_seen'] > self.confirm_timeout:
                    if candidate['count'] >= self.min_frames:
                        self.get_logger().info(f'[{color_type}] Candidate timeout ({candidate["count"]} frames)')
                    self.clear_candidate(color_type)

    # Display
    def show_debug_windows(self, frame, green_mask, red_mask):
        h, w = frame.shape[:2]
        
        sensors = []
        if self.depth_received:
            sensors.append("Depth")
        if self.scan_received:
            sensors.append("LiDAR")
        sensor_str = "+".join(sensors) if sensors else "NO SENSOR"
        
        green_count = sum(1 for obj in self.found_objects if obj[0] == 'green')
        red_count = sum(1 for obj in self.found_objects if obj[0] == 'red')
        
        info_lines = [
            f'Found: G={green_count} R={red_count} | {sensor_str}',
            f'Robot: ({self.robot_x:.2f}, {self.robot_y:.2f}) [{self.pose_source}]',
        ]
        
        for i, line in enumerate(info_lines):
            cv2.putText(frame, line, (10, h - 40 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # List of found objects
        y_offset = 50
        for obj_type, ox, oy, _ in self.found_objects:
            name = 'H' if obj_type == 'red' else 'B'
            color = (0, 0, 255) if obj_type == 'red' else (0, 255, 0)
            cv2.putText(frame, f'{name}:({ox:.2f},{oy:.2f})', 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18
        
        mask_colored = np.zeros_like(frame)
        mask_colored[green_mask > 0] = (0, 255, 0)
        mask_colored[red_mask > 0] = (0, 0, 255)
        
        cv2.imshow('Eye Node - Camera', frame)
        cv2.imshow('Eye Node - Detection', mask_colored)
        cv2.waitKey(1)
    
    def publish_detection(self, world_x, world_y, obj_type):
        world_point = PointStamped()
        world_point.header.frame_id = 'map'
        world_point.header.stamp = self.get_clock().now().to_msg()
        world_point.point.x = world_x
        world_point.point.y = world_y
        world_point.point.z = 0.0
        self.world_pos_pub.publish(world_point)
        
        point_msg = Point()
        point_msg.x = world_x
        point_msg.y = world_y
        point_msg.z = 1.0 if obj_type == 'red' else 0.0
        self.detection_pub.publish(point_msg)
    
    def publish_marker(self, obj_type, obj_name, world_x, world_y):
        marker_array = MarkerArray()
        
        cylinder = Marker()
        cylinder.header.frame_id = 'map'
        cylinder.header.stamp = self.get_clock().now().to_msg()
        cylinder.ns = 'found_objects'
        cylinder.id = self.marker_id
        cylinder.type = Marker.CYLINDER
        cylinder.action = Marker.ADD
        cylinder.pose.position.x = world_x
        cylinder.pose.position.y = world_y
        cylinder.pose.position.z = 0.3
        cylinder.pose.orientation.w = 1.0
        cylinder.scale.x = 0.4
        cylinder.scale.y = 0.4
        cylinder.scale.z = 0.6
        if obj_type == 'red':
            cylinder.color.r = 1.0
            cylinder.color.g = 0.2
            cylinder.color.b = 0.2
        else:
            cylinder.color.r = 0.2
            cylinder.color.g = 1.0
            cylinder.color.b = 0.2
        cylinder.color.a = 0.9
        cylinder.lifetime.sec = 0
        marker_array.markers.append(cylinder)
        
        text = Marker()
        text.header.frame_id = 'map'
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = 'object_labels'
        text.id = self.marker_id + 1000
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = world_x
        text.pose.position.y = world_y
        text.pose.position.z = 0.8
        text.pose.orientation.w = 1.0
        text.scale.z = 0.25
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        text.text = f'{obj_name}\n({world_x:.2f}, {world_y:.2f})'
        text.lifetime.sec = 0
        marker_array.markers.append(text)
        
        zone = Marker()
        zone.header.frame_id = 'map'
        zone.header.stamp = self.get_clock().now().to_msg()
        zone.ns = 'detection_zones'
        zone.id = self.marker_id + 2000
        zone.type = Marker.CYLINDER
        zone.action = Marker.ADD
        zone.pose.position.x = world_x
        zone.pose.position.y = world_y
        zone.pose.position.z = 0.01
        zone.pose.orientation.w = 1.0
        zone.scale.x = self.duplicate_distance * 2
        zone.scale.y = self.duplicate_distance * 2
        zone.scale.z = 0.02
        if obj_type == 'red':
            zone.color.r = 1.0
        else:
            zone.color.g = 1.0
        zone.color.a = 0.15
        zone.lifetime.sec = 0
        marker_array.markers.append(zone)
        
        self.marker_pub.publish(marker_array)

    def get_found_objects(self):
        return self.found_objects.copy()
    
    def get_object_count(self):
        return len(self.found_objects)


def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetector()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        green_count = sum(1 for obj in node.found_objects if obj[0] == 'green')
        red_count = sum(1 for obj in node.found_objects if obj[0] == 'red')
        
        node.get_logger().info('='*50)
        node.get_logger().info(f'Eye Node closed')
        node.get_logger().info(f'Total: green {green_count} , red {red_count} ')
        for t, x, y, _ in node.get_found_objects():
            name = 'Fire Hydrant' if t == 'red' else 'Trash Can'
            node.get_logger().info(f'  [{name}] ({x:.2f}, {y:.2f})')
        node.get_logger().info('='*50)
        
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
