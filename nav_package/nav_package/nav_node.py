import math
import time
import threading
from typing import Dict, List, Tuple, Optional

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import GoalStatus

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist, PointStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import String, Float32
from std_srvs.srv import Empty
from nav2_msgs.action import NavigateToPose

# =========================
# 0. Map -> Connected Graph
# Using discrete Math
# =========================

# Abandoned Features
SPEED = 0.2  # m/s

# Defined the position for Bridges and Checkpoints
NODES: Dict[str, Tuple[float, float]] = {
    "B1": (4.1, 0.9),
    "B2": (-0.25, 1.45),
    "B3": (0.4, 4.35),
    "B4": (-0.75, -1.0),

    "C11": (5.35, 0.1),
    "C12": (5.4, 2.8),
    "C13": (5.15, -3.0),

    "C21": (2.5, 0.0),
    "C22": (2.2, -2.9),
    "C23": (0.37, -3.1),
    "C24": (0.0, 0.0),

    "C31": (-0.25, 3.0),

    "C41": (1.4, 4.0),
    "C42": (3.75, 2.0),
    "C43": (1.4, 2.0),
    "C44": (3.4, 4.0),

    "C51": (-1.4, -3.0),
    "C52": (-1.5, 0.3),
    "C53": (-1.5, 3.2),
}

# Bind the relation between chunks and their checkpoints
ZONE_CHECKPOINTS = {
    "C1": ["C11", "C12", "C13"],
    "C2": ["C21", "C22", "C23", "C24"],
    "C3": ["C31"],
    "C4": ["C41", "C42", "C43", "C44"],
    "C5": ["C51", "C52", "C53"],
}

# Set of Checkpoints
CHECKPOINTS: Dict[str, Tuple[float, float]] = {
    k: v for k, v in NODES.items() if k.startswith("C")
}

# Bridge to Connect Different Chunks
BRIDGES = {"B1", "B2", "B3", "B4"}

# Whitelist for any Special Full Scan Checkpoints
SPECIAL_FULL_SCAN_CPS = {"C21", "C22", "C23", "C24", "C41", "C42", "C43", "C44"}

# Chunk Info (Polygon Points)
ZONES = {
    "C1": [
        (6.39, -4.57), (4.13, -4.55), (3.36, -2.24), (4.1, -0.236),
        (4.17, 1.37), (4.31, 3.74), (4.71, 3.74), (4.77, 4.614), (6.57, 4.59)
    ],
    "C2": [
        (4.17, 1.37), (4.1, -0.236), (3.36, -2.24), (4.13, -4.55),
        (-0.54, -4.4), (-0.56, -1.64), (-0.98, -0.228), (-1, 1.43), (0.38, 1.45)
    ],
    "C3": [
        (0.38, 4.83), (0.38, 1.45), (-1, 1.43), (-0.8, 4.8)
    ],
    "C4": [
        (0.38, 4.83), (0.38, 1.45), (4.17, 1.37),
        (4.31, 3.74), (4.71, 3.74), (4.77, 4.61)
    ],
    "C5": [
        (-0.91, 4.51), (-1, 1.43), (-0.98, -0.228), (-0.56, -1.64),
        (-0.54, -4.4), (-2.1, -4.32), (-1.86, 4.61)
    ]
}

# Base Adjacency List (Connected Graph)
def build_base_adj() -> Dict[str, List[str]]:
    adj = {nid: [] for nid in NODES}

    def connect(u: str, v: str):
        adj[u].append(v)
        adj[v].append(u)

    # Chunk C1
    connect("C11", "C12")
    connect("C11", "C13")
    connect("C11", "B1")
    connect("C12", "B1")

    # Chunk C2
    connect("C21", "C22")
    connect("C21", "C24")
    connect("C22", "C23")
    connect("C23", "C24")
    connect("C21", "B1")
    connect("C21", "B2")
    connect("C24", "B2")
    connect("C24", "B4")
    connect("C23", "B4")
    connect("B2", "B4")

    # Chunk C3
    connect("C31", "B2")
    connect("C31", "B3")

    # Chunk C4
    connect("C41", "C44")
    connect("C41", "C43")
    connect("C42", "C43")
    connect("C42", "C44")
    connect("C41", "B3")
    connect("C43", "B3")

    # Chunk C5
    connect("C51", "C52")
    connect("C52", "C53")
    connect("C51", "B4")
    connect("C52", "B4")

    return adj

ADJ = build_base_adj()

# =================
# 1. Tool Functions
# =================

# 1.1 Robot initial localization
# Calculate the straight-line (Euclidean) distance between two points.
def dist(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    return math.dist(a, b)

# Determine the robot's current zone.
def point_in_polygon(x: float, y: float, poly: List[Tuple[float, float]]) -> bool:
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        intersect = ((y1 > y) != (y2 > y)) and \
                    (x < (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1)
        if intersect:
            inside = not inside
    return inside

def find_zone(x: float, y: float) -> Optional[str]:
    for zname, poly in ZONES.items():
        if point_in_polygon(x, y, poly):
            return zname
    return None

# Find the nearest checkpoint
# !Abandoned (2D parameter will cause Trouble)!
# For example: Robot in C1 detect C42 which distance is right but route is wrong. 
def nearest_checkpoint(x: float, y: float) -> Tuple[str, float]:
    best_id = None
    best_d = float("inf")
    for cid, (cx, cy) in CHECKPOINTS.items():
        d = dist((x, y), (cx, cy))
        if d < best_d:
            best_id, best_d = cid, d
    return best_id, best_d

# Final Version: Robot will limited in the current chunk and find a nearest checkpoint.
def nearest_checkpoint_in_zone(x: float, y: float, zone: str) -> Tuple[Optional[str], float]:
    best_id = None
    best_d = float("inf")
    ids = ZONE_CHECKPOINTS.get(zone, [])
    for cid in ids:
        cx, cy = CHECKPOINTS[cid]
        d = dist((x, y), (cx, cy))
        if d < best_d:
            best_id, best_d = cid, d
    return best_id, best_d

# Sync with ROS2 and set angle ranges
def normalize_angle(a: float) -> float:
    while a > math.pi:
        a -= 2.0 * math.pi
    while a < -math.pi:
        a += 2.0 * math.pi
    return a


# 1.2 Shortest path on connectivity graph and TSP

# Use Dijkstra to compute the shortest path on a graph.
def shortest_path(start: str, goal: str) -> Tuple[Optional[List[str]], float]:

    import heapq

    pq = [(0.0, start, [start])]
    visited = set()

    while pq:
        cost, u, path = heapq.heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        if u == goal:
            return path, cost

        for v in ADJ[u]:
            if v not in visited:
                heapq.heappush(pq, (cost + dist(NODES[u], NODES[v]), v, path + [v]))

    return None, float("inf")


# 1.3 Build the shortest-path graph.

def build_checkpoint_graph() -> Tuple[List[str], Dict[str, int], List[List[float]], List[List[List[str]]]]:
    # Precompute the shortest distances and actual paths between checkpoints
    # Pass the results into shortest_route_from(start_cp).
    cp_list = sorted(CHECKPOINTS.keys())
    cp_index = {cp: i for i, cp in enumerate(cp_list)}
    n = len(cp_list)

    INF = float("inf")
    cp_dist = [[INF] * n for _ in range(n)]
    cp_path: List[List[List[str]]] = [[[] for _ in range(n)] for _ in range(n)]

    for i, cpi in enumerate(cp_list):
        for j, cpj in enumerate(cp_list):
            if i == j:
                cp_dist[i][j] = 0.0
                cp_path[i][j] = [cpi]
            else:
                path, d = shortest_path(cpi, cpj)
                if path is not None:
                    cp_dist[i][j] = d
                    cp_path[i][j] = path

    return cp_list, cp_index, cp_dist, cp_path

# Compute the shortest-path plan from a given start_point (start_cp).
def shortest_route_from(start_cp: str) -> Tuple[List[str], float, float]:

    import time as _time

    # (1) Data preprocessing and input validity checks.
    t0 = _time.perf_counter()

    cp_list, cp_index, cp_dist, cp_path = build_checkpoint_graph()
    n = len(cp_list)
    INF = float("inf")

    if start_cp not in cp_index:
        return [], 0.0, 0.0

    start_idx = cp_index[start_cp]

    # (2) Build dp[mask][i] and initialize the dynamic programming (DP) structure.
    dp = [[INF] * n for _ in range(1 << n)]
    parent: List[List[Optional[int]]] = [[None] * n for _ in range(1 << n)]

    dp[1 << start_idx][start_idx] = 0.0

    # (3) Iteration process.
    for mask in range(1 << n):
        for u in range(n):
            if not (mask & (1 << u)):
                continue
            if dp[mask][u] >= INF:
                continue
            for v in range(n):
                if mask & (1 << v):
                    continue
                if cp_dist[u][v] >= INF:
                    continue
                nmask = mask | (1 << v)
                nd = dp[mask][u] + cp_dist[u][v]
                if nd < dp[nmask][v]:
                    dp[nmask][v] = nd
                    parent[nmask][v] = u

    # (4) Find the best Destination
    full_mask = (1 << n) - 1
    best_dist = INF
    last = None
    for i in range(n):
        if dp[full_mask][i] < best_dist:
            best_dist = dp[full_mask][i]
            last = i

    if last is None or best_dist >= INF:
        return [], 0.0, 0.0

    # (5) Backtrack the visit order to obtain the route.
    seq_cp_idx = []
    mask = full_mask
    cur = last
    while cur is not None:
        seq_cp_idx.append(cur)
        p = parent[mask][cur]
        if p is None:
            break
        mask ^= (1 << cur)
        cur = p
    seq_cp_idx.reverse()

    cp_seq = [cp_list[i] for i in seq_cp_idx]

    route_nodes: List[str] = []
    for i in range(len(cp_seq) - 1):
        u = cp_seq[i]
        v = cp_seq[i + 1]
        u_idx = cp_index[u]
        v_idx = cp_index[v]
        segment = cp_path[u_idx][v_idx]
        if not segment:
            continue
        if route_nodes:
            route_nodes.extend(segment[1:])
        else:
            route_nodes.extend(segment)

    # (6) Calculate distance (Connectivity Graph) and time.
    total_dist = 0.0
    for i in range(len(route_nodes) - 1):
        a = route_nodes[i]
        b = route_nodes[i + 1]
        total_dist += dist(NODES[a], NODES[b])

    comp_t = _time.perf_counter() - t0
    total_time = total_dist / SPEED if SPEED > 1e-6 else 0.0

    return route_nodes, total_time, comp_t


# ==============================
# 2. NavNode Callbacks
# ==============================

class NavNode(Node):

    # 2.1 Initialization
    def __init__(self):
        super().__init__("zone_detector")

        # Robot’s initial position/zone/checkpoint
        self.robot_pose: Optional[Tuple[float, float]] = None
        self.robot_yaw: float = 0.0
        self.current_zone: Optional[str] = None
        self.current_nearest_cp: Optional[str] = None

        self.mode: Optional[str] = None  # "point" / "full" / None
        self.visited_checkpoints = set()
        self.first_checkpoint_done = False

        # Nav2 ActionClient status
        self.nav_client = ActionClient(self, NavigateToPose, "navigate_to_pose")
        self.current_target_id: Optional[str] = None
        self.current_goal_handle = None
        self.current_goal_active: bool = False
        self.current_goal_done: bool = False
        self.current_early_stop: bool = False
        self.current_threshold_remain: float = 0.0
        self.current_is_checkpoint: bool = False
        self.current_remaining: Optional[float] = None

        # Abandoned Features
        self.current_goal_start_time: Optional[float] = None

        # Variables for the 1-second protection window
        self.first_feedback_time: Optional[float] = None
        self.first_feedback_remaining: Optional[float] = None
        self.protection_active: bool = False
        self.protection_finished: bool = False

        # eye_node-related variables
        self.detected_objects: List[Tuple[float, float]] = []
        self.detected_min_dist = 0.2
        self.detected_lock = threading.Lock()
        self.is_scanning = False
        self.object_status = "none"
        self.object_offset = 0.0

        self.pending_detection: Optional[Tuple[float, float]] = None
        self.handling_detection: bool = False

        # Costmap-related
        # I HATE NAV2 PATHFIND LOGIC!!!
        self.clear_local_costmap_client = self.create_client(
            Empty, '/local_costmap/clear_entirely_local_costmap'
        )
        self.clear_global_costmap_client = self.create_client(
            Empty, '/global_costmap/clear_entirely_global_costmap'
        )

        # AMCL & ODOM Suscriptions
        # AMCL preferred, ODOM only as backup
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_callback,
            10
        )
        self.create_subscription(
            Odometry,
            "/odom",
            self.odom_callback,
            10
        )

        # Eye_Node Relate Subscriptions
        self.create_subscription(
            PointStamped,
            "/object_world_position",
            self.object_world_callback,
            10
        )

        self.create_subscription(
            String, "/object_status", self.object_status_callback, 10
        )
        self.create_subscription(
            Float32, "/object_offset", self.object_offset_callback, 10
        )

        # cmd_vel Publisher (for rotation scanning / centering)
        self.cmd_vel_pub = self.create_publisher(Twist, "/cmd_vel", 10)

        # Timer (currently no periodic logic)
        self.timer = self.create_timer(0.5, self.timer_callback)

        # Menu thread
        self.menu_thread = threading.Thread(target=self.menu_loop, daemon=True)
        self.menu_thread.start()

        self.zone_printed = False

        self.get_logger().info("nav_node launched, waiting for robot pose...")

    # 2.2 Costmap Clearing Functions
    """
    7/12/2025 Hope it would be useful.
    8/12/2025 But in fact it turned out to be completely pointless after testing.
    """
    def clear_costmaps(self):
        """
        Clear Nav2 local and global costmaps to prevent
        false obstacles caused during scanning from affecting navigation.
        Uses non-blocking calls to avoid blocking the main thread.
        """
        # Clear local costmap
        if self.clear_local_costmap_client.service_is_ready():
            self.clear_local_costmap_client.call_async(Empty.Request())
            self.get_logger().info("[Nav2] Requested clearing of local costmap")
        else:
            self.get_logger().info("[Nav2] local costmap clearing service not available, skipping")
        
        # Clear global costmap
        if self.clear_global_costmap_client.service_is_ready():
            self.clear_global_costmap_client.call_async(Empty.Request())
            self.get_logger().info("[Nav2] Requested clearing of global costmap")
        else:
            self.get_logger().info("[Nav2] global costmap clearing service not available, skipping")
    
    def wait_for_costmap_services(self, timeout_sec: float = 5.0) -> bool:
        """
        Wait for costmap clearing services to become available before navigation starts.
        Guess what? Never WORKS!!!
        """
        self.get_logger().info("[Nav2] Waiting for costmap clearing services...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_sec:

            rclpy.spin_once(self, timeout_sec=0.1)
            
            local_ready = self.clear_local_costmap_client.service_is_ready()
            global_ready = self.clear_global_costmap_client.service_is_ready()
            
            if local_ready and global_ready:
                self.get_logger().info("[Nav2] costmap clearing services are ready")
                return True
        
        self.get_logger().warn("[Nav2] costmap clearing services wait timed out, continuing anyway")
        return False

    # 2.3 Pose Callbacks
    # AMCL is preferred over ODOM

    def amcl_callback(self, msg: PoseWithCovarianceStamped):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        self.robot_pose = (x, y)

        # Calculate yaw
        q = msg.pose.pose.orientation
        self.robot_yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        )

        if self.current_zone is None:
            z = find_zone(x, y)
            if z is not None:
                self.current_zone = z
                cp, d0 = nearest_checkpoint_in_zone(x, y, z)
                self.current_nearest_cp = cp
                self.get_logger().info(
                    f"[Init] Robot is in zone {z}, nearest checkpoint {cp}, distance {d0:.2f}m"
                )

    def odom_callback(self, msg: Odometry):
        # Only set robot_pose if not already set by AMCL
        if self.robot_pose is None:
            x = msg.pose.pose.position.x
            y = msg.pose.pose.position.y
            self.robot_pose = (x, y)

    # 2.4 Object Detection Callback (Eye_Node)
    def object_world_callback(self, msg: PointStamped):
        x = msg.point.x
        y = msg.point.y

        # Determine whether the detected object is a duplicate.
        is_new = False
        with self.detected_lock:
            for ox, oy in self.detected_objects:
                if math.hypot(x - ox, y - oy) < self.detected_min_dist:
                    # Already recorded a nearby object, consider it a duplicate detection
                    break
            else:
                # New object
                self.detected_objects.append((x, y))
                is_new = True

        # If is new, log and report object
        if is_new:
            self.get_logger().info(
                f"[Vision] New detected object at world coordinates: ({x:.2f}, {y:.2f})"
            )
        else:
            # Duplicate detection of an old object, only debug output
            self.get_logger().debug(
                f"[Vision] Duplicate detected object near coordinates: ({x:.2f}, {y:.2f})"
            )

        # Slow down rotation if currently scanning (Capture Objects)
        if self.is_scanning:
            self.get_logger().info("[Vision-Assist] Object detected → Slow down")
            twist = Twist()
            twist.angular.z = 0.1
            for _ in range(10):
                self.cmd_vel_pub.publish(twist)
                time.sleep(0.1)
            return

        # Only trigger if it's a new object and in point/full mode and not already scanning
        if is_new and self.mode in ("point", "full") and not self.is_scanning:
            self.pending_detection = (x, y)
            self.get_logger().info(
                f"[Vision-Nav] New target detected during navigation ({x:.2f}, {y:.2f}), preparing to pause navigation and rotate in place"
            )
    # UPdate object status and offset callbacks
    def object_status_callback(self, msg):
        self.object_status = msg.data

    def object_offset_callback(self, msg):
        self.object_offset = msg.data


    # 2.5 Timer Callback
    # Reserved functionality (ultimately unused).
    def timer_callback(self):
        
        return

    # 2.6 Tools: Print Zone & Nearest Checkpoint
    def print_zone_and_nearest_cp(self):
        if self.robot_pose is None:
            return
        x, y = self.robot_pose
        zone = find_zone(x, y)
        if zone is not None:
            cp_id, d = nearest_checkpoint_in_zone(x, y, zone)
            self.get_logger().info(
                f"Current robot approximate coordinates: ({x:.2f}, {y:.2f}), located in zone {zone}"
            )
        else:
            cp_id, d = None, float("inf")
            self.get_logger().info(
                f"Current robot approximate coordinates: ({x:.2f}, {y:.2f}), not in any predefined zone"
            )

        self.get_logger().info(
            f"Nearest checkpoint: {cp_id} (Distance {d:.2f} m)"
        )
        self.zone_printed = True


    # 5. Menu Loop (Mode 1 for debug, Mode 2 for full scan)
    def menu_loop(self):
        # Wait for the first robot pose
        while rclpy.ok() and self.robot_pose is None:
            time.sleep(0.5)

        self.print_zone_and_nearest_cp()

        while rclpy.ok():
            print("\n===== Function Menu =====")
            print("1: P to P")
            print("2: Full Scan Mode")
            print("q: Quit Menu (Node continues running)")
            choice = input("Please enter the function number: ").strip().lower()
            if choice == "1":
                self.point_to_point_mode()
            elif choice == "2":
                self.full_scan_mode()
            elif choice == "q":
                print("Quit menu, nav_node continues running.")
                break
            else:
                print("Invalid input, please try again.")

    # (1) point to point mode

    def point_to_point_mode(self):
        print("Enter point-to-point mode (type exit to return to the previous menu)")
        self.mode = "point"

        while rclpy.ok():
            target = input("Please enter the target checkpoint (e.g., C42): ").strip().upper()
            if target in ("EXIT", "Q"):
                print("Exit point-to-point mode")
                self.mode = None
                return

            if target not in CHECKPOINTS:
                print(f"{target} is not a valid checkpoint, please try again.")
                continue

            if self.robot_pose is None:
                print("Robot pose not yet obtained, please try again later.")
                continue

            sx, sy = self.robot_pose
            zone = find_zone(sx, sy)

            if zone is None:
                print("Current robot is not in any zone, unable to select the nearest checkpoint within a zone.")
                return

            start_cp, d0 = nearest_checkpoint_in_zone(sx, sy, zone)
            print(f"Current zone: {zone}, nearest checkpoint: {start_cp} (Distance {d0:.2f}m) as the starting point")
            path_nodes, dist_sum = shortest_path(start_cp, target)
            if path_nodes is None:
                print(f"From {start_cp} to {target} is unreachable")
                continue

            print(f"Shortest path: {' -> '.join(path_nodes)}")
            print(f"Path distance on graph (sum of nodes): {dist_sum:.2f} m")
            self.execute_route(path_nodes, mode="point")


    # (2) full scan mode

    def full_scan_mode(self):
        print("Enter full scan mode")
        self.mode = "full"
        self.visited_checkpoints.clear()
        self.first_checkpoint_done = False

        if self.robot_pose is None:
            print("Robot pose not yet obtained, please try again later.")
            self.mode = None
            return

        sx, sy = self.robot_pose

        # Use zone determination + nearest checkpoint filtered by zone
        zone = find_zone(sx, sy)
        if zone is None:
            print(f"Current robot coordinates ({sx:.2f}, {sy:.2f}) are not within any zone, unable to initialize starting point.")
            self.mode = None
            return

        start_cp, d0 = nearest_checkpoint_in_zone(sx, sy, zone)

        if start_cp is None:
            print(f"Zone {zone} has no defined checkpoints, unable to perform full scan.")
            self.mode = None
            return

        print(f"[Full Scan Initialization] Current zone: {zone}, nearest checkpoint: {start_cp} (Distance {d0:.2f}m) as the starting point")
        
        # First go to start_cp and complete the initial 360° scan
        self.execute_route([start_cp], mode="full")

        # Then perform full map shortest path planning
        route_nodes, total_time, comp_t = shortest_route_from(start_cp)
        if not route_nodes:
            print("Full map shortest path planning failed")
            self.mode = None
            return

        print(
            "Full map optimal route: " + " -> ".join(route_nodes) +
            f"\nTotal travel time: {total_time:.2f} s, Computation time: {comp_t:.6f} s"
        )

        self.execute_route(route_nodes, mode="full")
        print("Full map scan completed")
        self.mode = None


    # 4. Execute Route & Single Point Navigation
    # 4.1 Execute Route
    def execute_route(self, route_nodes: List[str], mode: str):
        if not self.nav_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 Action Server not available (navigate_to_pose)")
            return

        self.mode = mode
        
        # Wait for costmap services to be ready before starting the route
        # HOWEVER, IT NEVER WORKS!!!
        self.wait_for_costmap_services(timeout_sec=3.0)

        for nid in route_nodes:
            ok = self.navigate_to_node(nid, mode) # Navigate to each node in the route
            if not ok:
                self.get_logger().warn(f"Failed to navigate to {nid}, aborting current route.")
                return False
        self.get_logger().info("Route execution completed")
        return True

    # 4.2 Nav2 feedback callback
    # Using feedback.distance_remaining for early stopping
    # Nav2 Stupid logic is unreliable and moving slowly like a TURTLE
    def nav_feedback_cb(self, feedback_msg):
        fb = feedback_msg.feedback
        remaining = fb.distance_remaining
        self.current_remaining = remaining

        if not self.current_goal_active or self.current_target_id is None:
            return

        if self.current_early_stop:
            return

        # 4.2.1 1 Second Protection Logic
        # Avoid ending early due to an incorrect remaining-distance value published at the start.
        if self.first_feedback_time is None:
            self.first_feedback_time = time.time()
            self.first_feedback_remaining = remaining
            self.protection_active = True
            self.protection_finished = False
            self.get_logger().info(
                f"[Protection] First feedback received → Starting 1-second protection timer | "
                f"Target={self.current_target_id}, remaining={remaining:.3f}m"
            )
            return

        # Protection timer still running: check if 1 second has passed
        if self.protection_active and not self.protection_finished:
            elapsed = time.time() - self.first_feedback_time

            if elapsed < 1.0:
                # Do not perform any arrival judgment during the protection period
                return

            # Protection timer ended
            self.protection_finished = True
            self.get_logger().info(
                f"[Protection] 1-second protection ended → Can start checking remaining distance | "
                f"Target={self.current_target_id}, elapsed={elapsed:.3f}s"
            )
            # Continue to check arrival conditions

        # If protection period is not finished, return directly (theoretically should not reach here)
        if not self.protection_finished:
            return

        # After protection period ends: officially perform arrival judgment
        thr = self.current_threshold_remain

        # 4.2.2 Arrival Judgment
        if remaining <= thr:
            self.get_logger().info(
                f"[Arrival Judgment] Early stop condition met → "
                f"Target={self.current_target_id}, remaining={remaining:.3f}m ≤ thr={thr:.3f}m"
            )
            self.current_early_stop = True
            if self.current_goal_handle is not None:
                self.current_goal_handle.cancel_goal_async()


    # 4.3 Very Important!!! Navigate to a single node
    def navigate_to_node(self, node_id: str, mode: str) -> bool:

        # (1) Early preparation (including target confirmation, etc.)
        # Set current target ID
        self.current_target_id = node_id

        # Read target coordinates
        target_x, target_y = NODES[node_id]

        # Restore old need_check logic
        need_check = (
            mode == "full"
            and node_id in CHECKPOINTS
            and (node_id not in self.visited_checkpoints or not self.first_checkpoint_done)
        )

        # If this is a check poitn, 0.1 m threshold; else (pass) 0.2 m
        threshold_remain = 0.1 if need_check else 0.2
        self.current_threshold_remain = threshold_remain

        # init status flags
        self.current_goal_active = False
        self.current_goal_done = False
        self.current_early_stop = False
        self.protection_active = False
        self.protection_finished = False
        self.first_feedback_time = None
        self.first_feedback_remaining = None
        self.pending_detection = None
        self.handling_detection = False
        self.current_goal_handle = None

        # Set max retries and initialize retry count
        max_retries = 3
        retry_count = 0

        # Outer loop: send the goal as long as error retries stay below the limit
        while retry_count < max_retries and rclpy.ok():

            aborted_this_attempt = False

            goal_msg = NavigateToPose.Goal()
            goal_msg.pose.header.frame_id = "map"
            goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
            goal_msg.pose.pose.position.x = float(target_x)
            goal_msg.pose.pose.position.y = float(target_y)
            goal_msg.pose.pose.orientation.w = 1.0

            self.get_logger().info(
                f"[Nav2] Sending goal to {node_id}, mode={mode}, "
                f"need_check={need_check}, thr={threshold_remain:.2f}"
            )

            send_goal_future = self.nav_client.send_goal_async(
                goal_msg, feedback_callback=self.nav_feedback_cb
            )

           # Wait for the goal to be accepted
            while rclpy.ok() and not send_goal_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)

            # On error: retry the error and increment the retry count
            goal_handle = send_goal_future.result()
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().warn(f"[Nav2] {node_id} goal not accepted → retrying")
                retry_count += 1
                continue

            # Save goal_handle for cancellation in feedback callback
            self.current_goal_handle = goal_handle
            self.current_goal_active = True
            
            # Reset protection status (each time a new goal is accepted)
            self.current_early_stop = False
            self.first_feedback_time = None
            self.protection_active = False
            self.protection_finished = False
            
            result_future = goal_handle.get_result_async()

            self.get_logger().info(
                f"[Nav2] Accepted goal to {node_id}, starting execution"
            )

            # Inner loop: keep monitoring continuously
            while rclpy.ok():
                rclpy.spin_once(self, timeout_sec=0.05)

                # （1） Vision Detection Interrupts Navigation
                if self.pending_detection and not self.handling_detection:
                    self.handling_detection = True
                    det_x, det_y = self.pending_detection

                    self.get_logger().info("[Vision] Detected object → Cancel navigation + center rotation")

                    goal_handle.cancel_goal_async()
                    self.center_on_detected_object(det_x, det_y)

                    # Reset protection status
                    self.current_goal_active = False
                    self.current_goal_done = False
                    self.current_early_stop = False
                    self.first_feedback_time = None
                    self.protection_active = False
                    self.protection_finished = False

                    self.pending_detection = None
                    self.handling_detection = False

                    retry_count += 1
                    aborted_this_attempt = True
                    break

                # (2) Early stop and try verify (remaining ≤ thr)
                if self.current_early_stop:
                    self.get_logger().info(
                        f"[Nav2] {node_id} Early stop reached threshold → Performing real distance verification"
                    )
                    self.current_goal_done = True
                    break

                # (3) Nav2 finished
                if result_future.done():
                    status = result_future.result().status

                    if status == GoalStatus.STATUS_SUCCEEDED:
                        self.get_logger().info(f"[Nav2] {node_id} SUCCEEDED")
                        self.current_goal_done = True
                        break

                    elif status == GoalStatus.STATUS_ABORTED:
                        self.get_logger().warn(f"[Nav2] {node_id} ABORTED, retrying")
                        # Clear costmaps and retry. STILL!!!
                        self.clear_costmaps()
                        time.sleep(0.3)
                        retry_count += 1
                        aborted_this_attempt = True
                        break

                    elif status == GoalStatus.STATUS_CANCELED:
                        # (4) Handle canceled status (triggered by early_stop)
                        if self.current_early_stop:
                            self.get_logger().info(f"[Nav2] {node_id} CANCELED (early stop)")
                            self.current_goal_done = True
                            break
                        else:
                            self.get_logger().warn(f"[Nav2] {node_id} CANCELED (unexpected) → Retrying")
                            retry_count += 1
                            aborted_this_attempt = True
                            break

                    else:
                        self.get_logger().warn(f"[Nav2] {node_id} Status={status} → Retrying")
                        retry_count += 1
                        aborted_this_attempt = True
                        break

            # Validate twice using relative-position checks. 2FA
            if self.current_goal_done and not aborted_this_attempt:

                if self.robot_pose is None:
                    self.get_logger().warn("[Verification] No pose → Retrying")
                    retry_count += 1
                    continue

                rx, ry = self.robot_pose
                real_dist = math.hypot(rx - target_x, ry - target_y)

                if real_dist > 0.5:
                    self.get_logger().warn(
                        f"[Verification Failed] Real distance {real_dist:.2f} > 0.5 → Continuing navigation"
                    )

                    # Reset early stop logic (old behavior)
                    self.current_goal_done = False
                    self.current_early_stop = False
                    self.protection_active = False
                    self.protection_finished = False
                    self.first_feedback_time = None

                    retry_count += 1
                    continue

                # Passed verification
                self.get_logger().info(
                    f"[Verification Passed] Real distance {real_dist:.2f} ≤ 0.5 → Reached {node_id}"
                )
                self.current_goal_handle = None
                self.on_reached_node(node_id, mode)
                return True

            # When aborted, retry directly
            if aborted_this_attempt:
                continue

            # Final check: not done yet
            if not self.current_goal_done:
                self.get_logger().warn(f"[Nav2] {node_id} Not done yet → Retrying")
                retry_count += 1
                continue

        # Exceeded max retries
        self.get_logger().error(f"[Nav2] {node_id} Multiple failures → Giving up")
        self.current_goal_handle = None
        return False
    
        # GUESS WHAT? THE WHOLE NAV2 LOGIC IS STILL UNRELIABLE AS HELL!!!
        # AND THAT'S WHY I ADD A LOT OF VERIFICATION TO AVOID ANY PROBLEM!!!
        # BUT IT STILL SOMETIMES FAILS... SIGH.

    # 6. Handling Arrival at a Node & Scanning
    def on_reached_node(self, nid: str, mode: str):
        self.get_logger().info(f"Reached {nid}")
        if self.robot_pose is None:
            self.get_logger().warn("Robot pose unknown, cannot perform distance check")
            return

        rx, ry = self.robot_pose
        nx, ny = NODES[nid]
        d = math.hypot(rx - nx, ry - ny)

        # Bridge logic: simply pass
        if nid in BRIDGES:
            self.get_logger().info(f"Passing bridge {nid} (Distance to target {d:.2f}m)")
            return

        # Checkpoint logic
        if nid in CHECKPOINTS:

            # In point-to-point mode: do not perform checks, just pass through
            if self.mode != "full":
                self.get_logger().info(f"[Checkpoint] (Point-to-point mode) Passing {nid}, no check performed (d={d:.2f}m)")
                return

            # In full map mode: first checkpoint reached → 360° scan
            if not self.first_checkpoint_done:
                self.first_checkpoint_done = True
                print(f"First checkpoint {nid} — performing 360° scan")
                self.full_scan_rotation()
                self.visited_checkpoints.add(nid)
                return

            # Already checked → do not repeat check
            if nid in self.visited_checkpoints:
                self.get_logger().info(f"[Checkpoint] Already checked {nid}, this time just passing through (d={d:.2f}m)")
                return

            # New checkpoint: decide 360 or light scan based on whether in SPECIAL_FULL_SCAN_CPS
            self.visited_checkpoints.add(nid)
            if nid in SPECIAL_FULL_SCAN_CPS:
                self.get_logger().info(f"Simulated check {nid} completed (special point, performing 360° scan)")
                self.full_scan_rotation()
            else:
                self.get_logger().info(f"Simulated check {nid} completed (new checkpoint, performing light left-right scan)")
                self.light_scan(nid)

    # Spawn Point Requires 360° Panorama Scan
    def full_scan_rotation(self):
        self.get_logger().info("[Scan] Starting 360° panorama scan")
        self.is_scanning = True

        twist = Twist()
        normal_speed = 0.6     # Normal rotation speed
        slow_speed = 0.1       # Slowdown rotation speed (Object detected)

        # Record starting yaw
        last_yaw = self.robot_yaw
        accumulated = 0.0       # Accumulated rotation angle (radians)
        self.get_logger().info("[DEBUG] 360° scan direction: left")

        # Target angle (2π)
        target_angle = 2 * math.pi

        while accumulated < target_angle and rclpy.ok():

            # External interrupt mechanism (avoid infinite rotation)
            if not self.is_scanning: # If not, it will rotate like CRAZY!!!
                break

            # slowdown dynamic speed adjustment
            if self.object_status == "slowdown":
                twist.angular.z = slow_speed
            else:
                twist.angular.z = normal_speed

            self.cmd_vel_pub.publish(twist)

            # Accumulate rotation angle for each frame
            current_yaw = self.robot_yaw
            delta = normalize_angle(current_yaw - last_yaw)
            accumulated += abs(delta)
            last_yaw = current_yaw

            time.sleep(0.02)

        # SOMETIMES ROBOT IS SLEEPY AFTER ROTATION
        # SO I NEED TO NOCK IT TO WAKE UP
        self.is_scanning = False
        stop_twist = Twist()
        for _ in range(10):  # That's not spam, it's just waking up!!!
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.05)
        
        # Clean up pending detection set during scan
        self.pending_detection = None
        
        # -> :483
        self.clear_costmaps()
        
        time.sleep(0.8)

        self.get_logger().info("[Scan] 360° panorama scan completed (accumulated angle: %.2f degrees)"
                            % math.degrees(accumulated))


    # Light scan for regular checkpoints
    def light_scan(self, nid: str):
        self.get_logger().info(f"[Scan] Performing light left-right scan at {nid}")
        self.is_scanning = True

        twist = Twist()
        normal_speed = 0.5
        slow_speed = 0.2

        # Set target angles for each segment (NOT TOO MUCH, JUST SO SO)
        left_angle = math.radians(30)
        right_angle = math.radians(60)
        back_left_angle = math.radians(30)

        # Step1: Turn left
        self.get_logger().info("[DEBUG] Light scan step 1: turn left")

        last_yaw = self.robot_yaw
        accumulated = 0.0

        while accumulated < left_angle and rclpy.ok():

            # Interrupt condition
            if not self.is_scanning:
                break

            if self.object_status == "slowdown":
                twist.angular.z = slow_speed
            else:
                twist.angular.z = normal_speed

            self.cmd_vel_pub.publish(twist)

            cur = self.robot_yaw
            delta = normalize_angle(cur - last_yaw)
            accumulated += abs(delta)
            last_yaw = cur

            time.sleep(0.02)

        # Step2: Turn right
        self.get_logger().info("[DEBUG] Light scan step 2: turn right")

        last_yaw = self.robot_yaw
        accumulated = 0.0

        while accumulated < right_angle and rclpy.ok():

            if not self.is_scanning:
                break

            if self.object_status == "slowdown":
                twist.angular.z = -slow_speed
            else:
                twist.angular.z = -normal_speed

            self.cmd_vel_pub.publish(twist)

            cur = self.robot_yaw
            delta = normalize_angle(cur - last_yaw)
            accumulated += abs(delta)
            last_yaw = cur

            time.sleep(0.02)

        # Step3: Turn back left
        self.get_logger().info("[DEBUG] Light scan step 3: turn back left")

        last_yaw = self.robot_yaw
        accumulated = 0.0

        while accumulated < back_left_angle and rclpy.ok():

            if not self.is_scanning:
                break

            if self.object_status == "slowdown":
                twist.angular.z = slow_speed
            else:
                twist.angular.z = normal_speed

            self.cmd_vel_pub.publish(twist)

            cur = self.robot_yaw
            delta = normalize_angle(cur - last_yaw)
            accumulated += abs(delta)
            last_yaw = cur

            time.sleep(0.02)

        # Still need to WAKE UP the robot if it's "STILL SLEEPING Zzz"
        self.is_scanning = False
        stop_twist = Twist()
        for _ in range(10):
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.05)
        
        # Clean up pending detection set during scan
        self.pending_detection = None
        
        self.clear_costmaps()
        
        time.sleep(0.8)
        
        self.get_logger().info(f"[Scan] Light scan at {nid} completed")


    # Center rotation based on detected object
    def center_on_detected_object(self, wx: float, wy: float):
        """
        Based on the current robot pose + yaw + object world coordinates,
        estimate the relative angle and rotate to roughly align the object in front of the robot.
        """
        if self.robot_pose is None:
            self.get_logger().warn("[Vision-Nav] Unable to center: robot pose unknown")
            return

        rx, ry = self.robot_pose
        angle_to_obj = math.atan2(wy - ry, wx - rx)
        angle_error = normalize_angle(angle_to_obj - self.robot_yaw)

        self.get_logger().info(
            f"[Vision-Nav] Centering rotation: "
            f"Target azimuth={math.degrees(angle_to_obj):.1f}°, "
            f"Current yaw={math.degrees(self.robot_yaw):.1f}°, "
            f"Error={math.degrees(angle_error):.1f}°"
        )

        # If the error is very small, no need to rotate
        if abs(angle_error) < math.radians(5.0):
            self.get_logger().info("[Vision-Nav] Object is already roughly in front, no additional rotation needed")
            return

        rot_speed = 0.4  # rad/s
        direction = 1.0 if angle_error > 0.0 else -1.0
        duration = abs(angle_error) / rot_speed

        twist = Twist()
        twist.angular.z = direction * rot_speed
        if direction > 0:
            self.get_logger().info("[DEBUG] Vision-Nav Centering: turning left")
        else:
            self.get_logger().info("[DEBUG] Vision-Nav Centering: turning right")

        t0 = time.time()
        while time.time() - t0 < duration and rclpy.ok():
            self.cmd_vel_pub.publish(twist)
            time.sleep(0.05)

        # Still WAKE UP
        stop_twist = Twist()
        for _ in range(5):
            self.cmd_vel_pub.publish(stop_twist)
            time.sleep(0.05)
            
        self.get_logger().info("[Vision-Nav] Centering rotation completed")



# main
def main(args=None):
    rclpy.init(args=args)
    node = NavNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()