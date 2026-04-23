#!/usr/bin/env python3
"""
ROS2 service node: MoveObjectsService
--------------------------------------
Service type : MoveObjects.srv  (object_mover_interface package)
  Request  : string[] object_names
             string[] goal_names
  Response : bool    success
             string  message

Goal resolution — each goal_names[i] is checked in this order:
  1. Known robot name  → use its latest /odom pose
  2. Known table name  → use per-object slot (or centre for single object)
  3. Known preset location → use preset pose
  4. Neither → skip with a warning
"""

import subprocess
import threading
from collections import defaultdict

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

from object_mover_interface.srv import MoveObjects


# ═══════════════════════════════════════════════════════════════════════════
#  WORLD
# ═══════════════════════════════════════════════════════════════════════════
WORLD = "food_court"

# ═══════════════════════════════════════════════════════════════════════════
#  ROBOTS
# ═══════════════════════════════════════════════════════════════════════════
TRACKED_ROBOTS: list[str] = [
    "delivery_bot1",
    "delivery_bot2",
    "delivery_bot3",
    "cleaning_bot",
]
_ROBOT_SET = set(TRACKED_ROBOTS)

# ═══════════════════════════════════════════════════════════════════════════
#  TABLE CENTRES  (dummy poses — replace with real coords from your world)
# ═══════════════════════════════════════════════════════════════════════════
TABLE_CENTRES: dict[str, list[float]] = {
    "table1": [ 6.0,  2.0, 0.5],
    "table2": [12.0,  2.0, 0.5],
    "table3": [18.0,  2.0, 0.5],
    "table4": [ 6.0, 18.0, 0.5],
    "table5": [12.0, 18.0, 0.5],
    "table6": [18.0, 18.0, 0.5],
}

# ═══════════════════════════════════════════════════════════════════════════
#  OBJECT ALIAS MAP  —  friendly name → possible Gazebo model names
# ═══════════════════════════════════════════════════════════════════════════
OBJECT_ALIASES: dict[str, list[str]] = {
    "cake":       ["cake", "Birthday_cake", "CAKE", "Cake"],
    "speaker":    ["speaker", "JBL_Speaker", "Speaker", "SPEAKER"],
    "gift":       ["gift", "gift_box", "GIFT", "Gift"],
    "cold_drink": ["food3", "pepsi2", "pepsi1", "pepsi", "juice", "juices", "drinks", "cold_drink", "cold_drinks", "cold drink", "drink", "cold drinks"],
}

# Per-object slots: centre ±0.3 on X axis (same pattern as original table2)
# Keywords are matched against the object name (lower-case substring match)
def _make_slots(cx: float, cy: float, z: float) -> dict[str, list[float]]:
    return {
        "gift":    [cx - 0.3, cy, z],
        "cake":    [cx,       cy, z],
        "speaker": [cx + 0.3, cy, z],
    }

TABLE_SLOTS: dict[str, dict[str, list[float]]] = {
    name: _make_slots(*centre)
    for name, centre in TABLE_CENTRES.items()
}

TABLE_NAMES = set(TABLE_CENTRES.keys())   # {"table1" … "table6"}

# Normalised lookup: "table 1" / "Table_1" / "TABLE1" → "table1"
_TABLE_NORM: dict[str, str] = {
    name.lower().replace(" ", "").replace("_", ""): name
    for name in TABLE_CENTRES
}

# ═══════════════════════════════════════════════════════════════════════════
#  OTHER PRESET LOCATIONS
# ═══════════════════════════════════════════════════════════════════════════
PRESET_LOCATIONS: dict[str, list[float]] = {
    "sink1":       [22.0,  6.0, 0.65],
    "sink2":       [ 2.0, 14.0, 0.65],
    "salad_stall": [ 7.0, 10.0, 0.5],
    "juice_stall": [17.0, 10.0, 0.5],
}
_LOCATION_SET = set(PRESET_LOCATIONS.keys())

# Normalised lookup: "Salad Stall" / "salad stall" / "SALAD_STALL" → "salad_stall"
_LOCATION_NORM: dict[str, str] = {
    name.lower().replace(" ", "").replace("_", ""): name
    for name in PRESET_LOCATIONS
}


# ═══════════════════════════════════════════════════════════════════════════
#  IGNITION GAZEBO HELPER
# ═══════════════════════════════════════════════════════════════════════════
def _ign_move(name: str, pose: list[float]) -> bool:
    x, y, z = pose
    req = (
        f'name: "{name}" '
        f'position: {{ x: {x}, y: {y}, z: {z} }} '
        f'orientation: {{ x: 0, y: 0, z: 0, w: 1 }}'
    )
    cmd = [
        "ign", "service",
        "-s", f"/world/{WORLD}/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000",
        "--req", req,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


# ═══════════════════════════════════════════════════════════════════════════
#  SERVICE NODE
# ═══════════════════════════════════════════════════════════════════════════
class MoveObjectsService(Node):

    def __init__(self):
        super().__init__("move_objects_service")

        self._robot_poses: dict[str, list[float]] = {}
        self._pose_lock = threading.Lock()

        for robot in TRACKED_ROBOTS:
            topic = f"/{robot}/odom"
            self.create_subscription(Odometry, topic, self._make_odom_cb(robot), 10)

        self.create_service(MoveObjects, "move_objects", self._handle_request)
        self.get_logger().info("MoveObjectsService ready  →  /move_objects")

    # ── name normaliser ──────────────────────────────────────────────────
    @staticmethod
    def _normalize_name(s: str) -> str:
        """'Table 1' → 'table1',  'Table_1' → 'table1',  'SALAD STALL' → 'saladstall'"""
        return s.lower().replace(" ", "").replace("_", "")

    # ── odom callback factory ────────────────────────────────────────────
    def _make_odom_cb(self, robot_name: str):
        def _cb(msg: Odometry):
            p = msg.pose.pose.position
            with self._pose_lock:
                self._robot_poses[robot_name] = [p.x, p.y, p.z]
        return _cb

    # ── table slot resolver ───────────────────────────────────────────────
    def _table_pose_map(self, table: str, objects: list[str]) -> dict[str, list[float]]:
        """
        1 object  → table centre.
        2+ objects → match keyword (gift_box / cake / speaker) in object name,
                     fallback to centre if no keyword matches.
        """
        centre = TABLE_CENTRES[table]
        if len(objects) == 1:
            return {objects[0]: centre}

        slots = TABLE_SLOTS[table]
        mapping: dict[str, list[float]] = {}
        used: set[str] = set()
        for obj in objects:
            assigned = False
            for keyword, slot_pose in slots.items():
                if keyword in obj.lower() and keyword not in used:
                    mapping[obj] = slot_pose
                    used.add(keyword)
                    assigned = True
                    break
            if not assigned:
                mapping[obj] = centre
        return mapping

    # ── goal → pose resolver ──────────────────────────────────────────────
    def _resolve_goal(self, goal: str) -> tuple[list[float] | None, str]:
        # 1. Robot name (exact match)
        if goal in _ROBOT_SET:
            with self._pose_lock:
                pose = self._robot_poses.get(goal)
            if pose is None:
                return None, f"robot '{goal}' has no odom data yet"
            return [pose[0], pose[1], 0.5], f"robot '{goal}' odom"

        # 2. Preset location — normalised match
        norm = self._normalize_name(goal)
        canonical_loc = _LOCATION_NORM.get(norm)
        if canonical_loc:
            return list(PRESET_LOCATIONS[canonical_loc]), f"location '{canonical_loc}'"

        return None, f"'{goal}' is not a robot, table, or known location"

    # ── service handler ──────────────────────────────────────────────────
    def _handle_request(
        self,
        request: MoveObjects.Request,
        response: MoveObjects.Response,
    ) -> MoveObjects.Response:

        # ── 1. Log incoming request ───────────────────────────────────────
        self.get_logger().info(
            f"Request received  →  objects: {list(request.object_names)}  |  goals: {list(request.goal_names)}"
        )

        obj_names  = [self._resolve_object_name(s.strip()) for s in request.object_names]
        goal_names = [s.strip() for s in request.goal_names]
        n = len(obj_names)

        if n == 0:
            response.success = False
            response.message = "object_names is empty"
            return response

        # pad goal_names if caller sent fewer entries
        goal_names += [""] * (n - len(goal_names))

        # ── pre-compute slot maps for every table that has objects going there
        table_bound: dict[str, list[str]] = defaultdict(list)
        for i in range(n):
            norm = self._normalize_name(goal_names[i])
            canonical = _TABLE_NORM.get(norm)
            if canonical:
                table_bound[canonical].append(obj_names[i])

        table_pose_maps: dict[str, dict[str, list[float]]] = {
            tbl: self._table_pose_map(tbl, objs)
            for tbl, objs in table_bound.items()
        }

        # ── process each (object, goal) pair ─────────────────────────────
        results: list[str] = []
        all_ok = True

        for i in range(n):
            obj  = obj_names[i]
            goal = goal_names[i]

            if not obj:
                results.append(f"[{i}] skipped — empty object name")
                continue
            if not goal:
                results.append(f"[{i}] '{obj}' skipped — empty goal")
                continue

            norm            = self._normalize_name(goal)
            canonical_table = _TABLE_NORM.get(norm)

            if canonical_table:
                pose   = table_pose_maps[canonical_table].get(obj, TABLE_CENTRES[canonical_table])
                source = canonical_table
            else:
                pose, source = self._resolve_goal(goal)

            # ── 2. Log resolved goal pose ─────────────────────────────────
            if pose is None:
                self.get_logger().warn(
                    f"[{i}] '{obj}'  →  goal '{goal}' could not be resolved — SKIPPED"
                )
                results.append(f"[{i}] '{obj}' SKIPPED — {source}")
                all_ok = False
                continue

            self.get_logger().info(
                f"[{i}] '{obj}'  →  goal pose {[round(v, 3) for v in pose]}  (via {source})"
            )

            # ── 3. Execute & log result ───────────────────────────────────
            ok = _ign_move(obj, pose)
            if ok:
                self.get_logger().info(f"[{i}] '{obj}'  →  ✓ moved successfully")
            else:
                self.get_logger().error(f"[{i}] '{obj}'  →  ✗ move FAILED")
                all_ok = False

            results.append(f"[{i}] '{obj}' → {pose}  via {source}  [{'OK' if ok else 'FAILED'}]")

        response.success = all_ok
        response.message = "\n".join(results)
        return response

    def _resolve_object_name(self, obj: str) -> str:
        ol = obj.lower().strip()

        matched_key = None
        candidates  = None

        # 1. exact key match
        if ol in OBJECT_ALIASES:
            matched_key = ol
            candidates  = OBJECT_ALIASES[ol]

        # 2. check if ol matches any VALUE in any alias list
        if matched_key is None:
            for key, vals in OBJECT_ALIASES.items():
                if ol in [v.lower() for v in vals]:
                    matched_key = key
                    candidates  = vals
                    break

        # 3. substring match against keys
        if matched_key is None:
            for key, vals in OBJECT_ALIASES.items():
                if key in ol or ol in key:
                    matched_key = key
                    candidates  = vals
                    break

        if matched_key:
            self.get_logger().info(f"Name matched  →  '{obj}'  →  '{matched_key}'")
            return matched_key

        self.get_logger().warn(f"No match for '{obj}' — using as-is")
        return obj


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
def main(args=None):
    rclpy.init(args=args)
    node = MoveObjectsService()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()