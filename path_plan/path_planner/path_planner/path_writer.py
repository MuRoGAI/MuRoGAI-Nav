#!/usr/bin/env python3
import os
import csv
import shutil
import rclpy
from rclpy.node import Node
from path_planner_interface.msg import (
    RobotTrajectoryArray,
    RobotTrajectory,
    DiffDriveTrajectory,
    HoloTrajectory,
)

USER  = os.environ.get("USER")

DIR = os.environ.get(
    "PLANNER_OUTPUT",
    f"/home/{USER}/murogai_nav/src/MuRoGAI-Nav/"
    "path_plan/path_planner/trajectory_logs"
)

PUBLISHED_ROOT = os.path.join(DIR, "published_paths")

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _is_run_dir(name: str) -> bool:
    """Matches run_NNN (3+ digits)."""
    if not name.startswith("run_"):
        return False
    return name[4:].isdigit() and len(name[4:]) >= 3


def _is_req_dir(name: str) -> bool:
    """Matches req_NNN (3+ digits)."""
    if not name.startswith("req_"):
        return False
    return name[4:].isdigit() and len(name[4:]) >= 3


def _run_number(name: str) -> int:
    return int(name[4:])


def _req_number(name: str) -> int:
    return int(name[4:])


def _read_csv(filepath):
    """Return list of row dicts."""
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def _infer_robot_type_and_rows(filepath):
    """
    Infer robot type from CSV columns:
      diff-drive  -> columns include: theta or omega
      holonomic   -> columns include: vx or vy
    """
    rows = _read_csv(filepath)
    if not rows:
        return "holonomic", rows
    cols = set(rows[0].keys())
    if "omega" in cols or "theta" in cols:
        return "diff-drive", rows
    return "holonomic", rows


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _extract_robot_name(stem: str) -> str:
    """
    Strip the leading agent key (F1, R1, R2, etc.) from a CSV stem.

    e.g.  "F1_delivery_bot1"  ->  "delivery_bot1"
          "R2_cleaning_robot" ->  "cleaning_robot"
    """
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def _is_trajectory_csv(filename: str) -> bool:
    """Accept only robot-trajectory CSVs -- skip centroid files."""
    if not filename.endswith(".csv"):
        return False
    if filename.endswith("_centroid.csv"):
        return False
    if filename.endswith("_archived.csv"):
        return False
    return True


def _build_and_publish_single_csv(fpath, logger, label, publisher):
    """
    Read one CSV file, build a RobotTrajectoryArray with a single entry,
    publish it, then rename the file to *_archived.csv.

    Returns True if published successfully, False otherwise.
    """
    fname = os.path.basename(fpath)
    robot_type, rows = _infer_robot_type_and_rows(fpath)

    if not rows:
        logger.warn(f"[{label}] Empty CSV, skipping: {fname}")
        return False

    stem       = fname[:-4]                   # strip ".csv"
    robot_name = _extract_robot_name(stem)

    robot_msg            = RobotTrajectory()
    robot_msg.robot_name = robot_name
    robot_msg.robot_type = robot_type

    times = [_safe_float(r["time"]) for r in rows]
    xs    = [_safe_float(r["x"])    for r in rows]
    ys    = [_safe_float(r["y"])    for r in rows]

    if robot_type == "diff-drive":
        dd       = DiffDriveTrajectory()
        dd.time  = times
        dd.x     = xs
        dd.y     = ys
        dd.theta = [_safe_float(r.get("theta")) for r in rows]
        dd.v     = [_safe_float(r.get("v"))     for r in rows]
        dd.omega = [_safe_float(r.get("omega")) for r in rows]
        robot_msg.diff_drive_trajectories = [dd]
        robot_msg.holo_trajectories       = []
    else:
        hl      = HoloTrajectory()
        hl.time = times
        hl.x    = xs
        hl.y    = ys
        hl.vx   = [_safe_float(r.get("vx")) for r in rows]
        hl.vy   = [_safe_float(r.get("vy")) for r in rows]
        robot_msg.diff_drive_trajectories = []
        robot_msg.holo_trajectories       = [hl]

    msg = RobotTrajectoryArray()
    msg.robot_trajectories.append(robot_msg)
    publisher.publish(msg)

    logger.info(
        f"[{label}] Published {robot_name} ({robot_type}): "
        f"{len(times)} pts, duration={times[-1]:.2f}s  |  file: {fname}"
    )

    # Rename to archived
    dir_      = os.path.dirname(fpath)
    archived  = os.path.join(dir_, f"{stem}_archived.csv")
    os.rename(fpath, archived)
    logger.info(f"[{label}] Archived: {fname} -> {stem}_archived.csv")

    return True


# ─────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────

class PathPublisher(Node):
    def __init__(self, node_name):
        super().__init__(node_name)

        self.path_pub = self.create_publisher(
            RobotTrajectoryArray, '/path_planner/paths', 10
        )

        # The single run dir we are watching (highest run_NNN on startup)
        self._active_run: str | None = None   # e.g. "run_012"

        # The current req dir we are watching inside the active run
        self._active_req: str | None = None   # e.g. "req_002"

        # Select the active run once at startup
        self._select_active_run()

        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f"PathPublisher started. Watching: {DIR} | "
            f"active run: {self._active_run}"
        )

    # ------------------------------------------------------------------
    # Startup: lock onto the highest run_NNN that already exists
    # ------------------------------------------------------------------

    def _select_active_run(self):
        if not os.path.isdir(DIR):
            os.makedirs(DIR, exist_ok=True)
            self.get_logger().info(f"Trajectory logs dir not found -- created: {DIR}")

        try:
            entries = os.listdir(DIR)
        except OSError as e:
            self.get_logger().warn(f"Cannot read trajectory logs dir: {e}")
            return

        run_dirs = [
            e for e in entries
            if os.path.isdir(os.path.join(DIR, e)) and _is_run_dir(e)
        ]
        if not run_dirs:
            self.get_logger().warn("No run_NNN directories found yet.")
            return

        self._active_run = max(run_dirs, key=_run_number)
        self.get_logger().info(f"Active run locked to: {self._active_run}")

    # ------------------------------------------------------------------
    # Timer callback
    # ------------------------------------------------------------------

    def _tick(self):
        # If we never found a run dir, keep retrying
        if self._active_run is None:
            self._select_active_run()
            return

        run_path = os.path.join(DIR, self._active_run)
        if not os.path.isdir(run_path):
            self.get_logger().warn(f"Active run dir disappeared: {run_path}")
            return

        # ── Find the highest req_NNN inside the active run ──────────────
        try:
            req_entries = os.listdir(run_path)
        except FileNotFoundError:
            return

        req_dirs = [
            r for r in req_entries
            if os.path.isdir(os.path.join(run_path, r)) and _is_req_dir(r)
        ]
        if not req_dirs:
            return  # nothing to do yet

        highest_req = max(req_dirs, key=_req_number)

        # If a newer req has appeared, switch to it
        if self._active_req != highest_req:
            self.get_logger().info(
                f"Switching active req: {self._active_req} -> {highest_req}"
            )
            self._active_req = highest_req

        req_path = os.path.join(run_path, self._active_req)
        label    = f"{self._active_run}/{self._active_req}"

        # ── Publish every unarchived trajectory CSV in the active req ───
        try:
            files = sorted(os.listdir(req_path))
        except FileNotFoundError:
            return

        for fname in files:
            if not _is_trajectory_csv(fname):
                continue
            fpath = os.path.join(req_path, fname)
            _build_and_publish_single_csv(
                fpath, self.get_logger(), label, self.path_pub
            )

    # ------------------------------------------------------------------
    # Shutdown: move the active run_NNN dir into published_paths
    # ------------------------------------------------------------------

    def destroy_node(self):
        self.get_logger().info(
            "Shutting down -- moving all run_NNN dirs to published_paths ..."
        )

        os.makedirs(PUBLISHED_ROOT, exist_ok=True)

        try:
            entries = os.listdir(DIR)
        except FileNotFoundError:
            entries = []

        run_dirs = sorted(
            e for e in entries
            if os.path.isdir(os.path.join(DIR, e)) and _is_run_dir(e)
        )

        if not run_dirs:
            self.get_logger().info("No run_NNN dirs found to move.")
        else:
            for run_name in run_dirs:
                src = os.path.join(DIR, run_name)
                dst = os.path.join(PUBLISHED_ROOT, run_name)

                if os.path.exists(dst):
                    self.get_logger().warn(
                        f"Destination already exists, skipping: {dst}"
                    )
                    continue

                shutil.move(src, dst)
                self.get_logger().info(f"Moved: {src}  ->  {dst}")

        self.get_logger().info("Shutdown complete.")
        super().destroy_node()


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

def main(args=None):
    rclpy.init(args=args)
    node = PathPublisher('path_publisher')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()