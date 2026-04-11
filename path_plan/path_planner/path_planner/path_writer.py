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

DIR = os.environ.get(
    "PLANNER_OUTPUT",
    "/home/multi-robot/murogai_nav/src/MuRoGAI-Nav/"
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


def _read_csv(filepath):
    """Return list of row dicts."""
    with open(filepath, newline="") as f:
        return list(csv.DictReader(f))


def _infer_robot_type_and_rows(filepath):
    """
    Infer robot type from CSV columns (matches _save_state_csvs output):
      diff-drive  -> columns: time, x, y, theta, v, omega
      holonomic   -> columns: time, x, y, vx, vy
    """
    rows = _read_csv(filepath)
    if not rows:
        return "holonomic", rows
    cols = set(rows[0].keys())
    if "omega" in cols:
        return "diff-drive", rows
    if "vx" in cols or "vy" in cols:
        return "holonomic", rows
    if "theta" in cols:
        return "diff-drive", rows
    return "holonomic", rows


def _safe_float(val):
    try:
        return float(val)
    except (TypeError, ValueError):
        return float("nan")


def _extract_robot_name(stem: str) -> str:
    """
    Extract the robot name from a CSV filename stem.

    Filenames produced by _save_state_csvs follow the pattern:
        {agent_key}_{robot_name}

    where agent_key is like  F1, R1, R2  (no underscores),
    and robot_name can itself contain underscores, e.g. delivery_bot1.

    We split on '_' exactly once and return everything after the first segment.

    Examples:
        "F1_delivery_bot1"   ->  "delivery_bot1"
        "F1_delivery_bot2"   ->  "delivery_bot2"
        "R1_cleaning_robot"  ->  "cleaning_robot"
        "R2_some_bot"        ->  "some_bot"
    """
    parts = stem.split("_", 1)
    return parts[1] if len(parts) > 1 else stem


def _is_trajectory_csv(filename: str) -> bool:
    """
    Accept only robot-trajectory CSVs -- skip centroid files.

    Centroid files end with '_centroid.csv' (e.g. F1_centroid.csv).
    All other .csv files are robot trajectory files.
    """
    if not filename.endswith(".csv"):
        return False
    if filename.endswith("_centroid.csv"):
        return False
    return True


def _build_msg_from_req_dir(req_path, logger, req_label):
    """
    Read every robot-trajectory CSV in req_path and build one
    RobotTrajectoryArray message.

    One RobotTrajectory entry is created per CSV file, so a formation
    with N robots produces N entries in the same message -- all published
    together in a single call.

    Skipped files:
        *_centroid.csv  -- formation centroid summary, not a robot path.
    """
    msg = RobotTrajectoryArray()

    try:
        csv_files = sorted(
            f for f in os.listdir(req_path)
            if _is_trajectory_csv(f)
        )
    except FileNotFoundError:
        logger.warn(f"req dir disappeared: {req_path}")
        return None

    if not csv_files:
        logger.warn(f"No trajectory CSV files found in {req_path}")
        return None

    logger.info(
        f"[{req_label}] Found {len(csv_files)} trajectory file(s): "
        + ", ".join(csv_files)
    )

    for fname in csv_files:
        fpath            = os.path.join(req_path, fname)
        robot_type, rows = _infer_robot_type_and_rows(fpath)

        if not rows:
            logger.warn(f"  [{req_label}] Empty CSV, skipping: {fname}")
            continue

        # Derive robot name from filename stem.
        # e.g.  "F1_delivery_bot1.csv"  ->  stem="F1_delivery_bot1"  ->  "delivery_bot1"
        #        "R1_cleaning_robot.csv" ->  stem="R1_cleaning_robot"  ->  "cleaning_robot"
        stem       = fname[:-4]                    # strip ".csv"
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
            dd.omega = [_safe_float(r.get("omega"))  for r in rows]
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

        msg.robot_trajectories.append(robot_msg)
        logger.info(
            f"  [{req_label}] {robot_name} ({robot_type}): "
            f"{len(times)} pts, duration={times[-1]:.2f}s"
        )

    return msg if msg.robot_trajectories else None


# ─────────────────────────────────────────────
# Node
# ─────────────────────────────────────────────

class PathPublisher(Node):
    def __init__(self, node_name):
        super().__init__(node_name)
        self.path_pub = self.create_publisher(
            RobotTrajectoryArray, '/path_planner/paths', 10
        )
        # Track which run_00N/req_00N combos have already been published
        self._published: set = set()

        self.create_timer(1.0, self.check_for_new_path)
        self.get_logger().info(f"PathPublisher started. Watching: {DIR}")

    # ------------------------------------------------------------------
    def check_for_new_path(self):
        # 1. Ensure published_paths root exists
        if not os.path.isdir(PUBLISHED_ROOT):
            os.makedirs(PUBLISHED_ROOT, exist_ok=True)
            self.get_logger().info(f"Created published_paths dir: {PUBLISHED_ROOT}")

        # 2. Scan trajectory_logs for run_00N dirs
        try:
            entries = os.listdir(DIR)
        except FileNotFoundError:
            self.get_logger().warn(f"Trajectory logs dir not found: {DIR}")
            return

        run_dirs = sorted(
            e for e in entries
            if os.path.isdir(os.path.join(DIR, e)) and _is_run_dir(e)
        )

        for run_name in run_dirs:
            run_path     = os.path.join(DIR, run_name)
            pub_run_path = os.path.join(PUBLISHED_ROOT, run_name)

            # Ensure mirror run dir exists under published_paths
            if not os.path.isdir(pub_run_path):
                os.makedirs(pub_run_path, exist_ok=True)
                self.get_logger().info(f"Created published run dir: {pub_run_path}")

            # 3. Scan run_00N for req_00N dirs
            try:
                req_entries = os.listdir(run_path)
            except FileNotFoundError:
                continue

            req_dirs = sorted(
                r for r in req_entries
                if os.path.isdir(os.path.join(run_path, r)) and _is_req_dir(r)
            )

            for req_name in req_dirs:
                req_src = os.path.join(run_path, req_name)
                req_dst = os.path.join(pub_run_path, req_name)
                uid     = f"{run_name}/{req_name}"   # e.g. run_001/req_003

                # 4. Copy req dir into published_paths if not already there
                if not os.path.isdir(req_dst):
                    shutil.copytree(req_src, req_dst)
                    self.get_logger().info(f"Copied: {req_src}  ->  {req_dst}")

                # 5. Publish only once per req dir
                if uid in self._published:
                    continue

                self.get_logger().info(f"Publishing trajectories from {uid} ...")
                msg = _build_msg_from_req_dir(req_dst, self.get_logger(), uid)

                if msg is not None:
                    self.path_pub.publish(msg)
                    self.get_logger().info(
                        f"Published {len(msg.robot_trajectories)} trajectory(ies) "
                        f"from {uid} on /path_planner/paths"
                    )
                else:
                    self.get_logger().warn(
                        f"No valid trajectories found in {uid}, skipping publish."
                    )

                # Mark as processed regardless so we don't retry endlessly
                self._published.add(uid)

    # ------------------------------------------------------------------
    def destroy_node(self):
        """
        On shutdown: rename every CSV file inside each req_NNN dir by
        appending '_archived' before the extension.
        e.g.  F1_delivery_bot1.csv  ->  F1_delivery_bot1_archived.csv

        Directory structure (run_NNN / req_NNN) is left completely untouched.
        published_paths is never touched.
        """
        self.get_logger().info("Shutting down -- archiving CSV files in trajectory_logs ...")
        try:
            entries = os.listdir(DIR)
        except FileNotFoundError:
            entries = []

        for run_name in sorted(entries):
            if run_name == "published_paths":
                continue
            run_path = os.path.join(DIR, run_name)
            if not (os.path.isdir(run_path) and _is_run_dir(run_name)):
                continue

            try:
                req_entries = os.listdir(run_path)
            except FileNotFoundError:
                continue

            for req_name in sorted(req_entries):
                req_path = os.path.join(run_path, req_name)
                if not (os.path.isdir(req_path) and _is_req_dir(req_name)):
                    continue

                try:
                    files = os.listdir(req_path)
                except FileNotFoundError:
                    continue

                for fname in sorted(files):
                    if not fname.endswith(".csv"):
                        continue
                    # Skip files already archived
                    if fname.endswith("_archived.csv"):
                        continue

                    stem        = fname[:-4]                          # strip .csv
                    new_name    = f"{stem}_archived.csv"
                    src         = os.path.join(req_path, fname)
                    dst         = os.path.join(req_path, new_name)
                    os.rename(src, dst)
                    self.get_logger().info(
                        f"  Renamed: {run_name}/{req_name}/{fname}"
                        f"  ->  {new_name}"
                    )

        self.get_logger().info("Archiving complete.")
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