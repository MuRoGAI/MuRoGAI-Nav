#!/usr/bin/env python3
"""

 THIS CODE IS TO TEST NAVIGATION MANAGER

Navigation Manager Node

"""
import os
import json
import base64
import pandas as pd
from datetime import datetime

import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory

from navigation_manager_interface.msg import NavigationRobotRequest, RobotGoalStatus, CancelNavigationRequest, StopRobotsRequest
from path_planner_interface.msg import PathPlannerRequest
from nav_msgs.msg import Odometry

import math
from openai import OpenAI


# ==================================================
# Global LLM Configuration
# ==================================================

LLM_MODEL_NAME = "gpt-4.1"

def quaternion_to_yaw(qx, qy, qz, qw) -> float:
    """
    Convert quaternion to yaw angle (rotation around Z axis).
    Output is in radians.
    """
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return yaw


class NavigationManager(Node):
    def __init__(self):
        super().__init__('navigation_manager')

        self.get_logger().info("Navigation Manager starting...")

        # --------------------------------------------------
        # Declare ROS parameters (optional overrides)
        # --------------------------------------------------
        self.declare_parameter("config_file", "")
        self.declare_parameter("map_metadata_file", "")
        self.declare_parameter("map_image_file", "")
        self.declare_parameter("save_data_file", "")

        # --------------------------------------------------
        # Initialize OpenAI Client
        # --------------------------------------------------
        self.llm_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        self.test_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.excel_filename = f"navigation_tests_{timestamp}.xlsx"

        # --------------------------------------------------
        # Resolve and load files (SIMPLIFIED)
        # --------------------------------------------------
        self.config_file_path = self.resolve_file(
            "config_file", "chatty", "config/robot_config_restaurant.json"
        )
        self.map_metadata_path = self.resolve_file(
            "map_metadata_file", "navigation_manager", "data/restaurant_5.json"
        )
        self.map_image_path = self.resolve_file(
            "map_image_file", "navigation_manager", "data/restaurant_5.png"
        )

        self.data_path = self.resolve_file(
            "save_data_file", "navigation_manager", f"data/{self.excel_filename}"
        )

        self.config_data = self.load_json(self.config_file_path)
        self.map_data = self.load_json(self.map_metadata_path)
        self.map_image_base64 = self.load_image_base64(self.map_image_path)


        # --------------------------------------------------
        # Create odometry subscribers for all robots
        # --------------------------------------------------
        self.robot_odom_subs_ = {}
        self.robot_latest_pose_ = {}

        # --------------------------------------------------
        # Persistent robot goal memory
        # --------------------------------------------------
        self.robot_goal_memory = {}

        self.offsets = {
            "robot1": (4.0, 9.0, 0.0),
            "robot2": (4.0, 7.0, 0.0),
            "robot3": (19.0, 12.0, 3.14),
            "robot4": (20.0, 11.0, 3.14),
            "robot5": (20.0, 13.0, 3.14),
            "robot6": (5.0, 8.0, 0.0),
        }

        robot_names = list(self.offsets.keys())

        if not robot_names:
            self.get_logger().warning(f"No robot_names found in {self.config_file_path}")
        else:
            for robot_name in robot_names:

                # -----------------------------------------
                # Initialize default pose from offsets
                # -----------------------------------------
                if robot_name in self.offsets:
                    x, y, yaw = self.offsets[robot_name]

                    self.robot_latest_pose_[robot_name] = {
                        "x": float(x),
                        "y": float(y),
                        "yaw": float(yaw)
                    }

                    self.get_logger().info(
                        f"Initialized default pose for {robot_name}: "
                        f"x={x}, y={y}, yaw={yaw}"
                    )
                else:
                    self.get_logger().warning(
                        f"No default offset found for {robot_name}"
                    )

                # -----------------------------------------
                # Subscribe to odometry
                # -----------------------------------------
                topic_name = f"/{robot_name}/odom_world"

                self.get_logger().info(f"Subscribing to odometry: {topic_name}")

                sub = self.create_subscription(
                    Odometry,
                    topic_name,
                    lambda msg, rn=robot_name: self.odom_callback(msg, rn),
                    10
                )

                self.robot_odom_subs_[robot_name] = sub


        eg_formation = {
            "F1": {
                "centroid_x": 3.5,
                "centroid_y": 6.3,
                "formation_yaw": 1.2,
                "desired_radius": 1.0,
                "robots": [ "robot1", "robot2" ]
            },
            "R1": [
                { "robot": "robot7", "x": 3.5, "y": 3.2, "yaw": 2.6 }
            ],
            "R2": [
                { "robot": "robot10", "x":9.4, "y": 7.3, "yaw": 2.6 }
            ],
            "F2": {
                "centroid_x": 4.2,
                "centroid_y": 5.0,
                "formation_yaw": 1.5,
                "desired_radius": 1.5,
                "robots": [ "robot5", "robot4", "robot6" ]
            },
            "R3": [
                { "robot": "robot15", "x":6.4, "y": 2.5, "yaw": 0.0 }
            ],
        }


        self.system_prompt = (
            "   You are a Navigation Manager for a multi-robot system. "
            "   Your job is to decide where each robot should navigate based on a request. "
            "   The robots can goto the goal position alone also in a formation with other robot. "
            "   Your output must be what should be the robot location if it has to goto goal. It must not collide with any objects or any other robots. "
            "   Incomplete goal must be resent. "
            "\n"
            # "   INPUTS:"
            # f"   Environment and robot configuration data: {json.dumps(self.config_data)}. "
            # f"   Map metadata information: {json.dumps(self.map_data)}. "
            # f"   Base64 encoded map image:{self.map_image_base64}.Position units for all x and y values are meters. "
            # # "   Also the Map metadata attached with the image. "
            "   INPUTS PROVIDED:"
            f"   1. Environment and robot configuration data (JSON): {json.dumps(self.config_data)}"
            f"   2. Map metadata with object positions and dimensions (JSON): {json.dumps(self.map_data)}"
            "   3. Map image showing the environment layout with grid lines for position estimation. Position units for all x and y values are meters."
            "\n"
            "   Refer to the grid lines in image it will help you in estimating the position more accurately, also use Map metadata for this"
            "   You must generate a navigation plan in JSON format only. "
            "   Robot names used in the output must exactly match one of the values listed in the robot_names array inside the provided configuration data and must not introduce any new or modified robot names. "
            "   If robots must move in formation, group them using a key starting with 'F' followed by a formation id such as F1,F2,F3. "
            "   If there is any Formation or single Robot in the assigned id, then that id can not be replacable, for that also goal pose must be found. "
            "   For any formation group key starting with 'F', the value must be an object containing the fields centroid_x, centroid_y, formation_yaw, desired_radius, and robots. "
            "   The centroid_x and centroid_y fields must be floating point values representing the formation center position in meters. "
            "   The formation_yaw field must be a floating point value expressed in radians and must not be expressed in degrees. "
            "   The desired_radius field must be a floating point value representing the desired formation radius in meters and must be greater than zero. "
            # "   The robots field must be a list of robot objects with fields:robot,x,y,yaw. "
            "   If a robot moves individually, group it using a key starting with 'R' followed by a robot id such as R1,R2 and the value must be a list of robot objects. "
            "   The x and y fields must be floating point values representing position in meters. "
            "   The yaw field must be a floating point value expressed in radians and must not be expressed in degrees. "
            # "   Each robot object may additionally include the following optional constraint and metadata fields and must preserve them when provided: "
            # "   radius (floating point meters) representing the robot physical footprint used for collision avoidance, "
            # "   max_velocity (floating point meters per second) representing the maximum allowed linear speed of the robot, "
            # "   max_angular_velocity_z (floating point radians per second) representing the maximum allowed angular speed of the robot, "
            # "   colour (string) representing robot visualization or identification label, "
            # "   type (string) representing the robot drive model such as Differential Drive, Omnidirectional Drive, or Holonomic Drive. "
            # "   These fields must not be modified arbitrarily and must remain consistent with the provided configuration and example format. "
            "   For formation groups, formation_yaw represents the global orientation of the entire formation in radians and applies to all robots inside that formation. "
            # "   Individual robot yaw still represents each robot's own heading relative to the world frame. "
            "   When generating goal positions, you must respect robot radius to avoid collisions with obstacles and other robots, "
            # "   and respect max_velocity and max_angular_velocity_z as motion constraints when determining feasible target poses. "
            "   While generating goal positions if any robot/s is asked to goto any object, the position must have a safe distance from the object so that there wont be any collision. "
            "   Robots must maintain safe distance that does not mean robot goal should be far  from actual goal"
            "   While generating goal pose for formation the goal pose of the EVERY robot in formation SHOULD NOT collide with any object. "
            "   It is always better to keep give distance from your calculated goal. Factor of Safety. "
            "   You can use the 'rotation' of each object in the Map metadata information to check the in which direction the object is faced. "
            "   The output must be in the json type. "
            "   Orientation of robot or formation should point towards the object. "
            "   Position of robot should be within 2 meters PROXIMITY to the object and NOT directly on the object. "
            "   Robot should not collide with the object. "
            "   Formation centroid can lie on the object. "

            f" Example format: {json.dumps(eg_formation)}."
            "  In all field of teh output like x, y, yaw, you have to say where the robot should go not where robot is. "
            # "   All field names shown in the example must always be preserved exactly in the output JSON, including: centroid_x, centroid_y, formation_yaw, desired_radius, robot, x, y, yaw, radius, max_velocity, max_angular_velocity_z, colour, and type. These keys must not be renamed, omitted, or reformatted. "
        )





        self.system_prompt = (
            "   You are a Navigation Manager for a multi-robot system. "
            "   Your job is to decide where each robot should navigate based on a request. "
            "   The robots can goto the goal position alone also in a formation with other robot. "
            "   Your output must be what should be the robot location if it has to goto goal. It must not collide with any objects or any other robots. "
            "   Incomplete goal must be resent. "
            "\n"
            "   INPUTS PROVIDED:"
            f"   1. Environment and robot configuration data (JSON): {json.dumps(self.config_data)}"
            f"   2. Map metadata with object positions and dimensions (JSON): {json.dumps(self.map_data)}"
            "   3. Map image showing the environment layout with grid lines for position estimation. Position units for all x and y values are meters."
            "\n"
            "   Refer to the grid lines in image it will help you in estimating the position more accurately, also use Map metadata for this"
            "   You must generate a navigation plan in JSON format only. "
            "   Robot names used in the output must exactly match one of the values listed in the robot_names array inside the provided configuration data and must not introduce any new or modified robot names. "
            "   If robots must move in formation, group them using a key starting with 'F' followed by a formation id such as F1,F2,F3. "
            "   If there is any Formation or single Robot in the assigned id, then that id can not be replacable, for that also goal pose must be found. "
            "   For any formation group key starting with 'F', the value must be an object containing the fields centroid_x, centroid_y, formation_yaw, desired_radius, and robots. "
            "   The centroid_x and centroid_y fields must be floating point values representing the formation center position in meters. "
            "   The formation_yaw field must be a floating point value expressed in radians and must not be expressed in degrees. "
            "   The desired_radius field must be a floating point value representing the desired formation radius in meters and must be greater than zero. "
            "   If a robot moves individually, group it using a key starting with 'R' followed by a robot id such as R1,R2 and the value must be a list of robot objects. "
            "   The x and y fields must be floating point values representing position in meters. "
            "   The yaw field must be a floating point value expressed in radians and must not be expressed in degrees. "
            "   For formation groups, formation_yaw represents the global orientation of the entire formation in radians and applies to all robots inside that formation. "
            "   When generating goal positions, you must respect robot radius to avoid collisions with obstacles and other robots, "
            "   While generating goal positions if any robot/s is asked to goto any object, the position must have a safe distance from the object so that there wont be any collision. "
            "   Robots must maintain safe distance that does not mean robot goal should be far  from actual goal"
            "   While generating goal pose for formation the goal pose of the EVERY robot in formation SHOULD NOT collide with any object. "
            "   It is always better to keep give distance from your calculated goal. Factor of Safety. "
            "   You can use the 'rotation' of each object in the Map metadata information to check the in which direction the object is faced. "
            "   The output must be in the json type. "
            "   Orientation of robot or formation should point towards the object. "
            "   Position of robot should be within 2 meters PROXIMITY to the object and NOT directly on the object. "
            "   Robot should not collide with the object. "
            "   Formation centroid can lie on the object. "
            f" Example format: {json.dumps(eg_formation)}."
            "  In all field of teh output like x, y, yaw, you have to say where the robot should go not where robot is. "
        )

        dynamic_system_prompt = (
            self.system_prompt
            + f" Current live robot poses in JSON format where each robot contains x,y,yaw in meters and radians:{robot_pose_string}."
            + f" Current assigned robot goals in JSON format where the top-level keys are robot names exactly matching robot_names and each value is an object with fields group(string),x(float meters),y(float meters),yaw(float radians) representing the last assigned goal for that robot:{goal_memory_string}. for all the last assigned goal must be send always. This list only contains the incomplete goal, so you have to always send this also. While resending old goals do not change its goal. Old goals have stored its own goals so do not change it."
        )



        # --------------------------------------------------
        # Subscriber
        # --------------------------------------------------
        self.navigate_request_sub_ = self.create_subscription(
            NavigationRobotRequest,
            '/navigation/request',
            self.navigate_request_callback,
            10
        )
        dynamic_system_prompt

        # --------------------------------------------------
        # Controller feedback subscriber
        # --------------------------------------------------
        self.goal_status_sub_ = self.create_subscription(
            RobotGoalStatus,
            "/controller/goal_status",
            self.goal_status_callback,
            10
        )

        # --------------------------------------------------
        # Cancel navigation subscriber
        # --------------------------------------------------
        self.cancel_sub_ = self.create_subscription(
            CancelNavigationRequest,
            "/navigation/cancel",
            self.cancel_navigation_callback,
            10
        )


        # --------------------------------------------------
        # Path planner publisher
        # --------------------------------------------------
        self.path_plan_pub_ = self.create_publisher(
            PathPlannerRequest,
            "/path_planner/request",
            10
        )

        # --------------------------------------------------
        # NEW: Stop robots publisher
        # --------------------------------------------------
        self.stop_robots_pub_ = self.create_publisher(
            StopRobotsRequest,
            "/navigation/stop_robots",
            10
        )

        self.get_logger().info("Navigation Manager ready to receive requests.")

    # ==================================================
    # Simplified Helpers
    # ==================================================

    def resolve_file(self, param_name: str, default_pkg: str, relative_path: str) -> str:
        param_value = self.get_parameter(param_name).value
        if param_value:
            self.get_logger().info(f"Using {param_name} from parameter: {param_value}")
            return param_value

        pkg_path = get_package_share_directory(default_pkg)
        default_path = os.path.join(pkg_path, relative_path)

        self.get_logger().info(f"Using default {param_name}: {default_path}")
        return default_path

    def load_json(self, path: str) -> dict:
        if not os.path.exists(path):
            self.get_logger().error(f"JSON file not found: {path}")
            return {}

        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.get_logger().error(f"Failed to load JSON {path}: {e}")
            return {}

    def load_image_base64(self, path: str) -> str:
        if not os.path.exists(path):
            self.get_logger().warning(f"Image not found: {path}")
            return ""

        try:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode("utf-8")
        except Exception as e:
            self.get_logger().error(f"Failed to load image {path}: {e}")
            return ""

    def extract_robot_names_from_plan(self, plan: dict) -> list:
        """
        Extract all robot names from the plan JSON.
        
        Args:
            plan: Dictionary containing the navigation plan
            
        Returns:
            List of robot names found in the plan
        """
        robot_names = []
        
        for group_name, group_data in plan.items():

            # -------- Formation group (F*) --------
            if group_name.startswith("F") and isinstance(group_data, dict):
                robots = group_data.get("robots", [])

                for robot_name in robots:   # <-- already string
                    if robot_name and robot_name not in robot_names:
                        robot_names.append(robot_name)

            # -------- Individual group (R*) --------
            elif group_name.startswith("R") and isinstance(group_data, list):

                for robot_entry in group_data:
                    robot_name = robot_entry.get("robot")

                    if robot_name and robot_name not in robot_names:
                        robot_names.append(robot_name)

        
        return robot_names

    def publish_stop_robots(self, robot_names: list, reason: str):
        """
        Publish a stop command for the given robots.
        
        Args:
            robot_names: List of robot names to stop
            reason: Reason for stopping
        """
        if not robot_names:
            self.get_logger().warning("No robot names to stop")
            return
        
        msg = StopRobotsRequest()
        msg.robot_names = robot_names
        msg.reason = reason
        
        self.stop_robots_pub_.publish(msg)
        
        self.get_logger().info(
            f"Published stop command for {len(robot_names)} robots: {robot_names}. Reason: {reason}"
        )

    def update_robot_goal_memory(self, plan: dict, goal: str):
        """
        Extract robot goals from LLM plan and store them persistently.
        """
        for group_name, group_data in plan.items():

            # ------------------------
            # Formation group (F*)
            # ------------------------
            if group_name.startswith("F") and isinstance(group_data, dict):

                robots = group_data.get("robots", [])

                for robot_name in robots:

                    if not robot_name:
                        continue

                    # Store formation reference only
                    self.robot_goal_memory[robot_name] = {
                        "group": group_name,
                        "formation": True,
                        "goal": goal
                    }

            # ------------------------
            # Individual group (R*)
            # ------------------------
            elif group_name.startswith("R") and isinstance(group_data, list):
                for robot_entry in group_data:
                    robot_name = robot_entry.get("robot")
                    if not robot_name:
                        continue

                    self.robot_goal_memory[robot_name] = {
                        "group": group_name,
                        "x": float(robot_entry.get("x", 0.0)),
                        "y": float(robot_entry.get("y", 0.0)),
                        "yaw": float(robot_entry.get("yaw", 0.0)),
                        "goal": goal
                    }



    def odom_callback(self, msg: Odometry, robot_name: str):
        position = msg.pose.pose.position
        orientation = msg.pose.pose.orientation

        yaw = quaternion_to_yaw(
            orientation.x,
            orientation.y,
            orientation.z,
            orientation.w
        )

        self.robot_latest_pose_[robot_name] = {
            "x": float(position.x),
            "y": float(position.y),
            "yaw": float(yaw)
        }

    def goal_status_callback(self, msg: RobotGoalStatus):
        robot_name = msg.robot_name
        goal_reached = msg.goal_reached

        if not goal_reached:
            return

        if robot_name in self.robot_goal_memory:
            completed_goal = self.robot_goal_memory.pop(robot_name)

            self.get_logger().info(
                f"Robot '{robot_name}' reached its goal and was removed from goal memory: {completed_goal}"
            )
        else:
            self.get_logger().warning(
                f"Received goal reached for '{robot_name}', but no active goal was found in memory."
            )

    def cancel_navigation_callback(self, msg: CancelNavigationRequest):
        robot_name = msg.robot_name
        reason = msg.reason
        formation_robots = msg.formation_robots

        self.get_logger().warning(
            f"Cancel request received for robot '{robot_name}'. Reason: {reason}"
        )

        # -----------------------------
        # Remove from goal memory
        # -----------------------------
        if robot_name in self.robot_goal_memory:
            removed_goal = self.robot_goal_memory.pop(robot_name)

            self.get_logger().info(
                f"Cancelled goal for robot '{robot_name}': {removed_goal}"
            )
        else:
            self.get_logger().warning(
                f"Robot '{robot_name}' had no active goal to cancel."
            )

        # -----------------------------
        # Replan for remaining robots
        # -----------------------------
        self.request_plan(
            trigger="cancel",
            robot_name=robot_name,
            goal=None,
            formation_robots=[]
        )

    def navigate_request_callback(self, msg: NavigationRobotRequest):

        # RESET goal memory before processing new request
        self.robot_goal_memory.clear()
        self.get_logger().info("🔄 Goal memory cleared for fresh test")


        robot_namespace = msg.robot_name
        target_goal = msg.goal
        formation_robots = msg.formation_robots

        self.get_logger().info(
            f"[Navigation Request] Robot: '{robot_namespace if robot_namespace else formation_robots}' | Goal: '{target_goal}'"
        )

        self.request_plan(
            trigger="navigation",
            robot_name=robot_namespace,
            goal=target_goal,
            formation_robots=formation_robots
        )


    def extract_json_from_llm(self, text: str):
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start == -1 or end == -1:
                raise ValueError("No JSON object found")

            json_str = text[start:end+1]
            return json.loads(json_str)

        except Exception as e:
            raise ValueError(f"Failed to extract JSON: {e}")



    def request_plan(self, trigger: str, robot_name: str, goal, formation_robots):
        """
        Unified planner pipeline for navigation + cancel events.
        """

        # --------------------------------------------------
        # Build live robot pose string
        # --------------------------------------------------
        robot_pose_string = (
            json.dumps(self.robot_latest_pose_)
            if self.robot_latest_pose_
            else "{}"
        )

        # --------------------------------------------------
        # Build persistent goal memory string
        # --------------------------------------------------
        goal_memory_string = (
            json.dumps(self.robot_goal_memory)
            if self.robot_goal_memory
            else "{}"
        )

        # --------------------------------------------------
        # Build dynamic system prompt
        # --------------------------------------------------
        dynamic_system_prompt = (
            self.system_prompt
            + f" Current live robot poses in JSON format where each robot contains x,y,yaw in meters and radians:{robot_pose_string}."
            + f" Current assigned robot goals in JSON format where the top-level keys are robot names exactly matching robot_names and each value is an object with fields group(string),x(float meters),y(float meters),yaw(float radians) representing the last assigned goal for that robot:{goal_memory_string}. for all the last assigned goal must be send always. This list only contains the incomplete goal, so you have to always send this also. While resending old goals do not change its goal. Old goals have stored its own goals so do not change it."
        )
        
        # --------------------------------------------------
        # Build user prompt
        # --------------------------------------------------
        if trigger == "navigation":
            if not formation_robots :
                user_prompt = (
                    f" Robot Name: {robot_name},"
                    f" Goal Position: {goal},"
                    # f" Formation Robots: {formation_robots}."
                )
            else :
                user_prompt = (
                    # f" Robot Name: {robot_name},"
                    f" Formation Robots: {formation_robots}."
                    f" Goal Position: {goal},"
                )

        elif trigger == "cancel":
            user_prompt = (
                f" Robot '{robot_name}' was cancelled."
                " Replan navigation for remaining robots only."
            )

        else:
            user_prompt = "Replan navigation."

        # --------------------------------------------------
        # Query LLM
        # --------------------------------------------------
        llm_response = self.query_llm(dynamic_system_prompt, user_prompt)

        self.get_logger().info("LLM Response:")
        self.get_logger().info(llm_response)

        # --------------------------------------------------
        # Validate and publish
        # --------------------------------------------------
        
        try:
            plan_json = self.extract_json_from_llm(llm_response)

            robot_names_in_plan = self.extract_robot_names_from_plan(plan_json)

            if robot_names_in_plan:
                self.publish_stop_robots(
                    robot_names=robot_names_in_plan,
                    reason="new_plan_received"
                )

            self.update_robot_goal_memory(plan_json, goal)

            self.save_test_result(
                robot_name=robot_name,
                formation_robots=formation_robots,
                goal=goal,
                llm_output=plan_json,
                trigger=trigger
            )

            msg = PathPlannerRequest()
            msg.plan_json = json.dumps(plan_json)
            self.path_plan_pub_.publish(msg)

            self.get_logger().info("Published plan to /path_planner/request")

        except Exception as e:

            self.get_logger().error(f"LLM output invalid: {e}")

            # --------------------------------------------------
            # Force valid JSON fallback
            # --------------------------------------------------
            fallback_json = {
                "error": "invalid_llm_output",
                "reason": str(e),
                "raw_response": llm_response
            }

            # Save to Excel
            self.save_test_result(
                robot_name=robot_name,
                formation_robots=formation_robots,
                goal=goal,
                llm_output=fallback_json,
                trigger=trigger,
                success=False
            )

            # Publish fallback JSON so pipeline never breaks
            msg = PathPlannerRequest()
            msg.plan_json = json.dumps(fallback_json)
            self.path_plan_pub_.publish(msg)

            self.get_logger().warning("Published fallback JSON to /path_planner/request")



    def save_to_excel(self):
        """Save all test results to Excel file at the configured path"""
        if not self.test_results:
            self.get_logger().warning("No test results to save!")
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
            
            # Create DataFrame
            df = pd.DataFrame(self.test_results)
            
            # Save to Excel with formatting
            with pd.ExcelWriter(self.data_path, engine='openpyxl') as writer:
                df.to_excel(writer, sheet_name='Test Results', index=False)
                
                # Auto-adjust column widths
                worksheet = writer.sheets['Test Results']
                for idx, col in enumerate(df.columns):
                    max_length = max(
                        df[col].astype(str).apply(len).max(),
                        len(col)
                    )
                    max_length = min(max_length, 50)  # Cap at 50 chars
                    worksheet.column_dimensions[chr(65 + idx)].width = max_length + 2
            
            self.get_logger().info(
                f"💾 Saved {len(self.test_results)} results to {self.data_path}"
            )
            
        except Exception as e:
            self.get_logger().error(f"❌ Failed to save Excel: {e}")


    def save_test_result(self, robot_name: str, formation_robots: list, goal: str, 
                        llm_output: dict, trigger: str, success: bool = True):
        """
        Save a single test result to the results list.
        
        Args:
            robot_name: Name of the robot (empty if formation)
            formation_robots: List of robots in formation (empty if single robot)
            goal: The goal description
            llm_output: The parsed JSON output from LLM
            trigger: Type of trigger ('navigation' or 'cancel')
            success: Whether the LLM call was successful
        """
        
        # Determine if this is formation or individual
        is_formation = bool(formation_robots)
        
        # Extract robot names involved
        if is_formation:
            robots_involved = ", ".join(formation_robots)
        else:
            robots_involved = robot_name
        
        # Count formations and individual robots in output
        num_formations = sum(1 for k in llm_output.keys() if k.startswith('F'))
        num_individuals = sum(1 for k in llm_output.keys() if k.startswith('R'))
        
        # Create result record
        result = {
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'Trigger': trigger,
            'Request_Type': 'Formation' if is_formation else 'Individual',
            'Robot(s)': robots_involved,
            'Goal': goal,
            'Success': success,
            'Formations_Generated': num_formations,
            'Individuals_Generated': num_individuals,
            'Total_Robots_Planned': len(self.extract_robot_names_from_plan(llm_output)),
            'LLM_Output': json.dumps(llm_output, indent=2)
        }
        
        # Add detailed formation info if applicable
        if is_formation and success:
            formations = [k for k in llm_output.keys() if k.startswith('F')]
            if formations:
                formation_details = []
                for f_key in formations:
                    f_data = llm_output[f_key]
                    formation_details.append(
                        f"{f_key}: centroid=({f_data.get('centroid_x', 'N/A')}, "
                        f"{f_data.get('centroid_y', 'N/A')}), "
                        f"yaw={f_data.get('formation_yaw', 'N/A')}, "
                        f"radius={f_data.get('desired_radius', 'N/A')}, "
                        f"robots={f_data.get('robots', [])}"
                    )
                result['Formation_Details'] = "; ".join(formation_details)
        
        # Add to results list
        self.test_results.append(result)
        
        # Auto-save after each test (optional - can be disabled for performance)
        self.save_to_excel()
        
        self.get_logger().info(
            f" Saved test result #{len(self.test_results)} to tracking list"
        )

    # ==================================================
    # LLM Interface
    # ==================================================

    def query_llm(self, system_prompt: str, user_prompt: str) -> str:
        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{self.map_image_base64}"
                            }
                        }
                    ]
                }
            ]
            
            response = self.llm_client.chat.completions.create(
                model=LLM_MODEL_NAME,  # Must be vision-capable like gpt-4o
                messages=messages,
                temperature=0.1,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            self.get_logger().error(f"LLM request failed: {e}")
            return "ERROR: LLM request failed."


# ==================================================
# Main
# ==================================================

def main(argv=None):
    rclpy.init(args=argv)

    node = NavigationManager()
    
    try:
        rclpy.spin(node)
    finally:
        # Final save on shutdown
        node.save_to_excel()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
