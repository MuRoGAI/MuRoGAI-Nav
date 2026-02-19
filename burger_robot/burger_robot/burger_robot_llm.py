#!/usr/bin/env python3

import os
import json
import time
import openai

import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool

from dataclasses import dataclass
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from ament_index_python.packages import get_package_share_directory

from navigation_manager_interface.msg import NavigationRobotRequest, RobotGoalStatus, CancelNavigationRequest

ROBOT_NAME   = 'burger1'
ROBOT_TYPE   = 'Differential Drive Robot'
NODE_NAME    = ROBOT_NAME + '_llm_node'
PACKAGE_NAME = 'burger_robot'


class TaskCancelledException(Exception):
    """Custom exception to signal task cancellation"""
    pass


# Define available actions
@dataclass
class TestOption:
    name: str
    id: int
    description: str
    example_code: str

option_list = [
    TestOption(
        name='Goto',
        id=0,
        # description='Navigate to any location. when it is a team task, you must specify formation robots as a list.',
        description='Navigate to any location.',
        example_code="node.goto(goal='fridge')"
    ),
    # TestOption(
    #     name='Find',
    #     id=1,
    #     description='Find and locate any object or person in the environment.',
    #     example_code="node.find('chair')"
    # )
]


class RobotLLMNode(Node):
    """
    A hub node that:
      - Listens to chat topics (/chat/output, /chat/history, /chat/task_status)
      - Publishes task status updates (/chat/task_status) and robot-specific status (<robot_name>_task_status)
      - Mirrors/consumes robot state messages on /robot_states (expects JSON strings)
    """

    _instance = None

    def __init__(self) -> None:
        super().__init__(NODE_NAME)

        # ---- Parameters ----
        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        self.declare_parameter("config_file", "robot_config_restaurant")
        cfg_file_name = self.get_parameter("config_file").get_parameter_value().string_value
        self.config_file = cfg_file_name + '.json'
        
        package_share = get_package_share_directory("chatty")
        cfg_path = os.path.join(package_share, "config", self.config_file)

        with open(cfg_path, 'r') as f:
            self.robot_config = json.load(f)

        RobotLLMNode._instance = self

        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_task = ""
        self.robot_states = {}

        # ========== CANCELLATION MECHANISM ==========
        self._task_cancelled = False  # Simple boolean flag
        # ============================================

        # ========== GOAL STATUS FLAG ==========
        self._goal_reached = None  # None=waiting, True=success, False=failed
        # ======================================

        # GROUPS
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # ---- Publishers ----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # self._goto_client = self.create_client(GotoPoseHolonomic, "/r1/goto_pose", callback_group=self.multi_group)
        # self._cancel_goto_pub = self.create_publisher(Bool, "/r1/cancel_goto_pose_goal", 10)
        self._request_goto_pub = self.create_publisher(NavigationRobotRequest, '/navigation/request', 10)
        self._cancel_goto_pub = self.create_publisher(CancelNavigationRequest, "/navigation/cancel", 10)

        # ---- Subscriptions ----
        self.sub_robot_states = self.create_subscription(
            String, '/robot_states', self.on_robot_states, 10,
            callback_group=self.seq_group
        )
        self.current_time_sub = self.create_subscription(
            String, '/current_time', self.on_current_time, 10,
            callback_group=self.seq_group
        )
        self.sub_tasks_json = self.create_subscription(
            String, '/task_manager/tasks_json', self.on_tasks_json, 10,
            callback_group=self.multi_group
        )
        self.sub_chat_output = self.create_subscription(
            String, '/chat/output', self.on_chat_output, 10,
            callback_group=self.single_group
        )

        # self.sub_chat_history = self.create_subscription(
        #     String, '/chat/history', self.on_chat_history, 10
        # )
        self.sub_chat_task_status = self.create_subscription(
            String, '/chat/task_status', self.on_chat_task_status, 10
        )

        self.goal_status_sub_ = self.create_subscription(
            RobotGoalStatus, "/controller/goal_status", self.goal_status_callback, 10
        )

        # ---- Timer Callbacks ----
        # self.timer_period = 1.0  # seconds
        # self.robot_task_status_callback = self.create_timer(
        #     self.timer_period, self.robot_task_status_update,
        #     callback_group=self.multi_group
        # )

        self.get_logger().info(
            f'RobotLLMNode started for robot="{self.robot_name}". '
            f'Publishing robot task status on "{robot_task_topic}".'
        )

        directry = "data"
        package_path = get_package_share_directory(PACKAGE_NAME)

        script_name = self.robot_name+"_chat_history.txt"
        self.history_file = os.path.join(package_path, directry, script_name)

        script_name = self.robot_name+'_task_history.txt'
        self.robot_task_history = os.path.join(package_path, directry, script_name)

        self.clear_files()
        self.robot_has_no_current_task()

    @classmethod
    def get_instance(cls):
        """Safely get or create the singleton node."""

        if cls._instance is None:
            raise RuntimeError("RobotLLMNode has not been created yet! Did you run the node?")
        return cls._instance

    # ========== CANCELLATION HELPERS ==========
    def check_cancelled(self):
        """Check if task has been cancelled. Raise exception if true."""
        if self._task_cancelled:
            self.get_logger().warn("Task cancellation detected!")
            raise TaskCancelledException("Task was cancelled")

    def reset_cancellation(self):
        """Clear the cancellation flag (call before starting new task)"""
        self._task_cancelled = False
    # ==========================================

    def clear_files(self) -> None:
        """Clear the chat history file on startup."""
        if os.path.exists(self.history_file):
            with open(self.history_file, "w") as file:
                file.write("")
            self.get_logger().info("Cleared chat history file on startup.")
        else:
            with open(self.history_file, "w") as file:
                file.write("")
            self.get_logger().warn(f"Chat history file not found. Created new file: {self.history_file}")

        if os.path.exists(self.robot_task_history):
            with open(self.robot_task_history, "w") as file:
                file.write("")
            self.get_logger().info("Cleared the robot task history file on startup.")
        else:
            self.get_logger().warn(f"Robot Task history file not found: {self.robot_task_history}")
            with open(self.robot_task_history, "w") as file:
                file.write("")
            self.get_logger().warn(f"Robot task history file not found. Created new file: {self.robot_task_history}")

    # -------------------- Callbacks --------------------

    def on_robot_states(self, msg: String) -> None:
        """Handle robot state updates (expects JSON string in msg.data)."""
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = rs
                self.get_logger().debug(f"robot_states keys: {list(self.robot_states.keys())}")
            else:
                self.get_logger().warning("/robot_states missing or not a dict; ignoring payload.")
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")

    def on_tasks_json(self, msg: String) -> None:
        """Handle tasks JSON from task manager."""
        self.get_logger().debug(f'Received /task_manager/tasks_json: {msg.data}')
        try:
            data = json.loads(msg.data)
            robot_tasks = {
                key.lower().replace(" ", "_"): value
                for key, value in data.get("robot_tasks", {}).items()
            }

            robot_names = list(robot_tasks.keys())

            if self.robot_name not in robot_names:
                self.get_logger().debug(
                    f"Exact key {self.robot_name} not found in robot_tasks. Ignoring message."
                )
                return

            robot_task = robot_tasks.get(self.robot_name, "").strip()
            self.get_logger().debug(f"{self.robot_name} task fetched [1]")

            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_task', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_task'} task fetched [2]")

            if not robot_task:
                robot_task = robot_tasks.get(self.robot_name+'_tasks', "").strip()
                self.get_logger().debug(f"{self.robot_name+'_tasks'} task fetched [3]")

            if not robot_task:
                robot_task = f"No {self.robot_name} task found."
                self.get_logger().debug(robot_task)
                return
            
            if "stop" in robot_task.lower():
                self.get_logger().info(f"{self.robot_name} task: {robot_task}")
                self.robot_task_interrupted(robot_task)
                self.stop_tasks()
                msg.data = f'{self.robot_name.capitalize()} (status): STOP TASKS COMPLETED'
                self.pub_task_status.publish(msg)
                return

            self.robot_task = robot_task
            
            self.get_logger().info(f"{self.robot_name} task: {robot_task}")
            
            self.get_logger().info("Robot Task in Progress ..")
            self.robot_task_in_progress(robot_task)

            # ========== RESET CANCELLATION BEFORE NEW TASK ==========
            self.reset_cancellation()
            # =========================================================

            self.get_logger().info("Executing Robot Task ..")
            try:
                self.execute_task(robot_task)
                self.get_logger().info(f'On Task Json Task executed')
            except TaskCancelledException:
                self.get_logger().warn("Task execution was cancelled")
                self.robot_task_interrupted(robot_task)
            # status.data = f'{self.robot_name}: received {len(tasks) if isinstance(tasks, list) else 1} task set(s)'

        except json.JSONDecodeError:
            self.get_logger().warn(
                f'Received /task_manager/tasks_json with invalid JSON; raw: {msg.data}'
            )
            # status.data = f'{self.robot_name}: received invalid tasks JSON'
        # self.pub_task_status.publish(status)

    def on_chat_output(self, msg: String) -> None:
        """Handle raw chat output and save it to the history file."""
        self.get_logger().debug(f'Received /chat/output: {msg.data}')

        timestamp = self.current_time
        parts = msg.data.split("|", 1)

        if len(parts) == 2:
            role, content = parts[0].strip(), parts[1].strip()
            self.chat_entry = f"[Time: {timestamp}] {role.capitalize()}: {content}"
        else:
            role, content = "Task Manager", msg.data
            self.chat_entry = f"[Time: {timestamp}] {role}:\n{content}"

        with open(self.history_file, "a") as file:
            file.write(self.chat_entry + "\n")


    # def on_chat_history(self, msg: String) -> None:
    #     """Handle chat history stream."""
    #     # self.get_logger().debug(f'Received /chat/history len={len(msg.data)}')

    def on_chat_task_status(self, msg: String) -> None:
        """Observe task status changes (external)."""
        self.get_logger().debug(f'Observed /chat/task_status: {msg.data}')
        with open(self.history_file, "a") as file:
            file.write(msg.data + "\n")

    def on_current_time(self, msg: String) -> None:
        """Update current time from /current_time topic."""
        self.get_logger().debug(f'Received /current_time: {msg.data}')
        self.current_time = msg.data

    def goal_status_callback(self, msg: RobotGoalStatus) -> None:
        """Handle goal status updates from navigation controller."""
        robot_name = msg.robot_name
        goal_reached = msg.goal_reached

        if robot_name == self.robot_name:
            self._goal_reached = goal_reached
            
            if goal_reached:
                self.get_logger().info(f'{self.robot_name}: Navigation goal reached successfully')
                self.update_robot_state({
                    "last_goal_status": "reached",
                    "last_goal_timestamp": time.time()
                })
            else:
                self.get_logger().warn(f'{self.robot_name}: Navigation goal failed')
                self.update_robot_state({
                    "last_goal_status": "failed",
                    "last_goal_timestamp": time.time()
                })


    # -------------------- Helpers (optional) --------------------

    def update_robot_state(self, incoming: dict, robot_name: str | None = None) -> None:
        """
        Merge a partial robot-state update into self.robot_states.

        - self.robot_states may be:
            * None
            * a dict without "robot_states" (e.g., only meta fields like date)
            * a dict that already has "robot_states"
            * (optionally) a bare robot-states dict from older code; we wrap it.

        - Ensures top-level "robot_states" key exists.
        - Ensures per-robot dict exists.
        - Adds new keys and updates changed ones.
        """

        # Default robot_name to this node's robot name if not given
        if robot_name is None:
            robot_name = self.robot_name

        # 1) Ensure top-level container exists
        if self.robot_states is None:
            self.robot_states = {}
            self.get_logger().info("Created top-level robot_states container dict")

        # 2) Make sure we have a "robot_states" key at top-level
        if "robot_states" not in self.robot_states:
            # Heuristic: if current dict already *looks* like it only contains robots,
            # then wrap it under "robot_states" instead of losing it.
            if all(isinstance(v, dict) for v in self.robot_states.values()) and len(self.robot_states) > 0:
                # Wrap existing as robot_states
                self.robot_states = {"robot_states": self.robot_states}
                self.get_logger().debug("Wrapped existing dict under 'robot_states'")
            else:
                # Start fresh robot_states dict, keep other meta fields as-is
                self.robot_states["robot_states"] = {}
                self.get_logger().info("Created 'robot_states' key in top-level dict")

        robots_dict = self.robot_states["robot_states"]

        # 3) Ensure the robot entry exists
        if robot_name not in robots_dict or not isinstance(robots_dict[robot_name], dict):
            robots_dict[robot_name] = {}
            self.get_logger().info(f"Created new robot entry '{robot_name}'")

        saved_state = robots_dict[robot_name]

        # 4) Merge incoming keys into that robot's state
        for key, new_val in incoming.items():
            if key not in saved_state:
                saved_state[key] = new_val
                self.get_logger().debug(f"{robot_name}: Added '{key}' = '{new_val}'")
            else:
                old_val = saved_state[key]
                if old_val != new_val:
                    saved_state[key] = new_val
                    self.get_logger().debug(f"{robot_name}: Updated '{key}' from '{old_val}' → '{new_val}'")

        # 5) Publish the updated robot_states
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

        # 6) Log the updated root structure (just the robot_states part to keep it readable)
        self.get_logger().debug(f"robot_states now: {self.robot_states['robot_states']}")

        return

    # -------------------- Task Status Methods --------------------

    def robot_task_in_progress(self, robot_task) -> None:
        # self.task_completed = False
        # self.task_in_progress = True
        # self.task_interrupted = False
        if not robot_task:
            robot_task = self.robot_task

        task_in_progress_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK IN PROGRESS"
        self.get_logger().info(task_in_progress_msg)
        self.robot_task_status_update(task_in_progress_msg)
        return

    def robot_task_completed(self, robot_task) -> None:
        # self.task_completed = True
        # self.task_in_progress = False
        # self.task_interrupted = False

        if not robot_task:
            robot_task = self.robot_task

        task_completed_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self.get_logger().info(task_completed_msg)
        self.robot_task_status_update(task_completed_msg)
        return

    def robot_task_interrupted(self, robot_task) -> None:
        # self.task_completed = False
        # self.task_in_progress = False
        # self.task_interrupted = True
        if not robot_task:
            robot_task = self.robot_task

        task_interrupted_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK INTERRUPTED"
        self.get_logger().info(task_interrupted_msg)
        self.robot_task_status_update(task_interrupted_msg)
        return

    def robot_has_no_current_task(self) -> None:
        # self.task_completed = False
        # self.task_in_progress = False
        # self.task_interrupted = False
        robot_task = ""
        no_task_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : NO CURRENT TASK"
        self.get_logger().info(no_task_msg)
        self.robot_task_status_update(no_task_msg)
        return

    def robot_task_status_update(self, status_msg: str) -> None:
        msg = String()
        msg.data = status_msg

        # if self.task_in_progress:
        #     msg.data = self.task_in_progress_msg
        # elif self.task_completed:
        #     msg.data = self.task_completed_msg
        # elif self.task_interrupted:
        #     msg.data = self.task_interrupted_msg
        # else:
        #     msg.data = f"{self.robot_name.capitalize()} (status) : No current task."
        
        self.pub_robot_task.publish(msg)

        try:
            with open(self.robot_task_history, "a") as file:
                file.write(f"{status_msg}\n")
        except FileNotFoundError as e:
            self.get_logger().warn(f"Robot Task history file not found: {e}")

        self.get_logger().info("Robot task status updated... ")
        return

    def tasks_completed(self, task) -> None:
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)
        return

    def stop_tasks(self) -> None:
        """Stop all robot tasks (stub function)."""
        self.get_logger().info("Stopping all robot tasks...")

        msg = Bool()
        msg.data = True
        self._cancel_find_pub.publish(msg)
        self._cancel_goto_pub.publish(msg)

        self.get_logger().info("Cancel request sent to /goto/cancel")
        self.get_logger().info("Cancel request sent to /find/cancel")
        self.get_logger().info("All tasks of the robot have been stopped.")

    # -------------------- Helper Methods --------------------

    def read_chat_history(self) -> str:
        """Read the entire chat history from the persistent file."""
        self.get_logger().debug(f"Attempting to read chat history from: {self.history_file}")

        if not os.path.exists(self.history_file):
            self.get_logger().debug(f"Chat history file not found: {self.history_file}")
            return "No previous chat history."

        if not os.path.isfile(self.history_file):
            self.get_logger().warn(f"Chat history path exists but is not a file: {self.history_file}")
            return "No previous chat history."

        try:
            with open(self.history_file, "r", encoding="utf-8") as file:
                history = file.read().strip()

            if not history:
                self.get_logger().debug("Chat history file is empty.")
                return "No previous chat history."

            self.get_logger().info("Successfully loaded chat history.")
            return history

        except PermissionError:
            self.get_logger().error(f"Permission denied when reading chat history file: {self.history_file}")
            return "Error: Unable to read chat history (permission denied)."
        except OSError as e:
            self.get_logger().error(f"OS error while reading chat history file: {e}")
            return "Error: Failed to read chat history due to system issue."
        except Exception as e:
            self.get_logger().error(f"Unexpected error reading chat history: {type(e).__name__}: {e}")
            return "Error: Failed to load chat history."

    def generate_action_prompt(self, prompt: str, task: str) -> str:
        """Generate action prompt for LLM."""
        self.get_logger().info(f"Generating code...")

        try:
            response = openai.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {'role': 'system', 'content': prompt},
                    {'role': 'user', 'content': task}
                ],
                max_tokens=500,
                temperature=0.5,
            )
            self.get_logger().debug(f"LLM Response: {response}")

            raw = response.choices[0].message.content.strip()
            self.get_logger().info(f"Content: {raw}")
            
            if "```python" in raw:
                parts = raw.split("```python")
                explaination = parts[0].strip()
                code = parts[1].split("```")[0].strip()
                self.get_logger().debug(f"code: {code}")
                self.get_logger().debug(f"explaination: {explaination}")
                return code, explaination

            return "", ""
            
        except openai.Timeout:
            self.get_logger().error("OpenAI request TIMED OUT (no response in time)")
        except openai.AuthenticationError:
            self.get_logger().error("OpenAI authentication failed — check your API key!")
        except openai.RateLimitError:
            self.get_logger().error("OpenAI rate limit hit — slow down or upgrade plan")
        except openai.APIError as e:
            self.get_logger().error("OpenAI API error: %s", e)
        except Exception as e:
            self.get_logger().error("Unexpected error in LLM call: %s: %s", type(e).__name__, e)

        self.get_logger().warn("Returning (None, None) due to LLM failure")
        return None, None

    def build_robot_context_from_config(self) -> str:
        """
        Convert robot configuration JSON into a structured natural-language context
        for LLM consumption (no raw dumping).
        """
        cfg = self.robot_config
        lines = []

        # -------------------------------------------------
        # Robot names
        # -------------------------------------------------
        robot_names = cfg.get("robot_names", [])
        if robot_names:
            lines.append(
                f"The system contains the following robots: {', '.join(robot_names)}."
            )

        # -------------------------------------------------
        # Robot capabilities
        # -------------------------------------------------
        robot_caps = cfg.get("robot_capabilities", {})
        if robot_caps:
            lines.append("Robot capabilities:")
            for name, desc in robot_caps.items():
                lines.append(f"- {name}: {desc}")

        # -------------------------------------------------
        # Robot morphology
        # -------------------------------------------------
        robot_morph = cfg.get("robot_morphology", {})
        if robot_morph:
            lines.append("Physical and morphological constraints:")
            for name, desc in robot_morph.items():
                lines.append(f"- {name}: {desc}")

        # -------------------------------------------------
        # Robot states
        # -------------------------------------------------
        robot_states = cfg.get("robot_states", {})
        if robot_states:
            lines.append("Current robot states:")
            for name, state_dict in robot_states.items():
                state_desc = ", ".join(
                    [f"{k.replace('_', ' ')}: {v}" for k, v in state_dict.items()]
                )
                lines.append(f"- {name}: {state_desc}")

        # -------------------------------------------------
        # Task-specific rules
        # -------------------------------------------------
        task_rules = cfg.get("task_specific_rules", [])
        if task_rules:
            lines.append("Task-specific operational rules:")
            for rule in task_rules:
                lines.append(f"- {rule}")

        # -------------------------------------------------
        # Task replanning rules
        # -------------------------------------------------
        replanning_rules = cfg.get("task_replanning_rules", [])
        if replanning_rules:
            lines.append("Task replanning and failure handling rules:")
            for rule in replanning_rules:
                lines.append(f"- {rule}")

        return "\n".join(lines)

    def execute_task(self, task: str) -> None:
        """Execute the given robot task using LLM decision-making."""
        chat_history = self.read_chat_history()
        robot_context = self.build_robot_context_from_config()

        try:
            self.get_logger().info("Building system action messages")
            available_actions = "\n".join(
                [f"Function Name: {opt.name} \nFunction Description: {opt.description} (e.g., {opt.example_code})" 
                 for opt in option_list]
            )

            self.get_logger().debug(f"Action message: {available_actions}")
            self.get_logger().info("Building system messages")
            
            prompt = (
                f"You are a robot control system controlling a {ROBOT_TYPE} named '{self.robot_name}'. "
                "You must generate python code to perform the task. "
                "Based on the given task, generate code using available actions."
                f"Robot system context:\n{robot_context}\n"
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
                "To serve anything to anyone first goto stall and then goto the person. "
            )

            self.get_logger().debug(f"Prompt message: {prompt}")

        except Exception as e:
            self.get_logger().warning(f"{e}")

        code, explanation = self.generate_action_prompt(prompt, task)
        
        if code:
            self.get_logger().info(f"Generated Code:\n{code}")
            if explanation:
                self.get_logger().info(f"Explanation:\n{explanation}")

            self.get_logger().info("Calling execute_python_code...")
            execute_python_code(code, node=self)

            self.get_logger().info("Robot Task Completed.")
            self.robot_task_completed(task)
            self.tasks_completed(task)
        else:
            self.get_logger().error("Failed to generate valid code for the task.")
            self.robot_task_interrupted(task)

    def find(self, target: str) -> bool:
        """
        Find and locate a target object or person.
        
        Args:
            target: Name of the object or person to find (e.g., 'chair', 'person', 'cup')
        
        Returns:
            bool: True if target found, False otherwise
        """
        # ========== CHECK CANCELLATION ==========
        self.check_cancelled()
        # ========================================
        
        self.get_logger().info(f"Starting search for: {target}")
        
        # Update robot state: Search started
        self.update_robot_state({
            "current_task": "find",
            "task_status": "searching",
            "search_target": target,
            "activity": f"Searching for {target}"
        })
        
        # Simulate searching process
        self.get_logger().info(f"Scanning environment for {target}...")
        
        # Update state: Actively scanning
        self.update_robot_state({
            "task_status": "scanning",
            "scan_started_at": time.time(),
            "activity": f"Actively scanning for {target}"
        })
        
        # Simulate search time with cancellation checks
        search_duration = 3.0  # seconds
        start_time = time.time()
        
        while time.time() - start_time < search_duration:
            # ========== CHECK CANCELLATION IN LOOP ==========
            self.check_cancelled()
            # ================================================
            time.sleep(0.5)
            self.get_logger().debug(f"Still searching for {target}...")
        
        # Simulate find result (dummy - always succeeds for now)
        found = True
        
        if found:
            self.get_logger().info(f"Successfully found {target}")
            
            # Update state: Target found
            self.update_robot_state({
                "task_status": "target_found",
                "search_result": "success",
                "found_target": target,
                "found_at": time.time(),
                "target_location": "unknown",  # In real implementation, this would be actual coordinates
                "activity": f"Found {target}"
            })
            return True
        else:
            self.get_logger().warn(f"Failed to find {target}")
            
            # Update state: Target not found
            self.update_robot_state({
                "task_status": "target_not_found",
                "search_result": "failed",
                "searched_for": target,
                "search_failed_at": time.time(),
                "activity": f"Could not locate {target}"
            })
            return False

    def wait_for_goto_complete(self, goal: str, timeout: float = 120.0) -> bool:
        """
        Wait for goto navigation to complete.
        
        Args:
            goal: The goal location being navigated to
            timeout: Maximum time to wait in seconds (default: 120s)
        
        Returns:
            bool: True if goal reached, False if failed or timed out
        """
        self.get_logger().info(f"Waiting for navigation to {goal} to complete (timeout={timeout}s)...")
        
        # Reset goal status flag before waiting
        self._goal_reached = None
        
        start_time = time.time()
        deadline = start_time + timeout
        
        while rclpy.ok():
            # ========== CHECK CANCELLATION ==========
            try:
                self.check_cancelled()
            except TaskCancelledException:
                self.get_logger().warn("Navigation cancelled during wait")
                return False
            # ========================================
            
            # Check if we've received a goal status update
            if self._goal_reached is not None:
                if self._goal_reached:
                    elapsed = time.time() - start_time
                    self.get_logger().info(f"Navigation to {goal} completed successfully in {elapsed:.2f}s")
                    return True
                else:
                    self.get_logger().warn(f"Navigation to {goal} failed")
                    return False
            
            # Check timeout
            if time.time() > deadline:
                elapsed = time.time() - start_time
                self.get_logger().error(f"Navigation to {goal} timed out after {elapsed:.2f}s")
                return False
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
        
        self.get_logger().error("ROS shutdown during navigation wait")
        return False

    def Goto(self, goal: str, formation: list = None) -> bool:
        self.goto(goal, formation)

    def goto(self, goal: str, formation: list = None) -> bool:
        """
        Navigate to a goal location, optionally with formation.
        
        Args:
            goal: Target location name (e.g., 'fridge', 'table', 'counter')
            formation: Optional list of robot names to form a formation (default: None/empty list)
        
        Returns:
            bool: True if navigation succeeded, False otherwise
        """
        # ========== CHECK CANCELLATION ==========
        self.check_cancelled()
        # ========================================
        
        self.get_logger().info(f"Requesting navigation to goal: {goal}")
        
        # Update robot state: Navigation requested
        self.update_robot_state({
            "current_task": "goto",
            "task_status": "requesting_navigation",
            "target_goal": goal,
            "activity": f"Navigating to {goal}"
        })
        
        # Create navigation request message
        msg = NavigationRobotRequest()
        msg.robot_name = self.robot_name
        msg.goal = goal
        
        # Handle formation parameter
        if formation is None:
            msg.formation_robots = []
        elif isinstance(formation, list):
            msg.formation_robots = formation
            self.get_logger().info(f"Formation requested with robots: {formation}")
        else:
            self.get_logger().warn(f"Invalid formation type: {type(formation)}. Expected list. Using empty list.")
            msg.formation_robots = []
        
        # Update state with formation info if applicable
        if msg.formation_robots:
            self.update_robot_state({
                "formation_active": True,
                "formation_robots": msg.formation_robots,
                "formation_role": "leader"
            })
        
        # Publish navigation request
        self._request_goto_pub.publish(msg)
        
        self.get_logger().info(
            f"Navigation request sent: robot={msg.robot_name}, goal={msg.goal}, "
            # f"formation={msg.formation_robots if msg.formation_robots else 'None'}"
        )
        
        # Update state: Navigation in progress
        self.update_robot_state({
            "task_status": "navigating",
            "navigation_requested_at": time.time()
        })
        
        # Wait for navigation to complete
        success = self.wait_for_goto_complete(goal)
        
        # Update state based on result
        if success:
            self.get_logger().info(f"Navigation to {goal} completed successfully")
            self.update_robot_state({
                "task_status": "navigation_completed",
                "navigation_result": "success",
                "current_location": goal,
                "arrived_at": time.time(),
                "activity": f"Arrived at {goal}"
            })
            return True
        else:
            self.get_logger().error(f"Navigation to {goal} failed or timed out")
            self.update_robot_state({
                "task_status": "navigation_failed",
                "navigation_result": "failed",
                "failed_goal": goal,
                "failure_time": time.time(),
                "activity": f"Failed to reach {goal}"
            })
            return False

def execute_python_code(code: str, node=None):
    """Execute generated Python code safely with cancellation support."""
    print("Inside the execute python code function")

    if node is None:
        node = RobotLLMNode.get_instance()
        if node is None:
            print("CRITICAL: Could not get node instance!")
            return

    node.get_logger().info(f"Executing generated Python code: {code}")

    try:
        # ========== PASS TaskCancelledException TO EXEC CONTEXT ==========
        exec(code, {"__builtins__": {}}, {
            "node": node,
            "TaskCancelledException": TaskCancelledException
        })
        # =================================================================
        node.get_logger().info("Code executed successfully")
    except TaskCancelledException:
        node.get_logger().warn("Code execution cancelled")
        raise  # Re-raise so execute_task can handle it
    except TypeError as e:
        pass
    except Exception as e:
        node.get_logger().error("Failed to execute generated code: %s", e)


def main(args=None):
    rclpy.init(args=args)
    node = RobotLLMNode()

    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt received")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()