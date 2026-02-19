#!/usr/bin/env python3

import json
import time
import re
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dataclasses import dataclass
import os
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from threading import Thread, Lock
from ament_index_python.packages import get_package_share_directory


ROBOT_NAME   = 'robots_team'
ROBOT_TYPE   = 'Team of Robots'
PACKAGE_NAME = 'robot_llm'
NODE_NAME    = 'teams_llm_node'


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
    # TestOption(
    #     name="Assemble", # GetIntoFormation
    #     id=0,
    #     description="This is used get into formation.",
    #     example_code="node.assemble()" # navigation manager selects a suitable point using the centroid of all the robots and a default radius
    # ),
    # TestOption(
    #     name="AssembleAt", #
    #     id=0,
    #     description="This is use to assemble at a particular location",
    #     example_code="node.assemble_at(goal='round table')"  # goal from position from task
    # ),
    TestOption(
        name="Goto",
        id=1,
        description="This is used to goto only after robots are already in formation",
        example_code="node.goto(goal='fridge')" # goal from position from task
    )
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

        self.declare_parameter("config_file", "robot_config_room_202")
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
                
        self.team_workers = {}      # team -> Thread
        self.team_cancel  = {}      # team -> bool
        self.team_queues  = {}      # team -> deque[str]
        self.team_lock    = Lock()

        # ========== CANCELLATION MECHANISM ==========
        self._task_cancelled = False  # Simple boolean flag
        # ============================================

        # GROUPS
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # ---- Publishers ----
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)


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

    def normalize_team_name(self, key: str) -> str | None:
        """
        Convert:
        team 5, Team_5, team5_task, team_5_tasks → team5
        """
        key = key.lower().replace(" ", "_")
        m = re.search(r"team[_]?(\d+)", key)
        if not m:
            return None
        return f"team{m.group(1)}"


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

    def cancel_team(self, team_name: str):
        with self.team_lock:
            self.team_cancel[team_name] = True

    # def run_team_tasks(self, team_name: str, tasks: list[str]) -> None:
    #     try:
    #         self.get_logger().info(f"[{team_name}] Started parallel execution: {tasks}")

    #         for task in tasks:
    #             if self._task_cancelled or self.team_cancel.get(team_name, False):
    #                 self.get_logger().warn(f"[{team_name}] Cancelled")
    #                 return

    #             task_label = f"{team_name}: {task}"
    #             self.robot_task_in_progress(task_label)

    #             self.execute_task(task_label)

    #         self.get_logger().info(f"[{team_name}] All tasks completed")
    #         self.robot_task_completed(f"{team_name} tasks")

    #     except TaskCancelledException:
    #         self.get_logger().warn(f"[{team_name}] Task cancelled")
    #         self.robot_task_interrupted(f"{team_name} tasks")

    #     except Exception as e:
    #         self.get_logger().error(f"[{team_name}] Task failed → {e}")
    #         self.robot_task_interrupted(f"{team_name} tasks")

    #     finally:
    #         # CLEANUP — THIS IS CRITICAL
    #         with self.team_lock:
    #             self.team_workers.pop(team_name, None)
    #             self.team_cancel.pop(team_name, None)

    #         self.get_logger().info(f"[{team_name}] Thread exited cleanly")


    def run_team_tasks(self, team: str) -> None:
        try:
            self.get_logger().info(f"[{team}] Thread started")

            while True:
                with self.team_lock:
                    if self.team_cancel.get(team, False):
                        self.get_logger().warn(f"[{team}] Cancelled")
                        return

                    if not self.team_queues.get(team):
                        return

                    task = self.team_queues[team].popleft()

                task_label = f"{team}: {task}"
                self.robot_task_in_progress(task_label)

                self.execute_task(task_label)

        except TaskCancelledException:
            self.get_logger().warn(f"[{team}] Task cancelled")

        except Exception as e:
            self.get_logger().error(f"[{team}] Task failed → {e}")

        finally:
            with self.team_lock:
                self.team_workers.pop(team, None)
                self.team_queues.pop(team, None)
                self.team_cancel.pop(team, None)

            self.robot_task_completed(f"{team} tasks")
            self.get_logger().info(f"[{team}] Thread exited cleanly")


    def on_tasks_json(self, msg: String) -> None:
        """Handle tasks JSON from task manager (team-based, parallel-safe)."""
        self.get_logger().debug(f"Received /task_manager/tasks_json: {msg.data}")

        try:
            data = json.loads(msg.data)

            # --------------------------------------------------
            # Normalize incoming task keys
            # --------------------------------------------------
            robot_tasks = {
                key.lower().replace(" ", "_"): value
                for key, value in data.get("robot_tasks", {}).items()
            }

            team_tasks = {}   # { "team5": [task1, task2, ...] }

            # --------------------------------------------------
            # Extract & normalize team tasks
            # --------------------------------------------------
            for key, task in robot_tasks.items():
                team = self.normalize_team_name(key)
                if not team:
                    continue

                if not isinstance(task, str) or not task.strip():
                    continue

                team_tasks.setdefault(team, []).append(task.strip())

            if not team_tasks:
                self.get_logger().debug("No team tasks found.")
                return

            self.get_logger().info(f"Team tasks detected: {team_tasks}")

            # --------------------------------------------------
            # HANDLE STOP PER TEAM (before threading)
            # --------------------------------------------------
            for team, tasks in list(team_tasks.items()):
                stop_tasks = [t for t in tasks if "stop" in t.lower()]
                if stop_tasks:
                    self.get_logger().warn(f"[{team}] STOP task received")

                    # cancel running team (if any)
                    self.cancel_team(team)

                    # publish stop status
                    status = String()
                    status.data = f"{team.upper()} (status): STOP TASKS COMPLETED"
                    self.pub_task_status.publish(status)

                    # remove STOP tasks from execution list
                    team_tasks[team] = [t for t in tasks if "stop" not in t.lower()]

            # remove teams that now have no executable tasks
            team_tasks = {t: q for t, q in team_tasks.items() if q}
            if not team_tasks:
                self.get_logger().info("Only STOP tasks received. Nothing to execute.")
                return

            # --------------------------------------------------
            # ENQUEUE TASKS + START THREADS IF NEEDED
            # --------------------------------------------------
            for team, tasks in team_tasks.items():
                with self.team_lock:

                    # create task queue if first time
                    if team not in self.team_queues:
                        from collections import deque
                        self.team_queues[team] = deque()

                    # enqueue new tasks
                    for task in tasks:
                        self.team_queues[team].append(task)
                        self.get_logger().info(f"[{team}] Queued task: {task}")

                    # if team already running → do NOT start new thread
                    if team in self.team_workers and self.team_workers[team].is_alive():
                        self.get_logger().info(
                            f"[{team}] Worker already running, tasks appended"
                        )
                        continue

                    # otherwise start a new worker thread
                    self.team_cancel[team] = False

                    worker = Thread(
                        target=self.run_team_tasks,
                        args=(team,),
                        daemon=True
                    )

                    self.team_workers[team] = worker
                    worker.start()

                    self.get_logger().info(f"[{team}] Worker thread started")

        except json.JSONDecodeError:
            self.get_logger().warn(
                f"Received /task_manager/tasks_json with invalid JSON; raw: {msg.data}"
            )


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

    def robot_task_completed(self, robot_task=None) -> None:
        # self.task_completed = True
        # self.task_in_progress = False
        # self.task_interrupted = False
        if not robot_task:
            robot_task = self.robot_task

        task_completed_msg = f"{self.robot_name.capitalize()} (status) : {robot_task} : TASK COMPLETED"
        self.get_logger().info(task_completed_msg)
        self.robot_task_status_update(task_completed_msg)
        return

    def robot_task_interrupted(self, robot_task=None) -> None:
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

    def tasks_completed(self, task=None) -> None:
        msg = String()
        msg.data = f"{self.robot_name.capitalize()} (status): ALL TASKS COMPLETED"
        self.pub_task_status.publish(msg)
        return

    def stop_tasks(self) -> None:
        self.get_logger().info("Stopping ALL team tasks")

        with self.team_lock:
            for team in self.team_cancel:
                self.team_cancel[team] = True

        # ========== SET CANCELLATION FLAG ==========
        self._task_cancelled = True
        # ===========================================

        msg = String()
        msg.data = "STOP"
        self._cancel_pub.publish(msg)

        self.get_logger().info("Cancel request sent to /start_pick/cancel")
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
                "You must generate python code that issues commands for the entire team. "
                # "You must generate python code to perform the task. "
                "Based on the given task, generate code using available actions."
                f"Robot system context:\n{robot_context}\n\n"
                f"Recent Tasks (History): {chat_history} "
                f"Current States of All Robots: {self.robot_states} "
                f"Available Actions: {available_actions} "
                "Using the class reference name same as the example is important. "
                "Use the name 'node' to refer to the RobotLLMNode instance. "
                "Your generating code note this, available actions are case-sensitive, so DO NOT change the case of any function or variable names. "
                # "IMPORTANT: Add 'node.check_cancelled()' at the start of loops and between long operations. "
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

    # —————————————————————— LLM FUNCTIONS ——————————————————————

    def assemble(self) -> bool:
        self.get_logger().info("Assemlbling ...")
        time.sleep(3)
        self.robot_task_completed(f"assemble")
        return True

    def assemble_at(self, goal='kitchen') -> bool :
        self.get_logger().info(f"Assemlbling at {goal}...")
        time.sleep(3)
        self.robot_task_completed(f"assemble at {goal}")
        return True

    def goto(self, goal='chair') -> bool :
        self.get_logger().info(f"Ging to {goal}...")
        time.sleep(3)
        self.robot_task_completed(f"goto {goal}")
        return True

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

        # node.get_logger().info("Robot Task Completed.")
        # node.robot_task_completed()
        # node.tasks_completed()

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
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()