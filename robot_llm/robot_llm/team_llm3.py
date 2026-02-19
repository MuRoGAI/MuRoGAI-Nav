#!/usr/bin/env python3

import json
import time
import re
import openai
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dataclasses import dataclass, field
import os
from rclpy.callback_groups import (
    MutuallyExclusiveCallbackGroup,
    ReentrantCallbackGroup
)
from threading import Thread, Lock
from collections import deque
from ament_index_python.packages import get_package_share_directory
from navigation_manager_interface.msg import NavigationRobotRequest, RobotGoalStatus, CancelNavigationRequest


PACKAGE_NAME = 'robot_llm'
NODE_NAME = 'team_llm_node'
ROBOT_NAME   = 'team'


class TaskCancelledException(Exception):
    """Custom exception to signal task cancellation"""
    pass


@dataclass
class TeamState:
    """Complete state tracking for a single team"""
    name: str
    robots: list[str]
    task_queue: deque = field(default_factory=deque)
    
    # Thread management
    worker_thread: Thread = None
    cancel_flag: bool = False
    
    # Status tracking
    current_task: str = ""
    status: str = "IDLE"  # IDLE, IN_PROGRESS, COMPLETED, FAILED, CANCELLED
    
    # Robot states for this team
    team_robot_states: dict = field(default_factory=dict)
    
    # Task history
    task_history: list = field(default_factory=list)
    
    # Timestamps
    created_at: float = field(default_factory=time.time)
    last_activity: float = field(default_factory=time.time)
    
    # ========== GOAL STATUS TRACKING ==========
    # Tracks navigation goal status for each robot in the team
    # Key: robot_name, Value: None (waiting), True (success), False (failed)
    robot_goal_status: dict = field(default_factory=dict)


@dataclass
class TestOption:
    name: str
    id: int
    description: str
    example_code: str


option_list = [
    # TestOption(
    #     name="assemble",
    #     id=0,
    #     description="This is used to get into formation. Both team_name and robots list are automatically provided.",
    #     example_code="node.assemble(team_name=team_name, robots=['burger1', 'waffle', 'robot6'])"
    # ),
    # TestOption(
    #     name="assemble_at",
    #     id=1,
    #     description="This is used to assemble at a particular location. Both team_name and robots list are automatically provided.",
    #     example_code="node.assemble_at(goal='round table', team_name=team_name, robots=['burger1', 'waffle', 'go2' ,'tb4'])"
    # ),
    TestOption(
        name="goto",
        id=2,
        description="This is used to goto only after robots are already in formation. Both team_name and robots list are automatically provided. The goal could be like this also 'fridge', 'red car', 'sink2', 'behind the car', 'in front of fridge', 'surround the table4', 'right side to the tall building', 'table right to the sink1'",
        example_code="node.goto(goal='fridge', team_name=team_name, robots=['burger3', 'waffle', 'robot6'])"
    )
]


class TeamLLMNode(Node):
    """
    Dynamic Team Manager that handles N teams with threading.
    Each team gets its own state, publishers, and worker thread.
    """

    _instance = None

    def __init__(self) -> None:
        super().__init__(NODE_NAME)

        self.declare_parameter('robot_name', ROBOT_NAME)
        self.robot_name: str = self.get_parameter('robot_name').value
        robot_task_topic = f'{self.robot_name}_task_status'

        # Configuration
        self.declare_parameter("config_file", "robot_config_room_202")
        cfg_file_name = self.get_parameter("config_file").get_parameter_value().string_value
        self.config_file = cfg_file_name + '.json'

        package_share = get_package_share_directory("chatty")
        cfg_path = os.path.join(package_share, "config", self.config_file)

        with open(cfg_path, 'r') as f:
            self.robot_config = json.load(f)

        TeamLLMNode._instance = self

        # ========== TEAM STATE MANAGEMENT ==========
        self.teams: dict[str, TeamState] = {}  # team_name -> TeamState
        self.team_lock = Lock()
        # ===========================================

        # Global state
        self.current_time = f"Hours: {00}, Minutes: {10}, Seconds: {00}"
        self.robot_states = {}

        # Global publishers
        self.pub_task_status = self.create_publisher(String, '/chat/task_status', 10)
        self.pub_robot_states = self.create_publisher(String, '/robot_states', 10)
        self._request_goto_pub = self.create_publisher(NavigationRobotRequest, '/navigation/request', 10)
        self._cancel_goto_pub = self.create_publisher(CancelNavigationRequest, "/navigation/cancel", 10)
        self.pub_robot_task = self.create_publisher(String, robot_task_topic, 10)

        # Callback groups
        self.single_group = MutuallyExclusiveCallbackGroup()
        self.seq_group = MutuallyExclusiveCallbackGroup()
        self.multi_group = ReentrantCallbackGroup()

        # Subscriptions
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
        self.sub_chat_task_status = self.create_subscription(
            String, '/chat/task_status', self.on_chat_task_status, 10
        )
        
        # ========== GOAL STATUS SUBSCRIPTION ==========
        self.goal_status_sub_ = self.create_subscription(
            RobotGoalStatus, "/controller/goal_status", self.goal_status_callback, 10
        )
        # ==============================================

        self.get_logger().info(f'TeamLLMNode started. Ready to handle dynamic teams.')

        # History files
        package_path = get_package_share_directory(PACKAGE_NAME)
        self.history_base_dir = os.path.join(package_path, "data")
        os.makedirs(self.history_base_dir, exist_ok=True)

    @classmethod
    def get_instance(cls):
        """Get singleton instance"""
        if cls._instance is None:
            raise RuntimeError("TeamLLMNode has not been created yet!")
        return cls._instance

    # ========== GOAL STATUS CALLBACK ==========
    
    def goal_status_callback(self, msg: RobotGoalStatus) -> None:
        """
        Handle goal status updates from navigation controller.
        Updates the status for the specific robot in ALL teams it belongs to.
        """
        robot_name = msg.robot_name
        goal_reached = msg.goal_reached
        
        self.get_logger().info(
            f"Goal status update: {robot_name} -> {'SUCCESS' if goal_reached else 'FAILED'}"
        )
        
        # Update status for this robot in all teams it belongs to
        with self.team_lock:
            for team_name, team in self.teams.items():
                if robot_name in team.robots:
                    # Update the goal status for this robot
                    team.robot_goal_status[robot_name] = goal_reached
                    
                    self.get_logger().info(
                        f"[{team_name}] Updated {robot_name} goal status: "
                        f"{'REACHED' if goal_reached else 'FAILED'}"
                    )
                    
                    # Update robot state
                    self.update_team_robot_states(team_name, robot_name, {
                        "last_goal_status": "reached" if goal_reached else "failed",
                        "last_goal_timestamp": time.time()
                    })
    
    # ==========================================

    # ========== TEAM MANAGEMENT ==========

    def get_or_create_team(self, team_name: str, robots: list[str]) -> TeamState:
        """Get existing team or create new one with proper initialization"""
        with self.team_lock:
            if team_name not in self.teams:
                # Create new team
                team = TeamState(
                    name=team_name,
                    robots=robots,
                    task_queue=deque(),
                    team_robot_states={robot: {} for robot in robots},
                    robot_goal_status={robot: None for robot in robots}  # Initialize goal status
                )
                
                self.teams[team_name] = team
                self.get_logger().info(f"Created new team: {team_name} with robots: {robots}")
                
                # Create team history file
                self.create_team_history_file(team_name)
            
            return self.teams[team_name]

    def cleanup_team(self, team_name: str):
        """Remove team and cleanup resources after completion"""
        with self.team_lock:
            if team_name in self.teams:
                team = self.teams[team_name]
                
                # Log final status
                self.get_logger().info(
                    f"[{team_name}] Cleanup: "
                    f"Completed {len(team.task_history)} tasks, "
                    f"Status: {team.status}"
                )
                
                # Remove from active teams
                del self.teams[team_name]
                
                self.get_logger().info(f"[{team_name}] Team cleaned up successfully")

    def create_team_history_file(self, team_name: str):
        """Create history file for team"""
        history_file = os.path.join(self.history_base_dir, f"{team_name}_history.txt")
        with open(history_file, "w") as f:
            f.write(f"=== Team {team_name} Task History ===\n")
        return history_file

    def log_team_history(self, team_name: str, message: str):
        """Log to team-specific history file"""
        history_file = os.path.join(self.history_base_dir, f"{team_name}_history.txt")
        with open(history_file, "a") as f:
            timestamp = self.current_time
            f.write(f"[{timestamp}] {message}\n")

    # ========== STATUS AND STATE UPDATES ==========

    def update_team_status(self, team_name: str, status: str, task: str = None):
        """Update team status and publish consolidated JSON with all teams' statuses"""
        with self.team_lock:
            if team_name not in self.teams:
                return
            
            team = self.teams[team_name]
            team.status = status
            team.last_activity = time.time()
            
            if task:
                team.current_task = task

            # Add to team's task history
            self.teams[team_name].task_history.append({
                "timestamp": time.time(),
                "task": task or "",
                "status": status
            })

        # Build consolidated status dictionary for ALL teams
        all_teams_status = {}
        with self.team_lock:
            for t_name, t_state in self.teams.items():
                all_teams_status[t_name] = {
                    "status": t_state.status,
                    "current_task": t_state.current_task,
                    "robots": t_state.robots,
                    "pending_tasks": list(t_state.task_queue),
                    "completed_tasks": len(t_state.task_history),
                    "last_activity": t_state.last_activity
                }
        
        # Publish consolidated status as JSON
        msg = String()
        msg.data = json.dumps(all_teams_status)
        self.pub_task_status.publish(msg)

        # Log to team-specific history file
        individual_status = f"{team_name.upper()}: {task} - {status}" if task else f"{team_name.upper()}: {status}"
        self.log_team_history(team_name, individual_status)

        self.get_logger().info(f"[{team_name}] Status updated: {status}")
        self.get_logger().debug(f"Published all teams status: {json.dumps(all_teams_status, indent=2)}")

    def update_team_robot_states(self, team_name: str, robot_name: str, state_updates: dict):
        """Update state for a specific robot in a team"""
        with self.team_lock:
            if team_name not in self.teams:
                return
            
            team = self.teams[team_name]
            
            if robot_name not in team.team_robot_states:
                team.team_robot_states[robot_name] = {}
            
            # Update robot state
            team.team_robot_states[robot_name].update(state_updates)
            team.last_activity = time.time()

        # Also update global robot states
        self.update_global_robot_state(robot_name, state_updates)

        self.get_logger().debug(
            f"[{team_name}] Updated {robot_name} state: {state_updates}"
        )

    def update_global_robot_state(self, robot_name: str, state_updates: dict):
        """Update global robot states and publish"""
        if self.robot_states is None:
            self.robot_states = {}

        if "robot_states" not in self.robot_states:
            self.robot_states["robot_states"] = {}

        robots_dict = self.robot_states["robot_states"]

        if robot_name not in robots_dict:
            robots_dict[robot_name] = {}

        robots_dict[robot_name].update(state_updates)

        # Publish updated states
        msg = String()
        msg.data = json.dumps(self.robot_states)
        self.pub_robot_states.publish(msg)

    def get_team_state_summary(self, team_name: str) -> dict:
        """Get complete state summary for a team"""
        with self.team_lock:
            if team_name not in self.teams:
                return {}
            
            team = self.teams[team_name]
            return {
                "name": team.name,
                "robots": team.robots,
                "status": team.status,
                "current_task": team.current_task,
                "pending_tasks": list(team.task_queue),
                "completed_tasks": len(team.task_history),
                "robot_states": team.team_robot_states,
                "last_activity": team.last_activity
            }

    # ========== TASK PARSING ==========

    def parse_task_input(self, task_str: str) -> tuple[list[str], str]:
        """
        Parse task string to extract robot names and actual task.
        
        Examples:
          "[Burger1, Burger2]: goto fridge" -> (["burger1", "burger2"], "goto fridge")
          "goto fridge" -> ([], "goto fridge")
        
        Returns: (robots_list, task_text)
        """
        import re
        
        # Match pattern like "[Robot1, Robot2]: task"
        match = re.match(r'\[(.*?)\]\s*:\s*(.*)', task_str)
        if match:
            robots_str = match.group(1)
            task = match.group(2).strip()
            
            # Parse robot names
            robots = [r.strip().lower() for r in robots_str.split(',') if r.strip()]
            return robots, task
        
        # No bracket notation found
        return [], task_str.strip()

    def normalize_team_name(self, key: str) -> tuple[str, list[str]]:
        """
        Parse team name and robot list from task key.
        Returns: (team_name, robots_list)
        
        Examples:
          - "team5_task" -> ("team5", [])
          - "team_burger1_burger2" -> ("team_burger1_burger2", ["burger1", "burger2"])
          - "Team 5 tasks" -> ("team5", [])
        """
        key = key.lower().replace(" ", "_")
        
        # Match "team5" or "team_5" or "team_5"
        m = re.search(r"team[_]?(\d+)", key)
        if m:
            return f"team{m.group(1)}", []
        
        # Match "team_robot1_robot2_robot3"
        if key.startswith("team_"):
            parts = key.split("_")[1:]  # Remove "team" prefix
            robots = [r for r in parts if r not in ("task", "tasks")]
            if robots:
                team_name = f"team_{'_'.join(robots)}"
                return team_name, robots
        
        return None, []

    # ========== CALLBACKS ==========

    def on_robot_states(self, msg: String) -> None:
        """Handle global robot state updates"""
        try:
            data = json.loads(msg.data)
            rs = data.get("robot_states")
            if isinstance(rs, dict):
                self.robot_states = data  # Store entire structure
        except json.JSONDecodeError:
            self.get_logger().error(f"/robot_states not JSON: {msg.data}")

    def on_current_time(self, msg: String) -> None:
        """Update current time"""
        self.current_time = msg.data

    def on_chat_output(self, msg: String) -> None:
        """Handle chat output - could log to relevant team histories"""
        pass

    def on_chat_task_status(self, msg: String) -> None:
        """Observe external task status changes"""
        pass

    def on_tasks_json(self, msg: String) -> None:
        """
        Handle incoming team tasks.
        Creates/updates teams and starts worker threads.
        """
        try:
            data = json.loads(msg.data)
            robot_tasks = data.get("robot_tasks", {})
            
            # Parse all team tasks
            team_assignments = {}  # {team_name: {"robots": [], "tasks": []}}
            
            for key, task_str in robot_tasks.items():
                team_name, robots = self.normalize_team_name(key)
                
                if not team_name:
                    continue
                
                if team_name not in team_assignments:
                    team_assignments[team_name] = {
                        "robots": robots,
                        "tasks": []
                    }
                
                # Don't split on commas - treat entire task_str as one task
                if task_str.strip():
                    team_assignments[team_name]["tasks"].append(task_str.strip())
            
            if not team_assignments:
                self.get_logger().debug("No team tasks found")
                return
            
            # Process each team
            for team_name, assignment in team_assignments.items():
                self.handle_team_assignment(
                    team_name,
                    assignment["robots"],
                    assignment["tasks"]
                )
        
        except json.JSONDecodeError:
            self.get_logger().error(f"Invalid JSON: {msg.data}")

    def handle_team_assignment(self, team_name: str, robots: list[str], tasks: list[str]):
        """Handle task assignment for a specific team"""
        
        # Check for STOP command
        stop_tasks = [t for t in tasks if "stop" in t.lower()]
        if stop_tasks:
            self.cancel_team(team_name)
            self.update_team_status(team_name, "STOP COMPLETED")
            return
        
        # Parse robots from task strings if not already provided
        parsed_tasks = []
        task_robots = robots.copy() if robots else []
        
        for task in tasks:
            task_robot_list, clean_task = self.parse_task_input(task)
            
            # If robots were specified in the task, add them to the team
            if task_robot_list:
                for robot in task_robot_list:
                    if robot not in task_robots:
                        task_robots.append(robot)
            
            parsed_tasks.append(clean_task)
        
        # Get or create team with combined robot list
        team = self.get_or_create_team(team_name, task_robots)
        
        with self.team_lock:
            # Add tasks to queue
            for task in parsed_tasks:
                team.task_queue.append(task)
                self.get_logger().info(f"[{team_name}] Queued task: {task}")
            
            # Start worker thread if not already running
            if team.worker_thread is None or not team.worker_thread.is_alive():
                team.cancel_flag = False
                team.worker_thread = Thread(
                    target=self.run_team_worker,
                    args=(team_name,),
                    daemon=True
                )
                team.worker_thread.start()
                self.get_logger().info(f"[{team_name}] Worker thread started")
            else:
                self.get_logger().info(f"[{team_name}] Worker already running, tasks queued")

    # ========== TEAM WORKER THREAD ==========

    def run_team_worker(self, team_name: str):
        """Worker thread that processes tasks for a team"""
        try:
            self.get_logger().info(f"[{team_name}] Worker thread running")
            
            while True:
                # Get next task
                with self.team_lock:
                    if team_name not in self.teams:
                        self.get_logger().warn(f"[{team_name}] Team no longer exists")
                        return
                    
                    team = self.teams[team_name]
                    
                    if team.cancel_flag:
                        self.update_team_status(team_name, "CANCELLED")
                        return
                    
                    if not team.task_queue:
                        # All tasks completed
                        self.update_team_status(team_name, "ALL TASKS COMPLETED")
                        self.cleanup_team(team_name)
                        return
                    
                    task = team.task_queue.popleft()
                
                # Execute task (outside lock)
                self.update_team_status(team_name, "IN PROGRESS", task)
                
                try:
                    self.execute_team_task(team_name, task)
                    self.update_team_status(team_name, "COMPLETED", task)
                
                except TaskCancelledException:
                    self.update_team_status(team_name, "CANCELLED", task)
                    return
                
                except Exception as e:
                    self.get_logger().error(f"[{team_name}] Task failed: {e}")
                    self.update_team_status(team_name, "FAILED", task)
                    # Continue to next task instead of stopping
        
        except Exception as e:
            self.get_logger().error(f"[{team_name}] Worker crashed: {e}")
            self.update_team_status(team_name, "FAILED")
        
        finally:
            with self.team_lock:
                if team_name in self.teams:
                    self.teams[team_name].worker_thread = None

    def cancel_team(self, team_name: str):
        """Cancel all tasks for a team"""
        with self.team_lock:
            if team_name in self.teams:
                team = self.teams[team_name]
                team.cancel_flag = True
                team.task_queue.clear()
                self.get_logger().info(f"[{team_name}] Cancellation requested")

    # ========== TASK EXECUTION ==========

    def execute_team_task(self, team_name: str, task: str):
        """Execute a single task for a team using LLM"""
        team = self.teams.get(team_name)
        if not team:
            return
        
        self.get_logger().info(f"[{team_name}] Executing: {task}")
        
        # Build context
        chat_history = self.read_team_history(team_name)
        robot_context = self.build_robot_context_from_config()
        team_state = self.get_team_state_summary(team_name)
        
        # Build prompt
        self.get_logger().info("Building system action messages")
        available_actions = "\n".join(
            [f"Function Name: {opt.name}\nFunction Description: {opt.description}\nExample: {opt.example_code}"
             for opt in option_list]
        )
        
        self.get_logger().debug(f"Action message: {available_actions}")
        self.get_logger().info("Building system messages")
        
        prompt = (
            f"You are a robot control system controlling a team of robots named '{team_name}' with members: {team.robots}.\n"
            f"You can generate Python code to perform actions.\n"
            f"Based on the given task, you need to choose the appropriate action from the available options.\n\n"
            f"Robot System Context:\n{robot_context}\n\n"
            f"Team State:\n{json.dumps(team_state, indent=2)}\n\n"
            f"Recent Tasks (History):\n{chat_history}\n\n"
            f"Available Actions:\n{available_actions}\n\n"
            f"IMPORTANT: All function names are LOWERCASE. Use 'assemble', 'assemble_at', 'goto' - NOT 'Assemble', 'AssembleAt', 'Goto'.\n"
            f"Using the class reference name same as the example is important.\n"
            f"Use the name 'node' to refer to the TeamLLMNode instance.\n"
            f"The variables 'team_name' and 'robots' are automatically available in the execution context.\n"
            f"Your generated code is case-sensitive, so you MUST use the exact function names from the examples.\n"
        )
        
        self.get_logger().debug(f"Prompt message: {prompt}")
        
        # Generate code
        code, explanation = self.generate_action_prompt(prompt, task)
        
        if code:
            self.get_logger().info(f"[{team_name}] Generated Code:\n{code}")
            if explanation:
                self.get_logger().info(f"[{team_name}] Explanation:\n{explanation}")
            
            self.get_logger().info(f"[{team_name}] Calling execute_code...")
            # Execute
            self.execute_code(code, team_name)
        else:
            self.get_logger().error(f"[{team_name}] Failed to generate code")
            raise Exception("Code generation failed")

    def generate_action_prompt(self, prompt: str, task: str) -> tuple[str, str]:
        """Generate action code using LLM"""
        self.get_logger().info("Generating code...")
        
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
            self.get_logger().info(f"Generated Content:\n{raw}")
            
            if "```python" in raw:
                parts = raw.split("```python")
                explanation = parts[0].strip()
                code = parts[1].split("```")[0].strip()
                self.get_logger().debug(f"Code:\n{code}")
                self.get_logger().debug(f"Explanation: {explanation}")
                return code, explanation
            
            self.get_logger().warn("Response doesn't contain ```python code block")
            return "", ""
        
        except openai.Timeout:
            self.get_logger().error("OpenAI request TIMED OUT (no response in time)")
        except openai.AuthenticationError:
            self.get_logger().error("OpenAI authentication failed — check your API key!")
        except openai.RateLimitError:
            self.get_logger().error("OpenAI rate limit hit — slow down or upgrade plan")
        except openai.APIError as e:
            self.get_logger().error(f"OpenAI API error: {e}")
        except Exception as e:
            self.get_logger().error(f"Unexpected error in LLM call: {type(e).__name__}: {e}")
        
        self.get_logger().warn("Returning (None, None) due to LLM failure")
        return None, None

    def execute_code(self, code: str, team_name: str):
        """Execute generated code with team context"""
        team = self.teams.get(team_name)
        if not team:
            return
        
        self.get_logger().info(f"[{team_name}] Executing generated Python code")
        
        try:
            context = {
                "node": self,
                "team_name": team_name,
                "robots": team.robots,
                "TaskCancelledException": TaskCancelledException,
                "time": time
            }
            
            exec(code, {"__builtins__": {}}, context)
            
            self.get_logger().info(f"[{team_name}] Code executed successfully")
            
        except TaskCancelledException:
            self.get_logger().warn(f"[{team_name}] Code execution cancelled")
            raise
        except TypeError as e:
            self.get_logger().error(f"[{team_name}] TypeError in execution: {e}")
            raise
        except Exception as e:
            self.get_logger().error(f"[{team_name}] Execution error: {e}")
            raise

    # ========== WAIT FOR TEAM NAVIGATION COMPLETION ==========

    def wait_for_team_goto_complete(self, team_name: str, goal: str, timeout: float = 60000.0) -> bool:
        """
        Wait for ALL robots in the team to complete navigation.
        
        Args:
            team_name: Name of the team
            goal: The goal location being navigated to
            timeout: Maximum time to wait in seconds (default: 120s)
        
        Returns:
            bool: True if ALL robots reached goal, False if any failed or timed out
        """
        team = self.teams.get(team_name)
        if not team:
            self.get_logger().error(f"[{team_name}] Team not found")
            return False
        
        robot_list = team.robots
        
        self.get_logger().info(
            f"[{team_name}] Waiting for navigation to {goal} for robots: {robot_list} "
            f"(timeout={timeout}s)..."
        )
        
        # Reset goal status flags for all robots in the team
        with self.team_lock:
            for robot in robot_list:
                team.robot_goal_status[robot] = None
        
        start_time = time.time()
        deadline = start_time + timeout
        
        while rclpy.ok():
            # ========== CHECK CANCELLATION ==========
            with self.team_lock:
                if team.cancel_flag:
                    self.get_logger().warn(f"[{team_name}] Navigation cancelled during wait")
                    return False
            # ========================================
            
            # Check status of all robots
            with self.team_lock:
                team = self.teams.get(team_name)
                if not team:
                    self.get_logger().error(f"[{team_name}] Team disappeared during wait")
                    return False
                
                # Count robots with status
                robots_with_status = {
                    robot: status 
                    for robot, status in team.robot_goal_status.items() 
                    if status is not None  # Only count robots that have received status
                }
                
                # Check if any robot has failed
                failed_robots = [
                    robot for robot, status in robots_with_status.items() 
                    if status is False
                ]
                
                if failed_robots:
                    elapsed = time.time() - start_time
                    self.get_logger().error(
                        f"[{team_name}] Navigation to {goal} FAILED. "
                        f"Failed robots: {failed_robots} (elapsed: {elapsed:.2f}s)"
                    )
                    return False
                
                # Check if ALL robots have succeeded
                # IMPORTANT: Only check robots that have status (not None)
                # Missing status means we're still waiting, not failed
                succeeded_robots = [
                    robot for robot, status in robots_with_status.items() 
                    if status is True
                ]
                
                # All robots must have succeeded (and we must have status for all)
                if len(succeeded_robots) == len(robot_list):
                    elapsed = time.time() - start_time
                    self.get_logger().info(
                        f"[{team_name}] Navigation to {goal} completed successfully for ALL robots "
                        f"in {elapsed:.2f}s"
                    )
                    return True
                
                # Log progress
                self.get_logger().debug(
                    f"[{team_name}] Navigation progress: "
                    f"{len(succeeded_robots)}/{len(robot_list)} robots reached goal. "
                    f"Waiting for: {[r for r in robot_list if r not in succeeded_robots]}"
                )
            
            # Check timeout
            if time.time() > deadline:
                elapsed = time.time() - start_time
                
                with self.team_lock:
                    pending = [
                        robot for robot in robot_list 
                        if team.robot_goal_status.get(robot) is None
                    ]
                
                self.get_logger().error(
                    f"[{team_name}] Navigation to {goal} TIMED OUT after {elapsed:.2f}s. "
                    f"Pending robots: {pending}"
                )
                return False
            
            # Sleep briefly to avoid busy waiting
            time.sleep(0.1)
        
        self.get_logger().error(f"[{team_name}] ROS shutdown during navigation wait")
        return False

    # ========== TEAM ACTIONS (Called by generated code) ==========

    def assemble(self, team_name: str = None, robots: list[str] = None) -> bool:
        """Team assembles at computed centroid"""
        if not team_name:
            raise ValueError("team_name is required")
        
        team = self.teams.get(team_name)
        if not team:
            return False
        
        # Use provided robots list or default to team's robots
        robot_list = robots if robots is not None else team.robots
        
        self.get_logger().info(f"[{team_name}] Assembling robots: {robot_list}")
        
        # Update states for all robots
        for robot in robot_list:
            self.update_team_robot_states(team_name, robot, {
                "action": "assembling",
                "formation": "active"
            })
        
        # Publish navigation request for assembly (goal is empty for centroid assembly)
        msg = NavigationRobotRequest()
        msg.goal = "assemble"  
        msg.formation_robots = robot_list
        self._request_goto_pub.publish(msg)
        
        self.get_logger().info(f"[{team_name}] Published assembly request for robots: {robot_list}")
        
        # ========== WAIT FOR ALL ROBOTS TO COMPLETE ==========
        success = self.wait_for_team_goto_complete(team_name, "assemble")
        # ====================================================
        
        if success:
            # Update completion
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "assembled",
                    "formation": "ready"
                })
            return True
        else:
            self.get_logger().error(f"[{team_name}] Assembly failed")
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "assembly_failed",
                    "formation": "broken"
                })
            return False

    def assemble_at(self, goal: str, team_name: str = None, robots: list[str] = None) -> bool:
        """Team assembles at specific goal location"""
        if not team_name:
            raise ValueError("team_name is required")
        
        team = self.teams.get(team_name)
        if not team:
            return False
        
        # Use provided robots list or default to team's robots
        robot_list = robots if robots is not None else team.robots
        
        self.get_logger().info(f"[{team_name}] Assembling at {goal}: {robot_list}")
        
        # Update states
        for robot in robot_list:
            self.update_team_robot_states(team_name, robot, {
                "action": "assembling_at",
                "target_goal": goal,
                "formation": "active"
            })
        
        # Publish navigation request for assembly at specific location
        msg = NavigationRobotRequest()
        msg.goal = goal
        msg.formation_robots = robot_list
        self._request_goto_pub.publish(msg)
        
        self.get_logger().info(f"[{team_name}] Published assembly request to {goal} for robots: {robot_list}")
        
        # ========== WAIT FOR ALL ROBOTS TO COMPLETE ==========
        success = self.wait_for_team_goto_complete(team_name, goal)
        # ====================================================
        
        if success:
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "assembled",
                    "current_location": goal,
                    "formation": "ready"
                })
            return True
        else:
            self.get_logger().error(f"[{team_name}] Assembly at {goal} failed")
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "assembly_failed",
                    "formation": "broken"
                })
            return False

    def goto(self, goal: str, team_name: str = None, robots: list[str] = None) -> bool:
        """Team navigates to goal in formation"""
        if not team_name:
            raise ValueError("team_name is required")
        
        team = self.teams.get(team_name)
        if not team:
            return False
        
        # Use provided robots list or default to team's robots
        robot_list = robots if robots is not None else team.robots
        
        self.get_logger().info(f"[{team_name}] Going to {goal}: {robot_list}")
        
        # Update states
        for robot in robot_list:
            self.update_team_robot_states(team_name, robot, {
                "action": "navigating",
                "target_goal": goal,
                "formation": "maintaining"
            })
        
        # Publish navigation request
        msg = NavigationRobotRequest()
        msg.goal = goal
        msg.formation_robots = robot_list
        self._request_goto_pub.publish(msg)
        
        self.get_logger().info(f"[{team_name}] Published navigation request to {goal} for robots: {robot_list}")
        
        # ========== WAIT FOR ALL ROBOTS TO COMPLETE ==========
        success = self.wait_for_team_goto_complete(team_name, goal)
        # ====================================================
        
        if success:
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "arrived",
                    "current_location": goal,
                    "formation": "ready"
                })
            return True
        else:
            self.get_logger().error(f"[{team_name}] Navigation to {goal} failed")
            for robot in robot_list:
                self.update_team_robot_states(team_name, robot, {
                    "action": "navigation_failed",
                    "formation": "broken"
                })
            return False
    
    # ========== HELPER METHODS ==========

    def read_team_history(self, team_name: str) -> str:
        """Read team's task history"""
        history_file = os.path.join(self.history_base_dir, f"{team_name}_history.txt")
        
        if not os.path.exists(history_file):
            return "No previous history."
        
        try:
            with open(history_file, "r") as f:
                return f.read().strip()
        except Exception as e:
            self.get_logger().error(f"Error reading history: {e}")
            return "Error reading history."

    def build_robot_context_from_config(self) -> str:
        """Build context from robot config"""
        cfg = self.robot_config
        lines = []
        
        robot_names = cfg.get("robot_names", [])
        if robot_names:
            lines.append(f"Available robots: {', '.join(robot_names)}")
        
        robot_caps = cfg.get("robot_capabilities", {})
        if robot_caps:
            lines.append("Capabilities:")
            for name, desc in robot_caps.items():
                lines.append(f"  - {name}: {desc}")
        
        return "\n".join(lines)


def main(args=None):
    rclpy.init(args=args)
    node = TeamLLMNode()
    
    executor = rclpy.executors.MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    
    try:
        executor.spin()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()