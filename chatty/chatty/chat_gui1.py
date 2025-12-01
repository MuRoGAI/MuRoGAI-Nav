#!/usr/bin/env python3
import json
import os
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from ament_index_python.packages import get_package_share_directory

import customtkinter as ctk
from threading import Thread
from datetime import datetime

class ChatGUI(Node):
    def __init__(self):
        super().__init__("human_gui")
        
        self.publisher = self.create_publisher(String, "/chat/input", 10)
        self.switch_publisher = self.create_publisher(String, "/switch_state", 10)
        self.subscription = self.create_subscription(String, "/chat/output", self.on_output, 10)
        self.time_sub = self.create_subscription(String, "/current_time", self.on_time, 10)

        self.declare_parameter("config_file", "robot_config_assmble_help")
        cfg_file_name = self.get_parameter("config_file").get_parameter_value().string_value
        package_share = get_package_share_directory("chatty")

        cfg_file_name = cfg_file_name + ".json"
        self.config_file_path = os.path.join(package_share, "config", cfg_file_name)

        self.read_json_config()
        self.robot_names = []
        self.robot_colors = []
        self.mice_active = False
        self.current_time = ""

        # Set theme
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        # Clean White/Black Theme with Colored Robot Messages
        self.colors = {
            "bg_white": "#FFFFFF",
            "bg_light_gray": "#F5F5F5",
            "border_gray": "#E0E0E0",
            "text_black": "#1A1A1A",
            "text_gray": "#666666",
            "human_msg": "#2196F3",      # Blue
            "task_msg": "#FF9800",       # Orange
            "go2_msg": '#607D8B',        # Blue Gray
            "burger_msg": "#009688",     # Teal
            "waffle_msg": "#9C27B0",     # Purple
            "drone_msg": "#795548",      # Brown
            "formation_msg": "#3F51B5",  # Indigo
            "x_arm_msg": "#4CAF50",      # Green
            "lerobot1_msg": "#E91E63",   # Pink
            "lerobot2_msg": "#8BC34A",   # Light Green
            "clock_msg": "#00BCD4",      # Cyan
            "lerobot_msg": "#FF5722",    # Deep Orange
        }

        self.window = ctk.CTk()
        self.window.title("CoMuRoS - Multi-Robot Communication System")
        self.window.geometry("900x700")
        self.window.configure(fg_color=self.colors["bg_light_gray"])

        # Main container
        self.main_container = ctk.CTkFrame(
            self.window, 
            fg_color=self.colors["bg_white"],
            corner_radius=15,
            border_width=1,
            border_color=self.colors["border_gray"]
        )
        self.main_container.pack(padx=20, pady=20, fill='both', expand=True)

        # Header
        self.header_frame = ctk.CTkFrame(
            self.main_container, 
            fg_color=self.colors["bg_white"],
            corner_radius=0,
            height=70
        )
        self.header_frame.pack(fill='x', padx=0, pady=0)
        self.header_frame.pack_propagate(False)
        
        # Header content
        header_content = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        header_content.pack(fill='both', expand=True, padx=20, pady=15)
        
        self.header_label = ctk.CTkLabel(
            header_content, 
            text="CoMuRoS Chat Interface", 
            font=("Arial", 24, "bold"),
            text_color=self.colors["text_black"]
        )
        self.header_label.pack(side='left')

        # Status indicator
        self.status_indicator = ctk.CTkLabel(
            header_content,
            text="Online",
            font=("Arial", 12),
            text_color="#4CAF50"
        )
        self.status_indicator.pack(side='right')

        # Divider line
        divider = ctk.CTkFrame(self.main_container, fg_color=self.colors["border_gray"], height=1)
        divider.pack(fill='x', padx=0, pady=0)

        # Chat area
        self.chat_container = ctk.CTkFrame(
            self.main_container,
            fg_color=self.colors["bg_light_gray"],
            corner_radius=0
        )
        self.chat_container.pack(padx=0, pady=0, fill='both', expand=True)

        self.chat_area = ctk.CTkScrollableFrame(
            self.chat_container,
            fg_color="transparent",
            scrollbar_button_color=self.colors["border_gray"],
            scrollbar_button_hover_color=self.colors["text_gray"]
        )
        self.chat_area.pack(padx=10, pady=10, fill='both', expand=True)

        # Divider line
        divider2 = ctk.CTkFrame(self.main_container, fg_color=self.colors["border_gray"], height=1)
        divider2.pack(fill='x', padx=0, pady=0)

        # Input area
        self.input_container = ctk.CTkFrame(
            self.main_container,
            fg_color=self.colors["bg_white"],
            corner_radius=0,
            height=80
        )
        self.input_container.pack(fill='x', padx=0, pady=0)
        self.input_container.pack_propagate(False)

        # Input content frame
        input_content = ctk.CTkFrame(self.input_container, fg_color="transparent")
        input_content.pack(fill='both', expand=True, padx=15, pady=15)

        # Entry field
        self.entry = ctk.CTkEntry(
            input_content,
            font=("Arial", 13),
            placeholder_text="Type your message here...",
            height=45,
            corner_radius=22,
            fg_color=self.colors["bg_light_gray"],
            text_color=self.colors["text_black"],
            placeholder_text_color=self.colors["text_gray"],
            border_width=1,
            border_color=self.colors["border_gray"]
        )
        self.entry.pack(side='left', expand=True, fill='x', padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        # Buttons container
        button_container = ctk.CTkFrame(input_content, fg_color="transparent")
        button_container.pack(side='right')

        # Mice button
        self.mice_button = ctk.CTkButton(
            button_container,
            text="VOICE",
            command=self.toggle_mice,
            font=("Arial", 12, "bold"),
            fg_color="#2196F3",
            hover_color="#1976D2",
            text_color="#FFFFFF",
            corner_radius=22,
            width=90,
            height=45,
            border_width=0
        )
        self.mice_button.pack(side='left', padx=5)

        # Send button
        self.send_button = ctk.CTkButton(
            button_container,
            text="SEND",
            command=self.send_message,
            font=("Arial", 12, "bold"),
            fg_color="#4CAF50",
            hover_color="#388E3C",
            text_color="#FFFFFF",
            corner_radius=22,
            width=90,
            height=45,
            border_width=0
        )
        self.send_button.pack(side='left', padx=5)

        self.get_logger().info("[ChatGUI] Enhanced GUI node initialized.")

        self.history_fetched = False
        self.window.after(500, self.fetch_history_once)

    def animate_button_press(self, button, callback=None):
        """Smooth button press animation"""
        original_width = button.cget("width")
        original_height = button.cget("height")
        
        # Shrink animation
        button.configure(width=original_width - 8, height=original_height - 4)
        
        def restore():
            button.configure(width=original_width, height=original_height)
            if callback:
                callback()
        
        self.window.after(100, restore)

    def toggle_mice(self):
        """Toggle mice button state with animation"""
        def publish_state():
            msg = String()
            if not self.mice_active:
                msg.data = "MICE BUTTON PRESSED"
                self.mice_button.configure(
                    text="VOICE ON",
                    fg_color="#F44336"
                )
                self.mice_active = True
                self.get_logger().info("[ChatGUI] Mice button activated")
            else:
                msg.data = "MICE BUTTON TURNED OFF"
                self.mice_button.configure(
                    text="VOICE",
                    fg_color="#2196F3"
                )
                self.mice_active = False
                self.get_logger().info("[ChatGUI] Mice button deactivated")
            
            self.switch_publisher.publish(msg)
        
        self.animate_button_press(self.mice_button, publish_state)

    def read_json_config(self):
        try:
            if not os.path.exists(self.config_file_path):
                self.get_logger().error(f"[ChatGUI] Config file not found: {self.config_file_path}")
                return

            with open(self.config_file_path, "r") as f:
                config = json.load(f)

            self.get_logger().info(f"[ChatGUI] Loaded config from {self.config_file_path}")

            robot_names = config.get("robot_names", [])
            colors_assigned = config.get("colors_assigned", {})
            robot_colors = [colors_assigned.get(name, "#FFFFFF") for name in robot_names]

            self.robot_names = robot_names
            self.robot_colors = robot_colors

            self.get_logger().info(f"[ChatGUI] Robots: {self.robot_names}")
            self.get_logger().info(f"[ChatGUI] Colors: {self.robot_colors}")

        except Exception as e:
            self.get_logger().error(f"[ChatGUI] Failed to load config: {e}")

    def on_time(self, msg):
        self.current_time = msg.data

    def on_output(self, msg):
        line = msg.data
        self.get_logger().info(f"[ChatGUI] /chat/output -> {line}")
        
        if "]" in line:
            line = line.split("] ", 1)[-1]

        self.window.after(0, lambda: self.append_text(line))

    def append_text(self, text_line):
        timestamp = datetime.now().strftime("%H:%M")

        # Message parsing logic
        if "Human:" in text_line:
            message_color = self.colors["human_msg"]
            label_text = "You"
            text_line = text_line.replace("Human:", "").strip()
            align = "e"
            icon = "H"
        elif "Burger (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
            message_color = self.colors["burger_msg"]
            label_text = "Burger"
            text_line = text_line.replace("Burger", "").strip()
            align = "w"
            icon = "B"
        elif "Waffle (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
            message_color = self.colors["waffle_msg"]
            label_text = "Waffle"
            text_line = text_line.replace("Waffle", "").strip()
            align = "w"
            icon = "W"
        elif "Go2 (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace(":", "").replace("(msg)", "").strip()
            message_color = self.colors["go2_msg"]
            label_text = "Go2"
            text_line = text_line.replace("Go2", "").strip()
            align = "w"
            icon = "G"
        elif "Drone (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace(":", "").replace("(msg)", "").strip()
            message_color = self.colors["drone_msg"]
            label_text = "Drone"
            text_line = text_line.replace("Drone", "").strip()
            align = "w"
            icon = "D"
        elif "Formation (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace(":", "").replace("(msg)", "").strip()
            message_color = self.colors["formation_msg"]
            label_text = "Formation"
            text_line = text_line.replace("Formation", "").strip()
            align = "w"
            icon = "F"
        elif "X Arm (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace(":", "").replace("(msg)", "").strip()
            message_color = self.colors["x_arm_msg"]
            label_text = "X Arm"
            text_line = text_line.replace("X Arm", "").strip()
            align = "w"
            icon = "X"
        elif "Lerobot1 (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
            message_color = self.colors["lerobot1_msg"]
            label_text = "Lerobot1"
            text_line = text_line.replace("Lerobot1", "").strip()
            align = "w"
            icon = "L1"
        elif "Lerobot2 (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
            message_color = self.colors["lerobot2_msg"]
            label_text = "Lerobot2"
            text_line = text_line.replace("Lerobot2", "").strip()
            align = "w"
            icon = "L2"
        elif "Clock (msg)" in text_line:
            text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
            message_color = self.colors["clock_msg"]
            label_text = "Clock"
            text_line = text_line.replace("Clock", "").strip()
            align = "w"
            icon = "C"
        elif "Task Manager" in text_line or "Unknown" in text_line:
            text_line = text_line.replace("Task Manager:", "").replace("Taskmanager:", "").replace("Unknown:", "").strip()
            message_color = self.colors["task_msg"]
            label_text = "Task Manager"
            align = "w"
            icon = "TM"
        else:
            # Check for dynamic robot names from config
            found = False
            for name in self.robot_names:
                if f"{name.capitalize()} (msg)" in text_line or f"{name.lower()} (msg)" in text_line:
                    self.get_logger().info(f"[ChatGUI] Match found for robot: {name}")
                    text_line = text_line.replace("Unknown:", "").replace("(msg)", "").replace(":", "").strip()
                    message_color = self.colors.get(f"{name.lower()}_msg", "#FF5722")
                    label_text = name.capitalize()
                    text_line = text_line.replace(name.capitalize(), "").replace(name.lower(), "").strip()
                    align = "w"
                    icon = "R"
                    found = True
                    break
            
            if not found:
                self.get_logger().warn(f"[ChatGUI] No match found, defaulting to System for: {text_line}")
                message_color = self.colors["text_gray"]
                label_text = "System"
                align = "w"
                icon = "SYS"

        # Create message bubble
        message_outer = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        message_outer.pack(fill="x", padx=5, pady=6, anchor=align)

        message_frame = ctk.CTkFrame(
            message_outer,
            fg_color=message_color,
            corner_radius=12,
            border_width=0
        )
        message_frame.pack(anchor=align)

        # Header with icon and name
        header_frame = ctk.CTkFrame(message_frame, fg_color="transparent")
        header_frame.pack(fill="x", padx=12, pady=(8, 4))

        # Icon badge
        icon_badge = ctk.CTkLabel(
            header_frame,
            text=icon,
            font=("Arial", 10, "bold"),
            text_color="#FFFFFF",
            fg_color="transparent",
            width=30,
            height=20
        )
        icon_badge.pack(side="left", padx=(0, 5))

        header_label = ctk.CTkLabel(
            header_frame,
            text=label_text,
            font=("Arial", 12, "bold"),
            text_color="#FFFFFF"
        )
        header_label.pack(side="left")

        time_label = ctk.CTkLabel(
            header_frame,
            text=timestamp,
            font=("Arial", 10),
            text_color="#FFFFFF"
        )
        time_label.pack(side="right", padx=(10, 0))

        # Message content
        message_label = ctk.CTkLabel(
            message_frame,
            text=text_line,
            font=("Arial", 12),
            text_color="#FFFFFF",
            wraplength=600,
            justify="left"
        )
        message_label.pack(padx=12, pady=(0, 8), anchor="w")

        self.chat_area.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1)

    def send_message(self, event=None):
        def publish_message():
            user_input = self.entry.get().strip()
            if user_input:
                out_msg = String()
                out_msg.data = f"human|{user_input}"
                self.publisher.publish(out_msg)
                self.get_logger().info(f"[ChatGUI] Sent -> {user_input}")
                self.entry.delete(0, 'end')
        
        self.animate_button_press(self.send_button, publish_message)

    def fetch_history_once(self):
        if self.history_fetched:
            return
        self.get_logger().info("[ChatGUI] Attempting to fetch old chat.")
        client = self.create_client(Trigger, "get_chat_history")
        if not client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn("[ChatGUI] Manager not ready, skipping old history fetch.")
            self.history_fetched = True
            return
        
        req = Trigger.Request()
        future = client.call_async(req)

        def check_done():
            if future.done():
                res = future.result()
                if res and res.success:
                    lines = res.message.split("\n")
                    self.get_logger().info(f"[ChatGUI] Received {len(lines)} lines of old chat.")
                    for line in lines:
                        if line.strip():
                            self.window.after(0, lambda l=line: self.append_text(l))
                else:
                    self.get_logger().error("[ChatGUI] Could not fetch old history.")
                self.history_fetched = True
            else:
                self.window.after(200, check_done)
        
        self.window.after(200, check_done)

    def run_gui(self):
        self.get_logger().info("[ChatGUI] Starting Tkinter mainloop.")
        self.window.mainloop()


def main():
    rclpy.init()
    node = ChatGUI()
    
    def spin_bg():
        rclpy.spin(node)
    
    t = Thread(target=spin_bg, daemon=True)
    t.start()

    try:
        node.run_gui()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()