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

        # ROS publishers / subscribers
        self.publisher = self.create_publisher(String, "/chat/input", 10)
        self.switch_publisher = self.create_publisher(String, "/switch_state", 10)
        self.subscription = self.create_subscription(String, "/chat/output", self.on_output, 10)
        self.time_sub = self.create_subscription(String, "/current_time", self.on_time, 10)

        # Config loading
        self.declare_parameter("config_file", "robot_config_assmble_help")
        cfg_file_name = self.get_parameter("config_file").get_parameter_value().string_value
        package_share = get_package_share_directory("chatty")
        self.config_file_path = os.path.join(package_share, "config", cfg_file_name + ".json")

        self.read_json_config()
        self.mice_active = False

        # Colors (will be overridden by config when present)
        self.colors = {
            "bg": "#F8FAFC",
            "surface": "#FFFFFF",
            "border": "#E2E8F0",
            "text_primary": "#1E293B",
            "human": "#3B82F6",
            "task": "#F59E0B",
            "go2_msg": "#4F46E5",
            "burger_msg": "#10B981",
            "waffle_msg": "#8B5CF6",
            "drone_msg": "#EF4444",
            "formation_msg": "#06B6D4",
            "x_arm_msg": "#10B981",
            "lerobot1_msg": "#EC4899",
            "lerobot2_msg": "#F97316",
            "clock_msg": "#14B8A6",
        }

        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")

        self.window = ctk.CTk()
        self.window.title("CoMuRoS  Multi-Robot Communication System")
        self.window.geometry("1000x750")
        self.window.minsize(900, 650)
        self.window.configure(fg_color=self.colors["bg"])

        self.build_ui()

        self.get_logger().info("[ChatGUI] Professional GUI ready")
        self.history_fetched = False
        self.window.after(800, self.fetch_history_once)

    def build_ui(self):
        # Main container
        main = ctk.CTkFrame(self.window, fg_color=self.colors["surface"], corner_radius=16)
        main.pack(fill="both", expand=True, padx=20, pady=20)

        # Header
        header = ctk.CTkFrame(main, fg_color="transparent", height=80)
        header.pack(fill="x", padx=20, pady=(20, 10))
        header.pack_propagate(False)

        ctk.CTkLabel(header, text="CoMuRoS", font=ctk.CTkFont(size=28, weight="bold"),
                     text_color=self.colors["text_primary"]).pack(side="left")

        status = ctk.CTkFrame(header, fg_color="#10B981", corner_radius=20)
        status.pack(side="right")
        ctk.CTkLabel(status, text="ONLINE", font=ctk.CTkFont(size=14, weight="bold"),
                     text_color="white").pack(padx=14, pady=8)

        # Chat area
        self.chat_area = ctk.CTkScrollableFrame(main, fg_color=self.colors["bg"], corner_radius=12)
        self.chat_area.pack(fill="both", expand=True, padx=20, pady=10)

        # Input area
        input_frame = ctk.CTkFrame(main, fg_color=self.colors["surface"], corner_radius=16, height=100)
        input_frame.pack(fill="x", padx=20, pady=(0, 20))
        input_frame.pack_propagate(False)

        inner = ctk.CTkFrame(input_frame, fg_color="transparent")
        inner.pack(fill="both", expand=True, padx=20, pady=20)

        self.entry = ctk.CTkEntry(inner, placeholder_text="Type your message…",
                                  font=ctk.CTkFont(size=16), height=50, corner_radius=25)
        self.entry.pack(side="left", fill="both", expand=True, padx=(0, 10))
        self.entry.bind("<Return>", self.send_message)

        btns = ctk.CTkFrame(inner, fg_color="transparent")
        btns.pack(side="right")

        self.mice_button = ctk.CTkButton(btns, text="VOICE", width=110, height=50, corner_radius=25,
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         fg_color="#3B82F6", hover_color="#2563EB",
                                         command=self.toggle_mice)
        self.mice_button.pack(side="left", padx=5)

        self.send_button = ctk.CTkButton(btns, text="SEND", width=110, height=50, corner_radius=25,
                                         font=ctk.CTkFont(size=14, weight="bold"),
                                         fg_color="#10B981", hover_color="#059669",
                                         command=self.send_message)
        self.send_button.pack(side="left", padx=5)

    def animate_button(self, btn):
        orig = btn.cget("fg_color")
        btn.configure(fg_color="#1E293B")
        self.window.after(100, lambda: btn.configure(fg_color=orig))

    def toggle_mice(self):
        if not self.mice_active:
            self.mice_button.configure(text="VOICE ON", fg_color="#EF4444")
            self.mice_active = True
            data = "MICE BUTTON PRESSED"
        else:
            self.mice_button.configure(text="VOICE", fg_color="#3B82F6")
            self.mice_active = False
            data = "MICE BUTTON TURNED OFF"
        msg = String()
        msg.data = data
        self.switch_publisher.publish(msg)
        self.animate_button(self.mice_button)

    def read_json_config(self):
        if not os.path.exists(self.config_file_path):
            self.get_logger().warn(f"Config not found: {self.config_file_path}")
            return
        try:
            with open(self.config_file_path) as f:
                cfg = json.load(f)
                self.robot_names = cfg.get("robot_names", [])
                colors = cfg.get("colors_assigned", {})
                for name in self.robot_names:
                    key = f"{name.lower()}_msg"
                    if name.lower() in colors:
                        self.colors[key] = colors[name.lower()]
            self.get_logger().info(f"Loaded {len(self.robot_names)} robots from config")
        except Exception as e:
            self.get_logger().error(f"Config error: {e}")

    def on_time(self, msg):
        pass  # not used right now

    def on_output(self, msg):
        line = msg.data.strip()
        if "]" in line:
            line = line.split("] ", 1)[-1]
        self.window.after(0, lambda l=line: self.append_message(l))

    def append_message(self, text):
        ts = datetime.now().strftime("%H:%M")

        # defaults
        bubble_color = "#64748B"
        sender = "System"
        icon = "SYS"
        align = "w"          # w = left (robots), e = right (human)

        if text.startswith("Human:"):
            bubble_color = self.colors["human"]
            text = text.replace("Human:", "").strip()
            sender = "You"
            icon = "YOU"
            align = "e"
        elif any(x in text for x in ["Task Manager", "Taskmanager", "Unknown"]):
            text = text.replace("Task Manager:", "").replace("Taskmanager:", "").replace("Unknown:", "").strip()
            bubble_color = self.colors["task"]
            sender = "Task Manager"
            icon = "TM"
        else:
            # dynamic robots from config
            matched = False
            for name in self.robot_names:
                if f"{name} (msg)" in text or f"{name.lower()} (msg)" in text:
                    text = text.replace("Unknown:", "").replace("(msg)", "").replace(name, "").strip()
                    sender = name.capitalize()
                    key = f"{name.lower()}_msg"
                    bubble_color = self.colors.get(key, "#8B5CF6")
                    icon = "".join([c[0].upper() for c in name.split()[:2]])
                    matched = True
                    break
            if not matched:
                # fallback hardcoded
                mapping = {
                    "Go2": ("G2", "#4F46E5"),
                    "Burger": ("BUR", "#10B981"),
                    "Waffle": ("WAF", "#8B5CF6"),
                    "Drone": ("DRN", "#EF4444"),
                    "Formation": ("FRM", "#06B6D4"),
                    "X Arm": ("ARM", "#10B981"),
                    "Lerobot1": ("L1", "#EC4899"),
                    "Lerobot2": ("L2", "#F97316"),
                    "Clock": ("CLK", "#14B8A6"),
                }
                for k, (i, c) in mapping.items():
                    if k in text:
                        text = text.replace(k, "").replace("(msg)", "").strip()
                        sender = k
                        icon = i
                        bubble_color = c
                        break

        # Message container
        container = ctk.CTkFrame(self.chat_area, fg_color="transparent")
        container.pack(fill="x", pady=8, padx=12, anchor=align)

        bubble = ctk.CTkFrame(container, fg_color=bubble_color, corner_radius=20)
        bubble.pack(anchor=align, padx=(70 if align == "w" else 0, 70 if align == "e" else 0))

        # Header (icon + name + time)
        header = ctk.CTkFrame(bubble, fg_color="transparent")
        header.pack(fill="x", padx=16, pady=(12, 4))

        ctk.CTkLabel(header, text=icon, font=ctk.CTkFont(size=13, weight="bold"),
                     width=38, height=38, corner_radius=19,
                     fg_color=bubble_color, text_color="white").pack(side="left")

        ctk.CTkLabel(header, text=sender, font=ctk.CTkFont(size=15, weight="bold"),
                     text_color="white").pack(side="left", padx=(10, 0))

        ctk.CTkLabel(header, text=ts, font=ctk.CTkFont(size=12),
                     text_color="#E2E8F0").pack(side="right")

        # Message text (THIS WAS THE BUG – pady must be separate, not a tuple)
        ctk.CTkLabel(bubble, text=text, font=ctk.CTkFont(size=16), text_color="white",
                     wraplength=620, justify="left", anchor="w",
                     padx=16, pady=12).pack(fill="x", pady=(0, 16))

        # Auto-scroll to bottom
        self.chat_area.update_idletasks()
        self.chat_area._parent_canvas.yview_moveto(1.0)

    def send_message(self, event=None):
        txt = self.entry.get().strip()
        if not txt:
            return
        msg = String()
        msg.data = f"human|{txt}"
        self.publisher.publish(msg)
        self.get_logger().info(f"Sent: {txt}")
        self.entry.delete(0, "end")
        self.animate_button(self.send_button)

    def fetch_history_once(self):
        if self.history_fetched:
            return
        client = self.create_client(Trigger, "get_chat_history")
        if not client.wait_for_service(timeout_sec=2.0):
            self.get_logger().warn("History service not available")
            self.history_fetched = True
            return

        future = client.call_async(Trigger.Request())

        def check():
            if future.done():
                try:
                    resp = future.result()
                    if resp.success:
                        for line in resp.message.strip().split("\n"):
                            if line.strip():
                                self.append_message(line)
                except Exception as e:
                    self.get_logger().error(f"History error: {e}")
                self.history_fetched = True
            else:
                self.window.after(300, check)

        self.window.after(300, check)

    def run_gui(self):
        self.window.mainloop()


def main():
    rclpy.init()
    node = ChatGUI()
    thread = Thread(target=rclpy.spin, args=(node,), daemon=True)
    thread.start()
    try:
        node.run_gui()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()