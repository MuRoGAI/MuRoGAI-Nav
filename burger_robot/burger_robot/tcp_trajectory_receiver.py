#!/usr/bin/env python3
# tcp_trajectory_receiver.py

import socket
import struct
import json

HOST = "192.168.0.129"   # IP of the robot / machine running the node
PORT = 5001


with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.connect((HOST, PORT))

    # Read 4-byte length header
    raw_len = s.recv(4)
    msg_len = struct.unpack(">I", raw_len)[0]

    # Read exactly msg_len bytes
    data = b""
    while len(data) < msg_len:
        chunk = s.recv(msg_len - len(data))
        if not chunk:
            break
        data += chunk

trajectory = json.loads(data.decode("utf-8"))
print(json.dumps(trajectory, indent=2))