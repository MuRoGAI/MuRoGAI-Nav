#!/usr/bin/env python3

import subprocess
import time


WORLD = "food_court"
cake_uri = "model://Birthday_cake"
speaker_uri = "model://JBL_Speaker"
gif_box_uri = "model://gift_box"


salad_uri = "model://SANDWICH_MEAL"

robot1_pose = [5.0, 7.5, 1.0]
robot2_pose = [5.0, 4.5, 1.0]
robot3_pose = [7.0, 6.0, 1.0]

spawn_pose_cake     = [6.9, 10, 0.5]
spawn_pose_speaker  = [7.5, 10, 0.5]
spawn_pose_gift_box = [6.5, 10, 0.5]

robot1_pose_salad_stall = [6.05, 9.75, 0.5]
robot2_pose_salad_stall = [6.14, 6.62, 0.5]
robot3_pose_salad_stall = [8.32, 8.32, 0.5]

robot1_pose_juice_stall = [6.9, 10, 0.5]
robot2_pose_juice_stall = [7.5, 10, 0.5]
robot3_pose_juice_stall = [6.5, 10, 0.5]

table2_gift_box_pose = [11.7, 2.0, 0.5]
table2_speaker_pose  = [12.3, 2.0, 0.5]
table2_cake_pose     = [12.0, 2.0, 0.5]

sink1 = [22, 6, 0.65]
sink2 = [3, 14, 0.65]

# ------------------ SPAWN ------------------
def spawn_model(name, model_uri, pose=[0.0, 0.0, 0.0]):
    x, y, z = pose

    sdf = f"<sdf version='1.7'><include><uri>{model_uri}</uri></include></sdf>"

    req = (
        f'name: "{name}" '
        f'sdf: "{sdf}" '
        f'pose: {{ position: {{ x: {x}, y: {y}, z: {z} }} }}'
    )

    cmd = [
        "ign", "service",
        "-s", "/world/food_court/create",
        "--reqtype", "ignition.msgs.EntityFactory",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ]

    subprocess.run(cmd)
    print(f"Spawned: {name}")


# ------------------ DELETE ------------------
def delete_model(name):

    req = f'name: "{name}" type: MODEL'

    cmd = [
        "ign", "service",
        "-s", f"/world/{WORLD}/remove",
        "--reqtype", "ignition.msgs.Entity",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "2000",
        "--req", req
    ]

    subprocess.run(cmd)
    print(f"Deleted: {name}")


# ------------------ MOVE ------------------
def move_model(name, pose=[0.0, 0.0, 0.0]):
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
        "--req", req
    ]

    subprocess.run(cmd)
    print(f"Moved: {name} → ({x}, {y}, {z})")



def first_spawn():
    spawn_model("cake", cake_uri,spawn_pose_cake)
    spawn_model("speaker", speaker_uri, spawn_pose_speaker)
    spawn_model("gift", gif_box_uri, spawn_pose_gift_box)


def move_cake_speakeer_gift_to_formation():
    move_model("gift", robot1_pose_salad_stall)
    move_model("speaker",  robot2_pose_salad_stall)
    move_model("cake",     robot3_pose_salad_stall)

def move_cake_speaker_gift_to_child_table():
    move_model("gift", table2_gift_box_pose)
    move_model("speaker",  table2_speaker_pose)
    move_model("cake",     table2_cake_pose)

# ------------------ MAIN ------------------
def main():

    # first_spawn()

    # move_cake_speakeer_gift_to_formation()

    move_cake_speaker_gift_to_child_table()





    # spawn_model("cake", cake_uri,robot3_pose)
    # spawn_model("speaker", speaker_uri, robot2_pose)
    # spawn_model("gift", gif_box_uri, robot1_pose)


    # time.sleep(10)



    # move_model("cake", robot3_pose)
    # time.sleep(3)

    # move_model("cake", robot1_pose)
    # time.sleep(3)

    # delete_model("cake")
    # delete_model("speaker")
    # delete_model("gift")


    # time.sleep(10)


if __name__ == "__main__":
    main()