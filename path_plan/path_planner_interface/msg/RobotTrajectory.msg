# RobotTrajectory.msg
# robot trajectories (both diff-drive and holonomic)

string robot_name
string robot_type
DiffDriveTrajectory[] diff_drive_trajectories
HoloTrajectory[] holo_trajectories