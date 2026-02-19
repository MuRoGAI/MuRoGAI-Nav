# DiffDriveTrajectory.msg
# Custom message for target trajectory with time-stamped waypoints and velocities

float64[] time    # Array of timestamps
float64[] x       # Array of x positions
float64[] y       # Array of y positions
float64[] theta   # Array of heading angles (yaw) in radians
float64[] v       # Array of linear velocities
float64[] omega   # Array of angular velocities