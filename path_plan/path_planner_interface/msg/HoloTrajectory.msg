# HoloTrajectory.msg
# Custom message for target trajectory with time-stamped waypoints and velocities

float64[] time  # Array of timestamps
float64[] x     # Array of x positions corresponding to each timestamp
float64[] y     # Array of y positions corresponding to each timestamp
float64[] vx    # Array of x velocities corresponding to each timestamp
float64[] vy    # Array of y velocities corresponding to each timestamp