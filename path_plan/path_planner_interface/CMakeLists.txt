cmake_minimum_required(VERSION 3.8)
project(path_planner_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/PathPlannerRequest.msg"
  "msg/RobotPath.msg"
  "msg/RobotPathArray.msg"
  "msg/RobotCmd.msg"
  "msg/RobotFleetCmd.msg"
  "msg/HoloTrajectory.msg"
  "msg/DiffDriveTrajectory.msg"
  "msg/RobotTrajectory.msg"
  "msg/RobotTrajectoryArray.msg"
)

ament_package()
