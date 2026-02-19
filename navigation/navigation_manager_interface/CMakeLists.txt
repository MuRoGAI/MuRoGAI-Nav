cmake_minimum_required(VERSION 3.8)
project(navigation_manager_interface)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/NavigationRobotRequest.msg"
  "msg/CancelNavigationRequest.msg"
  "msg/RobotGoalStatus.msg"
  "msg/RobotPath.msg"
  "msg/RobotPathArray.msg"
  "msg/StopRobotsRequest.msg"
)

ament_package()
