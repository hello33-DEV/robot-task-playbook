cmake_minimum_required(VERSION 3.0.2)
project(aic_competition)
find_package(catkin REQUIRED)
if(PYTHON_VERSION_MAJOR LESS 3)
  message(FATAL_ERROR "Python 3 ROS bindings are required; verify the target ROS environment.")
endif()
catkin_python_setup()
catkin_package()
catkin_install_python(PROGRAMS
  scripts/competition_node.py scripts/safety_bridge.py scripts/local_vision.py
  tools/record_waypoint.py tools/observe_targets.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION})
install(DIRECTORY launch config assets DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
