cmake_minimum_required(VERSION 3.0.2)
project(waypts_nav_pid)

# add_compile_options(-std=c++11)


find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp
  rospy
  std_msgs
  tf
  visualization_msgs
)

find_package(catkin REQUIRED dynamic_reconfigure)
catkin_python_setup()


generate_dynamic_reconfigure_options(
  cfg/Params.cfg
)


catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES waypts_nav_pid
#  CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs tf visualization_msgs
#  DEPENDS system_lib
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
)


#############
## Install ##
#############

catkin_install_python(PROGRAMS
  scripts/tracking_pid_node.py
  scripts/jackal_sim_pid.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)


