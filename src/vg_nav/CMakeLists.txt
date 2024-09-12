cmake_minimum_required(VERSION 3.0.2)
project(vg_nav)

# add_compile_options(-std=c++11)

find_package(Eigen3 REQUIRED)


## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_filters
  message_generation
  message_runtime
  nav_msgs
  pcl_conversions
  pcl_msgs
  pcl_ros
  roscpp
  rospy
  sensor_msgs
  std_msgs
  cv_bridge
  dynamic_reconfigure
)

catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## Generate messages in the 'msg' folder
add_message_files(
  FILES
  PclObsv.msg
  PosePcl2.msg
  PoseScan.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

generate_dynamic_reconfigure_options(
  cfg/ThresholdConfig.cfg  ## Add your dynamic reconfigure configuration file here
)

###################################
## catkin specific configuration ##
###################################
catkin_package(
 CATKIN_DEPENDS message_generation message_runtime roscpp rospy std_msgs geometry_msgs
)

###########
## Build ##
###########

include_directories(
# include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${Eigen_INCLUDE_DIRS}
)


add_executable(rgb_seg src/rgb_seg.cpp)
target_link_libraries(rgb_seg ${catkin_LIBRARIES} )# ${PCL_LIBRARES} ${Eigen_LIBRARIES})
#############
## Install ##
#############
catkin_install_python(PROGRAMS
  scripts/vg_nav_node.py 
  scripts/vg_sync_node.py 
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
