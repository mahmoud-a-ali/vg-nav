# vg-nav
Visual-Geometry GP-based Navigable Space for Autonomous Navigation

Steps to run the code:
- roslaunch jackal_gazebo postoffice_mud.launch
- rosrun vg_gpn vg_sync_node.py
- rosrun vg_gpn rgb_seg 
- rosrun rqt_reconfigure rqt_reconfigure 
- roslaunch vg_gpn vg_gpn_real.launch 
- roslaunch waypts_nav_pid pid_tracking.launch 


To set up realsense
- export JACKAL_URDF_EXTRAS=Path to folder/realsense.urdf.xacro
- export JACKAL_URDF_EXTRAS=realsense.urdf.xacro 