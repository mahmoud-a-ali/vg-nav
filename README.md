# vg-nav
Visual-Geometry GP-based Navigable Space for Autonomous Navigation

## Running the code
1. Run Simulation Environment and spawn jackal
```bash
roslaunch jackal_gazebo postoffice_mud.launch
```
2. Launch the `vg_nav.launch` to run the `nav_sync_node.py`, the `rgb_seg`, and the `vg_nav_node`
```bash
roslaunch vg_gpn vg_gpn_real.launch 
```
- Note: for simulation we are doing the segmentation based on the RGB values of the raw image. To tune the minimum and maximum values for each channel, we use the `rq_reconfigure` package. the values working for the postoffice_mud environment is stored in the `config` file. After you tune the RGB values and check the generated `nav_image` in `rviz`, run the `pid_tracking_node`.
3. Run the PID controller to move the robot to the select visual-geometry local navigation point `VG-LNP`
```bash
roslaunch waypts_nav_pid pid_tracking.launch 
```

## Notes
- export JACKAL_URDF_EXTRAS= "path to `realsense.urdf.xacro`" [export JACKAL_URDF_EXTRAS=realsense.urdf.xacro], this to ensure that the `realsense` plugin works in gazebo
- For realhardware experiments we used `mask2former` for segmenting the camera image and based on which classes we define as navigable, we generate the `nav_img` using the `mask2former_nav_img.py` script. to use this script, you need to first follow the `mask2former` readme file to install it and make it works.


## rosgraph as a reference for different nodes connection

