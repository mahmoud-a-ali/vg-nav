# GP-Frontiers  for Local Mapless Navigation 

## Dependencies
- cuda 
- tensorflow (TF)
- tensorflow_probability (TFP)
- gpflow (gpf)
#### Compatibility: 
- The code was tested using the following versions of dependencies: nvidia-driver 470, cuda 11.4, cudnn 8.2.4, TF 2.11.0, TFP 0.19.0, gpf 2.5.1.
- For building dependencies, check this repository [install] 

## RUN
#### Launch simulated environment:
Either one of the following environments
###### launch the cluttered environment
```
roslaunch jackal_gazebo gap_env_1.launch
```
###### launch the Maze-like environment
```
roslaunch jackal_gazebo gap_env_2.launch
```
#### Launch `gpf_navigate.launch` file from the `vsgp_nav_glb` package:
```
roslaunch vsgp_nav_glb gpf_navigate.launch motion:=true
```



## Files
### vsgp_nav_glb
- `oc_srfc_proj.cpp` : ros node to convert pointcloud from cartesian to spherical 
- `navigate_pcl.py` : ros node which form the SGP, predict the occupancy and variance surfaces and extract GP-Frontiers
- `nav_pcl.yaml` : yaml file contains parameters values 
- `gpf_navigate.launch`: launch file to launch the necessary nodes to navigate using GP-Frontier navigation


### diff_drive
PID controller to drive differential robot (Jackal) to a goal point.
### working jackal packages:
Packages required to simulate Jackal Robot, adapted from clearpath reposirtory with some custom modification. 


[install]:<https://github.com/mahmoud-a-ali/install_nvidiaDriver_cuda_cudnn_tensorflow_gpflow>



