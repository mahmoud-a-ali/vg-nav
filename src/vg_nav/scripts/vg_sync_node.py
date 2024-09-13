#!/usr/bin/env python
import rospy
import numpy as np
from sensor_msgs.msg import Image, PointCloud2
from cv_bridge import CvBridge
import ros_numpy
import time
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry


class CamImgPclDownsampling:
    def __init__(self):
        self.bridge = CvBridge()
        self.crnt_pos = None
        self.old_pos = None

        self.sync_header_seq = 0
        self.cam_img_sub = Subscriber('/cam_img', Image)
        self.cam_pcl_sub = Subscriber('/cam_pcl', PointCloud2)
        self.vlp_pcl_sub = Subscriber('/vlp_pcl', PointCloud2)
        self.odom_sub    = Subscriber('/odom', Odometry)

        self.ats = ApproximateTimeSynchronizer([self.cam_img_sub, self.cam_pcl_sub, self.odom_sub, self.vlp_pcl_sub], queue_size=1, slop=0.1)        
        self.ats.registerCallback(self.sync_callback)


        self.raw_img_pub = rospy.Publisher('/sync_cam_img', Image, queue_size=1)
        self.raw_pcl_pub = rospy.Publisher('/sync_cam_pcl', PointCloud2, queue_size=1)
        self.vlp_pcl_pub = rospy.Publisher('/sync_vlp_pcl', PointCloud2, queue_size=1)
        self.odom_pub    = rospy.Publisher('/sync_odom', Odometry, queue_size=1)
        print("aha a...")


    def odom_cb(self, odom_msg ):
        pos_z = odom_msg.pose.pose.position.z
        pos_x = odom_msg.pose.pose.position.x
        pos_y = odom_msg.pose.pose.position.y
        quaternion = [
            odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w
        ]
        euler = euler_from_quaternion(quaternion)
        pos_rol  = euler[0]
        pos_ptch = euler[1]
        pos_yw   = euler[2]
        self.crnt_pos = [pos_x, pos_y, pos_z, pos_rol, pos_ptch, pos_yw] 
        # print("current pose: ", self.crnt_pos)
 

    def sync_callback(self, cam_img_msg, cam_pcl_msg, odom_msg, vlp_pcl_msg):
        t1 = time.time()
        self.odom_cb(odom_msg)

        rbt_motion = False  
        if rbt_motion: # if true, then we won't send new obs till robot moves a distance > dst_th or rotate > orn_th
            if self.old_pos == None:
                print("first observation, sending new data")
            else:   
                dst_th = 0.1
                orn_th  = 0.3
                mvd_dst = np.sqrt( (self.crnt_pos[0]-self.old_pos[0])**2 + (self.crnt_pos[1]-self.old_pos[1])**2 + (self.crnt_pos[2]-self.old_pos[2])**2)
                rot_orn  = np.sqrt( (self.crnt_pos[3]-self.old_pos[3])**2 + (self.crnt_pos[4]-self.old_pos[4])**2 + (self.crnt_pos[5]-self.old_pos[5])**2)
                if  mvd_dst > dst_th or rot_orn > orn_th:
                    # print("robot moved, sending new data")
                    pass
                else:
                    return 0

        cam_img = self.bridge.imgmsg_to_cv2(cam_img_msg, desired_encoding='passthrough')
        point_cloud_array   = ros_numpy.point_cloud2.pointcloud2_to_array(cam_pcl_msg, squeeze = True)
        cam_pcl = np.zeros_like(point_cloud_array)
        # print("cam_img, point_cloud_array, cam_pcl: ", self.cam_img.shape, point_cloud_array.shape, cam_pcl.shape)

        cam_pcl[:,:]['x'] = point_cloud_array[:,:]['x']
        cam_pcl[:,:]['y'] = point_cloud_array[:,:]['y']
        cam_pcl[:,:]['z'] = point_cloud_array[:,:]['z']
        normalized_rgb = cam_img[:, :, :3] / 255.0

        rgb_packed = (normalized_rgb * 255).astype(np.uint32)
        rgb_packed = np.left_shift(rgb_packed[:, :, 0], 16) | np.left_shift(rgb_packed[:, :, 1], 8) | rgb_packed[:, :, 2]
        rgb_float = rgb_packed.view(np.float32)
        rgb_float = rgb_float.reshape(cam_img.shape[0], cam_img.shape[1])
        cam_pcl[:, :]['rgb'] = rgb_float
        
        cam_pcl = cam_pcl[::2, ::2]  # for half size 
        # cam_pcl = cam_pcl[::4, ::4] # for quarter size 
        # print("cam_pcl.shape: ", cam_pcl.shape)
        rgb_packed = cam_pcl[:, :]['rgb'].view(np.uint32)
        r = np.right_shift(np.bitwise_and(rgb_packed, 0x00FF0000), 16).astype(np.uint8)
        g = np.right_shift(np.bitwise_and(rgb_packed, 0x0000FF00), 8).astype(np.uint8)
        b = np.bitwise_and(rgb_packed, 0x000000FF).astype(np.uint8)
        rgb_image = np.stack((r, g, b), axis=-1)
        dwnsmpl_cam_img_msg = self.bridge.cv2_to_imgmsg(rgb_image, "bgr8")
        dwnsmpl_cam_img_msg.header = cam_img_msg.header
        

        dwnsmpl_cam_pcl_msg = PointCloud2()
        dwnsmpl_cam_pcl_msg.header = cam_pcl_msg.header
        dwnsmpl_cam_pcl_msg.height = cam_pcl.shape[0] # 640 #160 #640  # Assuming all points are in a single row
        dwnsmpl_cam_pcl_msg.width = cam_pcl.shape[1]
        dwnsmpl_cam_pcl_msg.fields = cam_pcl_msg.fields
        dwnsmpl_cam_pcl_msg.is_bigendian = False
        dwnsmpl_cam_pcl_msg.point_step = cam_pcl_msg.point_step
        dwnsmpl_cam_pcl_msg.row_step = dwnsmpl_cam_pcl_msg.point_step * dwnsmpl_cam_pcl_msg.width
        dwnsmpl_cam_pcl_msg.is_dense = True
        dwnsmpl_cam_pcl_msg.data = cam_pcl.tobytes()

        # print("cam_pcl  : ", cam_pcl_msg.header.stamp.to_sec())
        # print("cam_img  : ", cam_img_msg.header.stamp.to_sec())
        # print("odom     : ", odom_msg.header.stamp.to_sec())
        # print("vlp_pcl  : ", vlp_pcl_msg.header.stamp.to_sec())

        cam_pcl_msg.header.seq = self.sync_header_seq
        cam_img_msg.header.seq = self.sync_header_seq
        vlp_pcl_msg.header.seq = self.sync_header_seq
        odom_msg.header.seq    = self.sync_header_seq

        self.raw_img_pub.publish(dwnsmpl_cam_img_msg)
        self.raw_pcl_pub.publish(dwnsmpl_cam_pcl_msg)
        self.vlp_pcl_pub.publish(vlp_pcl_msg)
        self.odom_pub.publish(odom_msg)

        self.old_pos = self.crnt_pos
        self.sync_header_seq += 1 
        print("tot t: ", time.time()-t1)




if __name__ == '__main__':
    rospy.init_node('cam_img_pcl_dwnsmpl_node')
    cam_img_pcl_dwnsmplng = CamImgPclDownsampling()
    rospy.spin()
