import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

import numpy as np
from time import time
from scipy import stats
from datetime import datetime

import rospy
import ros_numpy

from sensor_msgs.msg import Image, PointCloud2
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointField
from sensor_msgs import point_cloud2
from std_msgs.msg import Header
from geometry_msgs.msg import PoseStamped
from message_filters import ApproximateTimeSynchronizer, Subscriber
from tf.transformations import euler_from_quaternion, quaternion_from_euler
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud

from cv_bridge import CvBridge 
from tf.transformations import quaternion_from_euler

from sgp2d import SGP2D
from nav_utils import convert_spherical_to_cartesian, convert_cartesian_to_spherical, normalize_array, create_quaternion

import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="-1" # to disable GPU
tf.device("gpu:0") ### to select GPU

#@ class VGNav
class VGNav:
    def __init__(self):
        rospy.init_node("vg_nav")
        print("##############################################")
        print("    Initialize visual_geometry_navigation     ")
        print("##############################################")

        #### subscriber 
        self.cam_pcl_sub   = Subscriber('sync_cam_pcl', PointCloud2)
        self.vlp_pcl_sub   = Subscriber("sync_vlp_pcl", PointCloud2)
        self.odom_sub      = Subscriber("sync_odom", Odometry)
        self.nav_img_sub   = Subscriber("nav_image", Image) 

        ##### sync filter 
        self.ats = ApproximateTimeSynchronizer([self.vlp_pcl_sub, self.odom_sub, self.nav_img_sub, self.cam_pcl_sub], queue_size=1, slop=0.1)
        self.ats.registerCallback(self.sync_callback)


        ######  publisher 
        self.org_oc_srfc_pub     = rospy.Publisher( "lfrq_org_oc_srfc", PointCloud2, queue_size=1)
        self.gp_oc_srfc_pub      = rospy.Publisher( "gp_nav_oc", PointCloud2, queue_size=1)
        self.gp_var_srfc_pub     = rospy.Publisher( "gp_nav_var", PointCloud2, queue_size=1)

        self.gpfs_wrld_pub       = rospy.Publisher( "sb_gls_xyz_wrld", PointCloud2, queue_size=1)
        self.gpfs_bslnk_pub      = rospy.Publisher( "gpfs_vldyn", PointCloud2, queue_size=1)
        self.gpf_camlnk_pub      = rospy.Publisher( "fr_cam_fov", PointCloud2, queue_size=1)
        
        self.nav_pcl_pub         = rospy.Publisher( "nav_pcl", PointCloud2, queue_size=1)
        self.nav_srfc_pub        = rospy.Publisher( "nav_srfc", PointCloud2, queue_size=1)
        self.nav_gp_pub          = rospy.Publisher( "nav_gp", PointCloud2, queue_size=1)
        
        self.trgt_gl_wrld_pub    = rospy.Publisher( "gl_wrt_wrld", PointCloud2, queue_size=1)
        self.lcl_nav_cmd_pub     = rospy.Publisher( "move_base_simple/goal", PoseStamped, queue_size = 1)  
        

        ######## pcl_pose ########
        self.pcl_skp        = rospy.get_param('/pcl_skp')
        self.pose           = None 
        self.org_unq_thetas = None
        self.pcl_unq_thetas = None
        self.vlp_thetas     = None
        self.vlp_alphas     = None
        self.vlp_rds        = None
        self.vlp_oc         = None
        self.vlp_sz         = None
        
        ######## Navigation #######
        self.oc_srfc_rds   = rospy.get_param('/oc_srfc_rds') 
        self.vlp_gp_grd     = None
        self.vlp_gp_grd_w   = None
        self.vlp_gp_grd_h   = None
        self.vlp_grd_ths = None
        self.vlp_gp_als = None
        self.vlp_gp_grd_oc  = None
        self.vlp_gp_grd_rds = None
        self.vlp_gp_grd_var = None
        self.gp_nav_var_thrshld = None

        ######## Visual-Geometry GPFrontiers #######
        self.optm_gpf  = None 
        self.gpfs_cost   = None 
        self.gpfs_sph = None 
        self.optm_gpfs_ars = None 

        self.gmtry_nav_mode     = rospy.get_param('/gmtry_nav_mode')  
        self.v_nvgblti_th       = rospy.get_param('/v_nvgblti_th')  
        self.gp_nav_indpts_sz   = rospy.get_param('/gp_nav_indpts_sz') 
        self.k_dir              = rospy.get_param('/k_dir')
        self.k_dst              = rospy.get_param('/k_dst') 
        self.k_nav              = rospy.get_param('/k_nav') 
        self.sb_gl_dst          = rospy.get_param('/sb_gl_dst')
        self.var_km             = rospy.get_param('/var_km') 
        self.var_kv             = rospy.get_param('/var_kv') 
        self.gp_nav_var_img_viz = rospy.get_param('/gp_nav_var_img_viz') 
        self.gp_nav_aug_data    = rospy.get_param('/gp_nav_aug_data') 
        self.gp_nav_var_viz     = rospy.get_param('/gp_nav_var_viz')
        self.org_oc_srfc_rds_viz= rospy.get_param('/org_oc_srfc_rds_viz') 
        self.gp_oc_srfc_rds_viz = rospy.get_param('/gp_oc_srfc_rds_viz') 
        self.nav_gp_rds_viz     = rospy.get_param('/nav_gp_rds_viz') 
        self.gl_x               = rospy.get_param('/gl_x')
        self.gl_y               = rospy.get_param('/gl_y')
        self.log_data           = rospy.get_param('/log_data')
        self.sim           = rospy.get_param('/sim')
        self.gl_wrt_wrld        = np.array([self.gl_x, self.gl_y, 1], dtype="float32")
        self.vldyn_hdr    =  Header()
        self.vldyn_hdr.frame_id    =  "velodyne"
        self.gp_nav_var_pblsh   = True
        self.gp_nav_xypts = None

        ######## navigability ########
        self.nav_img_rcvd   = False
        self.cam_gp_grid()
        self.vlp_gp_grid()
        self.gpf_cond1 = np.array([0, 0, 0, 1, 1, 1, 1])
        self.gpf_cond2 = np.array([0, 0, 1, 1, 1, 1, 1])
        self.gpf_cond3 = np.array([1, 0, 1, 1, 1, 1, 1])
        self.gpf_cond4 = np.array([0, 0, 1, 1, 1, 1, 1])
        self.wrld_hdr = Header()
        self.wrld_hdr.frame_id =  "world" #"J0/velodyne"    #   "camera_init"# "aft_mapped" 
        self.fields = [PointField('x', 0, PointField.FLOAT32, 1),
                  PointField('y', 4, PointField.FLOAT32, 1),
                  PointField('z', 8, PointField.FLOAT32, 1),
                  PointField('intensity', 12, PointField.FLOAT32, 1)]


        #### sim vs real 
        if self.sim:
            self.cam_lnk_frame = "front_realsense"
            self.bslnk_vldyne_z = 0.322
        else:
            self.cam_lnk_frame = "camera_link"
            self.bslnk_vldyne_z = 0.4
            
        if self.log_data:
            self.data_files()

        self.print_ros_param()    
        rospy.sleep(1)
        rospy.spin()


    def data_files(self):
        self.geo_fit = None
        self.geo_prdct_t = None
        self.vis_fit_t = None
        self.vis_prdct_t = None
        timestr = datetime.now() 
        vis_geo_dir = "../../../logs/"
        itr_folder = vis_geo_dir + "exp_" + str(timestr)
        print("save data to ", itr_folder)
        if not os.path.exists(itr_folder):
            os.mkdir(itr_folder)
        # self.param_file  = itr_folder +"/hyper_param.txt"
        # self.trng_pred_var_file = itr_folder +"/trng_pred_var.txt"
        self.odom_file = itr_folder +"/odom.txt"
        self.time_file = itr_folder +"/time.txt"


    def print_ros_param(self):
        print("############# parameters ###########")  
        print("oc_srfc_rds        : ", self.oc_srfc_rds                )  
        print("gp_nav_indpts_sz   : ", self.gp_nav_indpts_sz   )  
        print("k_dir              : ", self.k_dir       )  
        print("k_dst              : ", self.k_dst      )  
        print("gp_nav_var_thrshld : ", self.gp_nav_var_thrshld )  
        print("sb_gl_dst          : ", self.sb_gl_dst    ) 
        print("gp_nav_var_img_viz : ", self.gp_nav_var_img_viz )
        print("gp_nav_aug_data    : ", self.gp_nav_aug_data    )
        print("gl_x               : ", self.gl_x        )
        print("gl_y               : ", self.gl_y        )
        print("#####################################")  



    def gp_nav_fit(self, ls1, ls2, var, alpha, noise, noise_var):
        self.gp_nav = SGP2D()
        self.gp_nav.set_kernel_param(ls1, ls2, var, alpha, noise, noise_var)
        self.gp_nav.set_training_data(self.vlp_gp_din, self.vlp_gp_dout)
        self.gp_nav.set_indpts_from_training_data(self.gp_nav_indpts_sz, self.vlp_gp_din)#(indpts_size, data_size)
        self.gp_nav.set_sgp_model()
        self.gp_nav.select_trainable_param()
        self.gp_nav.minimize_loss()
        # self.gp_nav.adam_optimize_param() 
        
    def cam_gp_grid(self): 
        th_rsltion = 0.02 #0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.02 #0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        # self.cam_grd_ths = np.arange( -np.pi/2, np.pi/2, th_rsltion, dtype='float32')
        # self.cam_grd_als = np.arange(-0.3710505, 0.37016165,   al_rsltion, dtype='float32') 
        self.cam_grd_ths = np.arange( -0.47, 0.47, th_rsltion, dtype='float32')
        self.cam_grd_als = np.arange( 1.2, 1.95,   al_rsltion, dtype='float32')  
        self.cam_gp_grd = np.array(np.meshgrid(self.cam_grd_ths,self.cam_grd_als), dtype='float32').T.reshape(-1,2)
        self.cam_gp_grd_w = np.shape(self.cam_grd_ths)[0] 
        self.cam_gp_grd_h = np.shape(self.cam_grd_als)[0]
        print("grid: ", np.shape(self.cam_gp_grd), self.cam_gp_grd_w, self.cam_gp_grd_h)
     

    def calibrate_fov(self):
        ## ############ calibrate cam baselink axes using 4 corners############
        ## /base_link /velodyne: Translation: [0.000, 0.000, 0.322]
        ## /velodyne /camera_link: Translation: [0.220, 0.000, -0.216]
        ## /base_link /camera_link: Translation: [0.220, 0.000, 0.184]

        cam_crnr_ths = np.array( [-0.47794747, 0.47894406], dtype='float32')
        cam_crnr_als = np.array( [1.2, 1.95], dtype='float32')  
        cam_crnr_grd = np.array(np.meshgrid(cam_crnr_ths, cam_crnr_als), dtype='float32').T.reshape(-1,2)
        print("\ncam_crnr_grd: ", cam_crnr_grd)

        cam_crnr_dst = 9*np.ones(cam_crnr_grd.shape[0])
        x_cam, y_cam, z_cam = convert_spherical_to_cartesian(cam_crnr_grd.T[0], cam_crnr_grd.T[1], cam_crnr_dst)
        cam_crnr_xyz = np.column_stack((x_cam, y_cam, z_cam, cam_crnr_dst))
        print("\ncam_crnr_xyz: ", x_cam, y_cam, z_cam)

        x_vlp = x_cam + 0.22
        y_vlp = y_cam 
        z_vlp = z_cam - 0.184
        vlp_crnr_xyz = np.column_stack((x_vlp, y_vlp, z_vlp, 0*cam_crnr_dst))
        print("\nvlp_crnr_xyz: ", x_vlp, y_vlp, z_vlp)
        vlp_crnr_ths, vlp_crnr_als, dst_crnr_als = convert_cartesian_to_spherical(x_vlp, y_vlp, z_vlp)
        print("\nvlp_crnr_tad: ", vlp_crnr_ths, vlp_crnr_als, dst_crnr_als)

        crnr_pcl = np.row_stack((cam_crnr_xyz, vlp_crnr_xyz))
        print("\ncrnr_pcl: ", crnr_pcl.shape)

        nav_srfc_header = Header()
        nav_srfc_header.frame_id = self.cam_lnk_frame 
        pcl2 = point_cloud2.create_cloud(nav_srfc_header, self.fields, crnr_pcl)
        self.gpf_camlnk_pub.publish(pcl2)


    def  gp_nav_img_pcl_cb(self, nav_img_msg, cam_pcl_msg):
        print("nav_img_msg.header.seq :", nav_img_msg.header.seq)
        print("nav_img_msg.header.seq :", nav_img_msg.header.seq)
        self.nav_img_header = nav_img_msg.header
        self.nav_img = CvBridge().imgmsg_to_cv2(nav_img_msg)
        t1 = time()
        cam_pcl_arr   = ros_numpy.point_cloud2.pointcloud2_to_array(cam_pcl_msg, squeeze = True)
        # print("cam_pcl_arr data: ", cam_pcl_arr.shape)
        y_vec = - cam_pcl_arr[:,:]['x'].reshape(-1, 1)
        z_vec = - cam_pcl_arr[:,:]['y'].reshape(-1, 1)
        x_vec = cam_pcl_arr[:,:]['z'].reshape(-1, 1)
        nvgblty_vec = self.nav_img.reshape(-1, 1)

        self.cam_pcl = np.column_stack( (x_vec, y_vec, z_vec, nvgblty_vec) )
        self.nav_pcl = np.column_stack( (x_vec[nvgblty_vec==255], y_vec[nvgblty_vec==255], z_vec [nvgblty_vec==255], nvgblty_vec[nvgblty_vec==255]) )
        self.cam_o_nav_pcl = self.cam_pcl
        # self.cam_o_nav_pcl = self.nav_pcl
        # print("self.cam_o_nav_pcl data: ", self.cam_o_nav_pcl.shape)
        # print("nav pcl time: ", time() - t1)

        cam_o_nav_pcl_header = Header()
        cam_o_nav_pcl_header.frame_id =  self.cam_lnk_frame
        pcl2 = point_cloud2.create_cloud(cam_o_nav_pcl_header, self.fields, self.cam_o_nav_pcl)
        self.nav_pcl_pub.publish(pcl2)
        print("Publish nav pcl time: ", time() - t1)
        # return 0

        ################################ seg srfc ######################
        t2 = time()
        #limit points based on minimum and maximum distance value
        dst = np.sqrt(self.cam_o_nav_pcl.T[0]**2 + self.cam_o_nav_pcl.T[1]**2 + self.cam_o_nav_pcl.T[2]**2)
        self.cam_o_nav_pcl = self.cam_o_nav_pcl[dst > 0.4]
        dst = np.sqrt(self.cam_o_nav_pcl.T[0]**2 + self.cam_o_nav_pcl.T[1]**2 + self.cam_o_nav_pcl.T[2]**2)
        self.cam_o_nav_pcl = self.cam_o_nav_pcl[dst < 20]

        # print("self.cam_o_nav_pcl non zeros data: ", self.cam_o_nav_pcl.shape)
        dst = np.sqrt(self.cam_o_nav_pcl.T[0]**2 + self.cam_o_nav_pcl.T[1]**2 + self.cam_o_nav_pcl.T[2]**2)
        thts = np.arctan2(self.cam_o_nav_pcl.T[1] , self.cam_o_nav_pcl.T[0])
        alphs = np.arccos(self.cam_o_nav_pcl.T[2] / dst)
        # print("shape dst: ", dst.shape )
        # print("min and max dst: ", np.min(dst), np.max(dst) )
        # print("min and max thts: ", np.min(thts), np.max(thts) )
        # print("min and max alphs: ", np.min(alphs), np.max(alphs) )
        # print("min and max nav: ",np.min(self.cam_o_nav_pcl.T[3]), np.max(self.cam_o_nav_pcl.T[3]) )
        self.nav_gp_rds_viz = 9
        nav_srfc_o_pcl = 1
        if nav_srfc_o_pcl:
            seg_srfc_x = self.nav_gp_rds_viz * np.sin(alphs) * np.cos(thts)
            seg_srfc_y = self.nav_gp_rds_viz * np.sin(alphs) * np.sin(thts)
            seg_srfc_z = self.nav_gp_rds_viz * np.cos(alphs)
        else:
            seg_srfc_x = dst * np.sin(alphs) * np.cos(thts)
            seg_srfc_y = dst * np.sin(alphs) * np.sin(thts)
            seg_srfc_z = dst * np.cos(alphs)


        # print("seg_srfc_x shape:  ", seg_srfc_x.shape )
        print("nav srfc time: ", time()-t2)
        seg_gp_pcl = np.column_stack( (seg_srfc_x.reshape(-1,1), seg_srfc_y.reshape(-1,1), seg_srfc_z.reshape(-1,1), self.cam_o_nav_pcl.T[3].reshape(-1,1)) )
        nav_srfc_header = Header()
        nav_srfc_header.frame_id = self.cam_lnk_frame 
        pcl2 = point_cloud2.create_cloud(nav_srfc_header, self.fields, seg_gp_pcl)
        self.nav_srfc_pub.publish(pcl2)
        print("Publish nav srfc time: ", time()-t2 )
        # return 0

        ############################## sgp ##############################
        t3 = time()
        #limit points by skip percentage of points 
        skp = min(10, self.cam_o_nav_pcl.shape[0])
        self.cam_o_nav_pcl = self.cam_o_nav_pcl[::skp]
        print("self.cam_o_nav_pcl after skp: ", self.cam_o_nav_pcl.shape)

        dst = np.sqrt(self.cam_o_nav_pcl.T[0]**2 + self.cam_o_nav_pcl.T[1]**2 + self.cam_o_nav_pcl.T[2]**2)
        thts = np.arctan2(self.cam_o_nav_pcl.T[1] , self.cam_o_nav_pcl.T[0])
        alphs = np.arccos(self.cam_o_nav_pcl.T[2] / dst)

        seg_gp_din  = np.column_stack( (thts.reshape(-1, 1), alphs.reshape(-1, 1)) ) 
        # seg_gp_dout = np.array(rgb_float.reshape(-1, 1), dtype='float32').reshape(-1,1)
        seg_gp_dout = np.array(self.cam_o_nav_pcl.T[3].reshape(-1, 1), dtype='float32').reshape(-1,1)
        print("seg_gp_din, seg_gp_dout:", seg_gp_din.shape, seg_gp_dout.shape)
        # gp_nav_fit(0.09, 0.11, 0.7, 10, 10, 0.005) #(ls1, ls2, var, alpha, noise, noise_var)
        # ls1, ls2, var, alpha, noise, noise_var = 0.09, 0.11, 0.7, 10, 10, 0.005
        ls1, ls2, var, alpha, noise, noise_var = 0.11, 0.15, 0.7, 10, 10, 0.005
        t_f = time()
        seg_gp = SGP2D()
        seg_gp.set_kernel_param(ls1, ls2, var, alpha, noise, noise_var)
        seg_gp.set_training_data(seg_gp_din, seg_gp_dout)
        seg_gp.set_indpts_from_training_data(300, seg_gp_din)#(indpts_size, data_size)
        
        seg_gp.set_sgp_model()
        seg_gp.select_trainable_param()
        seg_gp.minimize_loss()
        self.vis_fit_t = time() - t_f

        ### publish nav_gp_grid 
        pub_nav_gp = 1
        if pub_nav_gp:
            # grd_tnsr = tf.convert_to_tensor(self.cam_gp_grd, dtype=tf.float32)
            grd_tnsr = tf.Variable(self.cam_gp_grd,  dtype=tf.float32)
            gp_seg_cls, gp_seg_var = seg_gp.model.predict_f(grd_tnsr)

            gp_seg_var = gp_seg_var.numpy()
            self.gp_seg_cls  = gp_seg_cls.numpy()
            # print("self.gp_seg_cls.shape: ", self.gp_seg_cls.shape)
            print("min and max self.gp_seg_cls: ",np.min(self.gp_seg_cls), np.max(self.gp_seg_cls) )
            # print("nav_cls: ", nav_cls.numpy().shape, nav_cls.numpy()[0] )
            self.gp_seg_cls[ self.gp_seg_cls<120] = 0
            self.gp_seg_cls[ self.gp_seg_cls>120] = 1

            ############ visualize seg_gp  ############
            seg_gp_rds_viz = 10
            gp_pcl_x = seg_gp_rds_viz * np.sin(self.cam_gp_grd.T[1].reshape(-1,1)) * np.cos(self.cam_gp_grd.T[0].reshape(-1,1))
            gp_pcl_y = seg_gp_rds_viz * np.sin(self.cam_gp_grd.T[1].reshape(-1,1)) * np.sin(self.cam_gp_grd.T[0].reshape(-1,1))
            gp_pcl_z = seg_gp_rds_viz * np.cos(self.cam_gp_grd.T[1].reshape(-1,1))
            # gp_pcl_x = seg_gp_rds_viz * np.cos(self.cam_gp_grd.T[1].reshape(-1,1)) * np.cos(self.cam_gp_grd.T[0].reshape(-1,1))
            # gp_pcl_y = seg_gp_rds_viz * np.cos(self.cam_gp_grd.T[1].reshape(-1,1)) * np.sin(self.cam_gp_grd.T[0].reshape(-1,1))
            # gp_pcl_z = seg_gp_rds_viz * np.sin(self.cam_gp_grd.T[1].reshape(-1,1))
            
            # seg_gp_pcl = np.column_stack( (gp_pcl_z, -gp_pcl_y, gp_pcl_x, self.gp_seg_cls) )
            seg_gp_pcl = np.column_stack( (gp_pcl_x, gp_pcl_y, gp_pcl_z, self.gp_seg_cls) )
            print("nav gp time: ", time()-t3)

            cam_link_header = Header()
            cam_link_header.frame_id = self.cam_lnk_frame
            pcl2 = point_cloud2.create_cloud(cam_link_header, self.fields, seg_gp_pcl)
            self.nav_gp_pub.publish(pcl2)
            print("Publish nav gp time: ", time()-t3 )


 #################### sync vlp_pcl odom nav_img nav_pcl####################
    def sync_callback(self, vlp_pcl_msg, odom_msg, nav_img_msg, cam_pcl_msg):
        msg_rcvd_time = time()
        print("\n\n\n\n###################################") 
        
        self.publish_goal()
        self.odom_cb(odom_msg)
        # self.nav_img_cb(nav_img_msg, cam_pcl_msg)
        # self.nav_pcl_cb(cam_pcl_msg)
        self.vlp_pcl_cb(vlp_pcl_msg)
        # print("check 1")

        self.wrld_hdr.stamp= vlp_pcl_msg.header.stamp
        self.vldyn_hdr.stamp= vlp_pcl_msg.header.stamp
        self.pose = odom_msg.pose  
        self.tf_rbt_2_wrld() 
        self.tf_wrld_2_rbt()
        
        self.rbt2gl_error()
        if self.gl_dst_err < 0.8:
            print("-------------- GOAL REACHED -----------")
            self.stop_cmd()
            quit()

        # print("check 2")
        ######## organize input and outpu data for SGP navigation  ###############
        self.vlp_gp_din  = np.column_stack( (self.vlp_thetas, self.vlp_alphas) ) 
        self.vlp_gp_dout = np.array(self.vlp_oc, dtype='float').reshape(-1,1)
        # print("check 3")
        t_f = time()
        self.gp_nav_fit(0.09, 0.11, 0.7, 10, 10, 0.005) #(ls1, ls2, var, alpha, noise, noise_var)
        self.geo_fit_t = time() - t_f
        # print("self.vlp_gp_grd: ", self.vlp_gp_grd.shape)
        
        vlp_grd_oc, vlp_grd_var = self.gp_nav.model.predict_f(self.vlp_gp_grd)
        # print("check 4")
        self.vlp_gp_grd_var = vlp_grd_var.numpy()
        self.vlp_gp_grd_oc  = vlp_grd_oc.numpy()
        self.vlp_gp_grd_rds = self.oc_srfc_rds - self.vlp_gp_grd_oc
        print("check 5")
        self.gp_nav_grd_oc_pcl()
        # self.gp_nav_grd_var_pcl()

        # gpfs_thts       =  np.arange( -np.pi, np.pi, 0.25, dtype='float32')
        gpfs_thts       =  np.arange( -np.pi, np.pi, 0.1, dtype='float32')
        gpfs_als        =  np.arange( 1.8, 1.2, -0.1, dtype='float32')
        self.gpfs_sph      =  np.array(np.meshgrid(gpfs_thts, gpfs_als)).T.reshape(-1,2)
        t_p = time()
        gpfs_oc, gpfs_var = self.gp_nav.model.predict_f(self.gpfs_sph)
        self.geo_prdct_t = time() - t_p
        self.gpfs_var = gpfs_var.numpy()
        gpfs_oc  = gpfs_oc.numpy()
        gpfs_rds = self.oc_srfc_rds - gpfs_oc

        # print("gpfs_sph: ",  self.gpfs_sph.shape)
        # print("gpfs_sph: ",  self.gpfs_sph.T[0])
        # print("gpfs_var: ",  self.gpfs_var.shape)

        # self.var_km = 0.7
        # self.var_kv = 0.1
        self.gpfs_var_threshold()
        self.gpfs_var[self.gpfs_var < self.gpfs_var_thrshld] = 0
        self.gpfs_var[self.gpfs_var > self.gpfs_var_thrshld] = 1
        # print("aha: ", self.gpfs_var)
        # srtd_ids = np.argsort(self.gpfs_sph[:, 1])
        # self.gpfs_sph = self.gpfs_sph[srtd_ids]
        # self.gpfs_var = self.gpfs_var[srtd_ids]
        self.gpfs_var_pcl() ###as oc_var


        self.gp_nav_img_pcl_cb(nav_img_msg, cam_pcl_msg) # return self.gpfs_nav_cls and gp_seg_cls
        gpfs_sph = self.gpfs_sph.reshape(gpfs_thts.shape[0], gpfs_als.shape[0], 2)
        gpfs_var = self.gpfs_var.reshape(gpfs_thts.shape[0], gpfs_als.shape[0])
        # gpfs_cls = self.gpfs_nav_cls.reshape(gpfs_thts.shape[0], gpfs_als.shape[0])

        seg_gp_grd = self.cam_gp_grd.reshape(self.cam_gp_grd_w, self.cam_gp_grd_h, 2)
        seg_gp_cls = self.gp_seg_cls.reshape(self.cam_gp_grd_w, self.cam_gp_grd_h)
        # print("self.seg_gp_grd:  ", seg_gp_grd.shape, seg_gp_grd[0], seg_gp_grd[-1])
        # print("self.gp_seg_cls:  ", seg_gp_cls.shape, seg_gp_cls[0], seg_gp_cls[-1])

        # print("gpfs_sph: ",  gpfs_sph.shape )
        # print("gpfs_var: ",  gpfs_var.shape)

        self.lft_neg_th_dir_nav = np.sum(seg_gp_cls[-1])
        self.ryt_pos_th_dir_nav = np.sum(seg_gp_cls[0])
        print("self.self.lft_neg_th_dir_nav:  ", self.lft_neg_th_dir_nav)
        print("self.self.ryt_pos_th_dir_nav:  ", self.ryt_pos_th_dir_nav)

        ######## selects potential frontiers
        gpfs_als = []
        gpfs_ths = []
        gpfs_th_al = []
        gpfs_nvgblti = []
        gpfs_gmtry = []

        gpf_vis_cnt = 0
        for f in range(gpfs_var.shape[0]):
            gpf = gpfs_sph[f]
            gpf_v = gpfs_var[f]
            gpf_th = gpf[0][0]
            gpf_al = gpf[0][1]
            # print("aha gpf: ", gpf_th)
            # print("ah agpf_th: ", gpf_th)
            # print("ah agpf_al: ", gpf_al)

            if self.gmtry_nav_mode:  ## if no visual mode
                gpfs_nvgblti.append(38) # all considered visual navigable
                if np.array_equal(gpf_v, self.gpf_cond1)  or np.array_equal(gpf_v, self.gpf_cond3) :
                    gpfs_als.append(gpf[2][1])
                    gpfs_ths.append(gpf_th)
                    gpfs_th_al.append( [gpf[0][0], gpf[2][1]])
                    gpfs_gmtry.append(1)
                elif np.array_equal(gpf_v, self.gpf_cond2) or np.array_equal(gpf_v, self.gpf_cond4):
                    gpfs_als.append(gpf[3][1])
                    gpfs_ths.append(gpf_th)
                    gpfs_th_al.append( [gpf[0][0], gpf[3][1]])
                    gpfs_gmtry.append(1)
                else:
                    gpfs_als.append(np.pi)
                    gpfs_ths.append(gpf_th)
                    gpfs_th_al.append( [gpf[0][0], 0])
                    gpfs_gmtry.append(0)
           
            else: 
                #outside OF CAM FOV
                if gpf_th < -0.55 or gpf_th > 0.55: #-0.47794747, 0.47894406 OUTSIDE OF CAM FOV
                # if gpf_th < -0.66 or gpf_th > 0.66: #-0.47794747, 0.47894406 OUTSIDE OF CAM FOV
                    # print(" ======================== outside Fov ========================")
                    # print("gpf_th   : ", gpf_th)
                    # print("gpf_nvg   : ", 0)
                    gpfs_nvgblti.append(38) # outside fov considers half navigable
                    if np.array_equal(gpf_v, self.gpf_cond1)  or np.array_equal(gpf_v, self.gpf_cond3) :
                        gpfs_als.append(gpf[2][1])
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], gpf[2][1]])
                        gpfs_gmtry.append(1)
                    elif np.array_equal(gpf_v, self.gpf_cond2) or np.array_equal(gpf_v, self.gpf_cond4):
                        gpfs_als.append(gpf[3][1])
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], gpf[3][1]])
                        gpfs_gmtry.append(1)
                    else:
                        gpfs_als.append(np.pi)
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], 0])
                        gpfs_gmtry.append(0)

                #INSIDE OF CAM FOV
                else:  
                    print(" ======================== inside Fov ========================")
                    print("gpf_th   : ", gpf_th)
                    gpf_clmns = 4
                    strt, end =  int(gpf_clmns*gpf_vis_cnt), int(gpf_clmns*(gpf_vis_cnt+1))
                    end = min(end , self.cam_gp_grd_w)
                    print("strt, end: ", strt, end)
                    gpf_nvg=  np.sum(seg_gp_cls[strt:end]) /(end - strt)
                    gpf_vis_cnt += 1
                    print("gpf_nvg: ", gpf_nvg)
                    # if gpf_nvg < 18:
                        # gpf_nvg = 1
                    gpfs_nvgblti.append(gpf_nvg)
                    if np.array_equal(gpf_v, self.gpf_cond1)  or np.array_equal(gpf_v, self.gpf_cond3) :
                        gpfs_als.append(gpf[2][1])
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], gpf[2][1]])
                        gpfs_gmtry.append(1)
                    elif np.array_equal(gpf_v, self.gpf_cond2) or np.array_equal(gpf_v, self.gpf_cond4):
                        gpfs_als.append(gpf[3][1])
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], gpf[3][1]])
                        gpfs_gmtry.append(1)
                    else:
                        gpfs_als.append(np.pi)
                        gpfs_ths.append(gpf_th)
                        gpfs_th_al.append( [gpf[0][0], 0])
                        gpfs_gmtry.append(0)

        # print("all geo_gpfs gpfs_nvgblti: ", np.array(gpfs_nvgblti).shape)
        ###### find ids of geometry navigable gpfs
        gpfs_gmtry = np.array(gpfs_gmtry)
        gmtry_nav_gpfs_idx   = np.where( gpfs_gmtry == 1 )
        # print("gmtry_nav_gpfs_idx: ", gmtry_nav_gpfs_idx)
        self.gpfs_sph     = np.column_stack((gpfs_ths, gpfs_als)).reshape(-1,2)[gmtry_nav_gpfs_idx]
        self.gpfs_var     = np.array(gpfs_als).reshape(-1,1)[gmtry_nav_gpfs_idx]
        self.gpfs_gmtry   = np.array(gpfs_gmtry).reshape(-1,1)[gmtry_nav_gpfs_idx]
        self.gpfs_nvgblti = np.array(gpfs_nvgblti).reshape(-1,1)[gmtry_nav_gpfs_idx]
        # print("only geo-navigable_gpfs  self.gpfs_gmtry  : ", self.gpfs_gmtry.shape)
        # print("only geo-navigable_gpfs  self.gpfs_gmtry  : ", self.gpfs_gmtry)
        # print("only geo-navigable_gpfs  self.gpfs_nvgblti: ", self.gpfs_nvgblti)


        self.gpfs_nmbr = self.gpfs_var.shape[0]
        if self.gpfs_nmbr < 1:
            print("NO Frontiers found !!!")
            return 0
        # self.gpfs_var_pcl_2()

        self.gpfs_cost_func()
        # self.calibrate_fov()
        self.gpfs_vldyn_pcl()

        # global goal in polar coordinate wrt to velodyne to check its occupancy
        self.gl_wrt_rbt = self.tf_2d_inv @ self.gl_wrt_wrld
        # print("gl_wrt_wrld: ", self.gl_wrt_wrld )
        # print("gl_wrt_rbt: ", self.gl_wrt_rbt )

        # check occupancy at the global goal theta
        self.gl_th  = np.arctan2(self.gl_wrt_rbt[1], self.gl_wrt_rbt[0])
        self.gl_al  = np.pi/2
        gl_oc, gl_var = self.gp_nav.model.predict_f(np.array([self.gl_th, self.gl_al], dtype="float32").reshape(1,2) )
        gl_rds = self.oc_srfc_rds-gl_oc.numpy()
        # print("self.gl_th: ",self.gl_th )
        # print("gl_var, var_thrshld: ", gl_var.numpy(), self.gp_nav_var_thrshld)
        # print("gl_dst_err, gl_rds: ", self.gl_dst_err, gl_rds)        
       
        ### mode: if no obstacle betwen goal and robot, then go directly to goal 
        if(self.gl_dst_err < gl_rds ):
            self.glbl_gl_pbl() 
            print("\t\t******* goal Mode ********")
        else: ## goal to selected GPFrontier 
            print("\t\t******* subg Mode ********") 
            self.optm_gpf_cmd_pbl() 
        
        if self.log_data:
            file = open(self.time_file, "a")
            np.savetxt( file, np.array( [odom_msg.header.seq, odom_msg.header.stamp.to_sec(), self.geo_fit_t, self.geo_prdct_t, self.vis_fit_t, self.vis_prdct_t], dtype='float32').reshape(-1,6) , delimiter=" ", fmt='%1.3f')
            file.close()
        gpf_t = time()
        print("#### END Nav time: ", gpf_t- msg_rcvd_time, "\n\n\n")  



    def gpfs_cost_func(self):
        # print("gp_nav_pkup_nav_pt:: ")
        sb_gls_dsts = self.sb_gl_dst * np.ones(self.gpfs_nmbr, dtype='float32').reshape(-1,1)
        lcl_x, lcl_y, lcl_z = convert_spherical_to_cartesian(self.gpfs_sph.T[0].reshape(-1,1), self.gpfs_sph.T[1].reshape(-1,1), sb_gls_dsts)
        # lcl_z = np.zeros(self.gpfs_nmbr).reshape(-1,1) # not working for 2d TF r
        lcl_z = np.ones(self.gpfs_nmbr).reshape(-1,1) 
        self.sb_gls_xyz_vldyn = np.hstack((lcl_x, lcl_y, lcl_z))
        self.sb_gls_xyz_wrld = self.vldyn2wrld_sb_gls(self.sb_gls_xyz_vldyn)


        #### utlity function 
        sbgl2gl_dsts = np.sqrt( (self.gl_y - self.sb_gls_xyz_wrld.T[1])**2 + (self.gl_x - self.sb_gls_xyz_wrld.T[0])**2 )
        # sbgl2gl_dsts = self.normalize_array(sbgl2gl_dsts)
        # sbgl2rbt_dir = self.normalize_array(abs(self.gpfs_sph.T[0]) ) #**2)
        
        sbgl2gl_dsts = normalize_array(sbgl2gl_dsts)
        sbgl2rbt_dir = normalize_array(abs(self.gpfs_sph.T[0]) ) #**2)

        g_cost =  self.k_dst*(sbgl2gl_dsts) + self.k_dir*sbgl2rbt_dir
        # g_cost =  self.k_dst*(self.sb_gl_dst + sbgl2gl_dsts) + self.k_dir*sbgl2rbt_dir
        print("cost sbgl2gl_dsts, sbgl2rbt_dir, g_cost: ", sbgl2gl_dsts.shape, sbgl2rbt_dir.shape, g_cost.shape)

        if self.gmtry_nav_mode:
            print("~~~~~~~~ Geometry Mode ~~~~~~~~~")
            self.gpfs_cost = g_cost 
            self.gpfs_cost_vis =  self.gpfs_cost
        
        else:
            print("~~~~~~~~ Visual Mode ~~~~~~~~~")
            gpfs_nvgblti_fltn =  self.gpfs_nvgblti.flatten()
            out_fov_ids = gpfs_nvgblti_fltn == 38
            v_nvgble_ids = gpfs_nvgblti_fltn > self.v_nvgblti_th
            v_non_nvgble_ids = gpfs_nvgblti_fltn < self.v_nvgblti_th


            nvgblty_cost = np.zeros(gpfs_nvgblti_fltn.shape)
            nvgblty_cost[v_non_nvgble_ids] = 1
            nvgblty_cost[v_nvgble_ids]     = (1-self.k_nav)*g_cost[v_nvgble_ids]
            nvgblty_cost[out_fov_ids]      = self.k_nav*g_cost[out_fov_ids]  
            # print("nvgblty_cost: ", nvgblty_cost)

            # self.gpfs_cost_vis =  self.k_nav*nvgblty_cost + self.k_dst*(self.sb_gl_dst + sbgl2gl_dsts) + self.k_dir*sbgl2rbt_dir
            # self.gpfs_cost =  self.k_dst*(self.sb_gl_dst + sbgl2gl_dsts)
            self.gpfs_cost =  self.gpfs_cost_vis = nvgblty_cost

        # print("gap_dir: ", sbgl2rbt_dir)
        # print("sbgl2gl_dsts : ", sbgl2gl_dsts)
        # print("self.gpfs_cost_vis : ", self.gpfs_cost)

        # self.optm_gpf_idx = self.gpfs_cost.argmax(axis=0) ## based on area and direction
        self.optm_gpf_idx = self.gpfs_cost.argmin(axis=0) ## based on area and direction
        self.optm_gpf = self.gpfs_sph[self.optm_gpf_idx]
        # print("self.gpfs_cost: ", self.gpfs_cost.shape)
        # print("self.optm_gpf_idx: ", self.optm_gpf_idx)


    def vlp_pcl_cb(self,vlp_pcl_msg):
        strt_time = time()
        vlp_pcl_arr = ros_numpy.point_cloud2.pointcloud2_to_xyz_array(vlp_pcl_msg)
        # print("vlp_pcl_arr shape: ", vlp_pcl_arr.shape)

        vlp_x = vlp_pcl_arr.T[0]
        vlp_y = vlp_pcl_arr.T[1]
        vlp_z = vlp_pcl_arr.T[2]

        vlp_th, vlp_al, vlp_dst = convert_cartesian_to_spherical( vlp_x, vlp_y, vlp_z)
        self.vlp_xyz_pcl = np.row_stack( (vlp_x, vlp_y, vlp_z) )
        # self.vlp_sph_pcl = np.row_stack( (vlp_th, vlp_al, vlp_dst, self.oc_srfc_rds-vlp_dst ) )
        self.vlp_sph_pcl = np.round( np.row_stack( (vlp_th, vlp_al, vlp_dst, self.oc_srfc_rds-vlp_dst ) ), 4)
        
        self.vlp_thetas  = self.vlp_sph_pcl[0]
        self.vlp_alphas  = self.vlp_sph_pcl[1]
        self.vlp_rds     = self.vlp_sph_pcl[2]
        self.vlp_oc      = self.vlp_sph_pcl[3]
        self.vlp_sz      = self.vlp_sph_pcl.shape
        # print("vlp_xyz_pcl shape: ", self.vlp_xyz_pcl.shape)
        # print("vlp_sph_pcl shape: ", self.vlp_sph_pcl.shape)

        #### publish org_oc_srfc
        srfc_rds = self.org_oc_srfc_rds_viz* np.ones(vlp_th.shape)
        org_srfc_x, org_srfc_y, org_srfc_z = convert_spherical_to_cartesian( vlp_th, vlp_al, srfc_rds)
        org_srfc_arr = np.column_stack( (org_srfc_x, org_srfc_y, org_srfc_z, self.oc_srfc_rds-vlp_dst) )
        org_srfc_pcl = point_cloud2.create_cloud(vlp_pcl_msg.header, self.fields, org_srfc_arr)
        self.org_oc_srfc_pub.publish(org_srfc_pcl)
        print("Pcl_arr time: ", float(time()- strt_time))

    def odom_cb(self, odom_msg):
        self.pose_z = odom_msg.pose.pose.position.z
        self.pose_x = odom_msg.pose.pose.position.x
        self.pose_y = odom_msg.pose.pose.position.y
        quaternion = [
            odom_msg.pose.pose.orientation.x, odom_msg.pose.pose.orientation.y,
            odom_msg.pose.pose.orientation.z, odom_msg.pose.pose.orientation.w
        ]
        euler = euler_from_quaternion(quaternion)
        self.roll  = euler[0]
        self.pitch = euler[1]
        self.yaw   = euler[2]
        vel_x = odom_msg.twist.twist.linear.x
        vel_y = odom_msg.twist.twist.linear.y
        vel_z = odom_msg.twist.twist.linear.z
        vel_2d = np.sqrt(vel_x**2 + vel_y**2)
        ang_x = odom_msg.twist.twist.angular.x
        ang_y = odom_msg.twist.twist.angular.y
        ang_z = odom_msg.twist.twist.angular.z
        # print("current pose: ", self.pose_x, self.pose_y, self.pose_z, self.yaw)
        ### for saving data
        if self.log_data:
            file = open(self.odom_file, "a")
            np.savetxt( file, np.array( [odom_msg.header.seq, odom_msg.header.stamp.to_sec() , self.pose_x, self.pose_y, self.pose_z,
                                        self.roll, self.pitch, self.yaw, vel_x, vel_y, vel_z, vel_2d, ang_x, ang_y, ang_z], 
                                        dtype='float32').reshape(-1,15) , delimiter=" ", fmt='%1.3f')
            file.close()



    def rbt2gl_error(self):
        self.gl_dst_err = np.sqrt( (self.gl_x- self.pose_x)**2 + (self.gl_y- self.pose_y)**2)
        self.gl_dir = np.arctan2( self.gl_y- self.pose_y, self.gl_x- self.pose_x)
        # self.rbt_dir = self.pose.theta
        self.gl_dir_err = self.gl_dir - self.yaw
        # print("gl_dst_err, gl_dir_err: ", self.gl_dst_err, self.gl_dir_err )


    ### Form threshold based on varinace mean over variance surface
    def gp_nav_mask_thrshld(self):
        gp_nav_var_stats = stats.describe(self.vlp_gp_grd_var)
        gp_nav_var_mean = gp_nav_var_stats.mean[0]
        gp_nav_var_var  = gp_nav_var_stats.variance[0]
        self.gp_nav_var_thrshld = 0.4*(gp_nav_var_mean - 3*gp_nav_var_var) 
        print("gp_nav_var_mean, var: ", gp_nav_var_mean, gp_nav_var_var)
        print("self.gp_nav_var_thrshld: ", self.gp_nav_var_thrshld)


    def gpfs_var_threshold(self):
        gpfs_var_stats = stats.describe(self.vlp_gp_grd_var)
        gpfs_var_mean = gpfs_var_stats.mean[0]
        gpfs_var_var  = gpfs_var_stats.variance[0]
        self.gpfs_var_thrshld = self.var_km*gpfs_var_mean + self.var_kv*gpfs_var_var
        print("gpfs_var_mean, var: ", gpfs_var_mean, gpfs_var_var)
        print("self.gpfs_var_thrshld: ", self.gpfs_var_thrshld)


    #### publish coordinates of the selected gpfrontier (optm_gpf_idx) as next navigation sub-goal
    def optm_gpf_cmd_pbl(self):
        optm_gpf_xy_wrld = self.sb_gls_xyz_wrld[self.optm_gpf_idx].reshape(3,-1)
        yaw = np.arctan2(self.sb_gls_xyz_vldyn[self.optm_gpf_idx][1], self.sb_gls_xyz_vldyn[self.optm_gpf_idx][0])
        qtrn = quaternion_from_euler(0,0,yaw)
        # print("optm_gpf_xy_wrld: ", optm_gpf_xy_wrld)
        gl_msg = PoseStamped()
        gl_msg.pose.position.x =  optm_gpf_xy_wrld[0]
        gl_msg.pose.position.y =  optm_gpf_xy_wrld[1]
        gl_msg.pose.position.z =  0 #optm_gpf_xy_wrld[2]
        gl_msg.pose.orientation.x =   qtrn[0]
        gl_msg.pose.orientation.y =   qtrn[1]
        gl_msg.pose.orientation.z =   qtrn[2]
        gl_msg.pose.orientation.w =   qtrn[3]
        gl_msg.header =  self.wrld_hdr
        self.lcl_nav_cmd_pub.publish(gl_msg)
        # print("glbl_nav goal: ", optm_gpf_xy_wrld[0], optm_gpf_xy_wrld[1])
  

    ### publish the final goal coordinates as next navigation sub-goal 
    def glbl_gl_pbl(self):
        yaw = np.arctan2(self.gl_x, self.gl_y)
        qtrn = quaternion_from_euler(0,0,yaw)
        gl_msg = PoseStamped()
        gl_msg.pose.position.x =  self.gl_x
        gl_msg.pose.position.y =  self.gl_y
        gl_msg.pose.position.z =  0
        gl_msg.pose.orientation.x =   qtrn[0]
        gl_msg.pose.orientation.y =   qtrn[1]
        gl_msg.pose.orientation.z =   qtrn[2]
        gl_msg.pose.orientation.w =   qtrn[3]
        gl_msg.header =  self.wrld_hdr
        self.lcl_nav_cmd_pub.publish(gl_msg)
        # print("glbl_nav goal: ", nav_xy_gl[0], nav_xy_gl[1])
  
    ### send stop command 
    def stop_cmd(self):
        gl_msg = PoseStamped()
        gl_msg.pose.position.x =  self.pose_x
        gl_msg.pose.position.y =  self.pose_y
        gl_msg.pose.position.z =  0
        gl_msg.pose.orientation.x =   0
        gl_msg.pose.orientation.y =   0
        gl_msg.pose.orientation.z =   0
        gl_msg.pose.orientation.w =   1
        gl_msg.header =  self.wrld_hdr
        self.lcl_nav_cmd_pub.publish(gl_msg)

    def tf_rbt_2_wrld(self):
        self.tf_2d = np.array([ [np.cos(self.yaw), -np.sin(self.yaw), self.pose_x],
                                [np.sin(self.yaw),  np.cos(self.yaw), self.pose_y],
                                [0,0,1]   ])
        # print("tf_rbt_2_wrld: ", self.tf_2d)
          
    def tf_wrld_2_rbt(self):
        cos_th = np.cos(self.yaw)
        sin_th = np.sin(self.yaw)
        self.tf_2d_inv = np.array([ [cos_th,   sin_th, -self.pose_x*cos_th -self.pose_y*sin_th],
                                [-sin_th,  cos_th,  self.pose_x*sin_th -self.pose_y*cos_th],
                                [0,0,1]                                                          ])

    ##### to convert point coordinates from robot to world frame
    def vldyn2wrld_sb_gls(self, gls):
        xy_gls = np.empty([0, 3])
        for gl in gls:
            xy_gls = np.vstack( (xy_gls, self.tf_2d@gl) )
        # xy_gls = np.clip(xy_gls, -self.map_2d_h/2+1, self.map_2d_h/2-1 ) # limit to map  border,  say w=h
        # print("wrld sbgls: ", xy_gls)
        return xy_gls


    ################# visualization ##############
    def gpfs_var_pcl_2(self):
        rds = (self.gp_nav_var_viz-0.5)*np.ones(np.shape(self.gpfs_var)[0], dtype='float32').reshape(-1,1)
        x, y, z = convert_spherical_to_cartesian(self.gpfs_sph.T[:][0].reshape(-1,1), self.gpfs_sph.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.gpfs_var, dtype='float32').reshape(-1, 1)
        gpfs_var_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.vldyn_hdr, self.fields, gpfs_var_pcl)
        self.gp_var_srfc_pub.publish(pc2)

    def gpfs_var_pcl(self):
        rds = (self.gp_nav_var_viz-0.5)*np.ones(np.shape(self.gpfs_var)[0], dtype='float32').reshape(-1,1)
        x, y, z = convert_spherical_to_cartesian(self.gpfs_sph.T[:][0].reshape(-1,1), self.gpfs_sph.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.gpfs_var, dtype='float32').reshape(-1, 1)
        gpfs_var_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.vldyn_hdr, self.fields, gpfs_var_pcl)
        self.gp_var_srfc_pub.publish(pc2)

    def gp_nav_grd_var_pcl(self):
        rds = self.gp_nav_var_viz*np.ones(np.shape(self.vlp_gp_grd_var)[0], dtype='float32').reshape(-1,1)
        x, y, z = convert_spherical_to_cartesian(self.vlp_gp_grd.T[:][0].reshape(-1,1), self.vlp_gp_grd.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.vlp_gp_grd_var, dtype='float32').reshape(-1, 1)
        nav_grid_var_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.vldyn_hdr, self.fields, nav_grid_var_pcl)
        self.gp_var_srfc_pub.publish(pc2)

    def gp_nav_grd_oc_pcl(self):
        rds = self.gp_oc_srfc_rds_viz*np.ones(np.shape(self.vlp_gp_grd_rds)[0], dtype='float32').reshape(-1,1)
        x, y, z = convert_spherical_to_cartesian(self.vlp_gp_grd.T[:][0].reshape(-1,1), self.vlp_gp_grd.T[:][1].reshape(-1,1), rds)
        intensity = np.array( self.vlp_gp_grd_oc, dtype='float32').reshape(-1, 1)
        nav_grid_oc_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.vldyn_hdr, self.fields, nav_grid_oc_pcl)
        self.gp_oc_srfc_pub.publish(pc2)

    def gpfs_vldyn_pcl(self):
        print("gpfs_vldyn_pcl ..")
        rds = self.sb_gl_dst*np.ones(self.gpfs_nmbr, dtype='float32').reshape(-1,1)
        x, y, z = convert_spherical_to_cartesian(self.gpfs_sph.T[0].reshape(-1,1), self.gpfs_sph.T[1].reshape(-1,1), rds)
        intensity = np.array( self.gpfs_cost_vis, dtype='float32').reshape(-1, 1)
        nav_pts_pcl = np.column_stack( (x, y, z, intensity) )
        pc2 = point_cloud2.create_cloud(self.vldyn_hdr, self.fields, nav_pts_pcl)
        self.gpfs_bslnk_pub.publish(pc2)

    # def gp_nav_xypts_actul_pcl(self):
    #     # print(">> gp_nav_xypts_actul_pcl:: ")
    #     intensity = np.array( self.gpfs_cost, dtype='float32').reshape(-1, 1)
    #     # print("intensity: ", intensity)
    #     # print("self.sb_gls_xyz_wrld: ", self.sb_gls_xyz_wrld)
    #     nav_pts_pcl = np.column_stack( (self.sb_gls_xyz_wrld, intensity) )
    #     self.wrld_hdr.frame_id = "world"
    #     pc2 = point_cloud2.create_cloud(self.wrld_hdr, self.fields, nav_pts_pcl)
    #     self.gpfs_wrld_pub.publish(pc2)

    def publish_goal(self):
        # print(">> publish_goal:: ")
        gl_pcl = np.column_stack( (self.gl_x, self.gl_y, 0, 1) )
        self.wrld_hdr.frame_id = "world"
        pc2 = point_cloud2.create_cloud(self.wrld_hdr, self.fields, gl_pcl)
        self.trgt_gl_wrld_pub.publish(pc2)




    ################# sampling: down sample PCL, sample grid  ##############
    def downsample_pcl(self):  
        pcl_arr = self.vlp_sph_pcl[np.argsort(self.vlp_sph_pcl[0])]
        # pcl_arr = pcl_arr[np.argsort(pcl_arr[:, 0])] ## sort based on thetas
        thetas = pcl_arr.transpose()[:][0].reshape(-1,1)
        
        self.org_unq_thetas = np.array( sorted( set(thetas.flatten())) ) #.reshape(-1,1)
        #### percentage to keep  or fraction to delete 
        keep_th_ids = [ t for t in range(0, np.shape(self.org_unq_thetas)[0], self.pcl_skp)]    
        ids = []
        for t in keep_th_ids:
            ids = ids + list(np.where(thetas == self.org_unq_thetas[t] )[0])    
        # self.pcl_arr = np.delete(pcl_arr, ids, 0) # dimension along delete
        pcl_arr = pcl_arr[ids] 
        pcl_arr = pcl_arr.transpose()
        self.vlp_thetas = np.round(pcl_arr[:][0].reshape(-1,1), 4 )
        self.vlp_alphas = np.round(pcl_arr[:][1].reshape(-1,1), 4 )
        self.vlp_rds    = np.round(pcl_arr[:][2].reshape(-1,1), 4 )
        self.vlp_oc     = np.round(pcl_arr[:][3].reshape(-1,1), 4 )
        self.vlp_sz     = np.shape(self.vlp_thetas)[0]
        # print("downsampled size : ", np.shape(self.vlp_thetas)[0] )
        self.pcl_unq_thetas = np.array( sorted( set(self.vlp_thetas.flatten())) ) #.reshape(-1,1)
        self.unq_smpld_th_size = np.shape(self.pcl_unq_thetas)[0]
        # print("self.pcl_unq_thetas: ", self.unq_smpld_th_size )
        # quit()


    def vlp_gp_grid(self):
        ## for occupancy 
        # th_rsltion = 0.0174# 0.0174 #0.05 #0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        # al_rsltion = 0.00149#0.0249 #0.05 #0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)  
        ## for navigation
        th_rsltion = 0.1 #0.00174  # 0.02 # #from -pi to pi rad -> 0 35999
        al_rsltion = 0.05 #0.0349  #vpl16 resoltuion is 2 deg (from -15 to 15 deg)
        self.vlp_grd_ths = np.arange(- np.pi+0.0, np.pi+0.02, th_rsltion, dtype='float32')
        # self.vlp_gp_als = np.arange(np.pi/2-0.261799, np.pi/2+0.261799,   al_rsltion, dtype='float32')
        self.vlp_gp_als = np.arange(np.pi/2-0.161799, np.pi/2+0.201799,   al_rsltion, dtype='float32')
        self.vlp_gp_grd = np.array(np.meshgrid(self.vlp_grd_ths,self.vlp_gp_als)).T.reshape(-1,2)
        self.vlp_gp_grd_w = np.shape(self.vlp_grd_ths)[0] 
        self.vlp_gp_grd_h = np.shape(self.vlp_gp_als)[0]
        print("\vlp_gp_grd: ", np.shape(self.vlp_gp_grd))
        print("vlp_gp_grd w, h: ", self.vlp_gp_grd_w, self.vlp_gp_grd_h)
        # print("th_s: ", np.shape(self.vlp_grd_ths))
        # print("al_s: ", np.shape(self.vlp_gp_als))


if __name__ == "__main__":
    try:
        VGNav()
    except rospy.ROSInterruptException:
        pass
