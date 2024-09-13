#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# Modified from: https://github.com/facebookresearch/detectron2/blob/master/demo/demo.py

import argparse
import glob
import multiprocessing as mltproc
import os

import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
print("sys.path: ", sys.path)
import tempfile
import time
import warnings


import rospy
import cv2
import numpy as np
import tqdm
from sensor_msgs.msg import Image, CameraInfo #, PointCloud2, PointField
from cv_bridge import CvBridge
import ros_numpy
# import sensor_msgs.point_cloud2 as pc2

# ... (existing imports) ...
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from mask2former import add_maskformer2_config
from predictor import VisualizationDemo



class SegPredictor:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("mask2former_demo")
            # Set up detectron2 configuration
        mltproc.set_start_method("spawn", force=True)
        args = self.get_parser().parse_args()
        # setup_logger(name="fvcore")
        # logger = setup_logger()
        # logger.info("Arguments: " + str(args))
        cfg = self.setup_cfg(args)
        self.demo = VisualizationDemo(cfg)


  

        self.cam_img_sub = rospy.Subscriber("/sync_cam_img", Image, self.image_callback, queue_size=1) 
        self.seg_img_pub = seg_img_pub = rospy.Publisher("/seg_image", Image, queue_size=1)
        self.nav_img_pub = rospy.Publisher("/nav_image", Image, queue_size=1)
        
        rospy.spin()


    # ROS topic callback function to process the received image
    def image_callback(self, img_msg):
        strt_time = time.time()
        cv_bridge = CvBridge()
        cv_image = cv_bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")

        predictions, visualized_output = self.demo.run_on_image(cv_image)
        mask = predictions["sem_seg"].argmax(dim=0).to(self.demo.cpu_device).numpy().copy()
 
        # Define Navigable class   
        # nav_clss = np.array([3, 6, 11, 13, 29, 52, 94, 68,26]) 
        nav_clss = np.array([9]) 
        #grass 9, rock/stone 34, hill 68, mountain 16, mount 6, sand 46, sea 26


        # Create a binary mask
        for i in nav_clss:
           mask[mask == i] = 255

        mask[mask != 255] = 0
        nav_img = np.array(mask).astype(np.uint8)


        ####### Publish nav_img
        nav_img_msg = cv_bridge.cv2_to_imgmsg(nav_img, encoding="mono8")
        nav_img_msg.header = img_msg.header
        self.nav_img_pub.publish(nav_img_msg)
        # print("pub nav img time: ",  time.time() - strt_time)

        ####### Publish seg_img
        seg_img_msg = cv_bridge.cv2_to_imgmsg(visualized_output.get_image()[:, :, ::-1], encoding="bgr8")
        seg_img_msg.header = img_msg.header
        # print("header_stamp: ", seg_img_msg.header.stamp.to_sec())

        self.seg_img_pub.publish(seg_img_msg)
        print("pub nav n seg imgs time: ",  time.time() - strt_time, "\n\n\n")




    def setup_cfg(self, args):
        # load config from file and command-line arguments
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_maskformer2_config(cfg)
        cfg.merge_from_file(args.config_file)
        cfg.merge_from_list(args.opts)
        cfg.freeze()
        return cfg


    def get_parser(self):
        parser = argparse.ArgumentParser(description="maskformer2 demo for builtin configs")
        parser.add_argument(
            "--config-file",
            default="configs/coco/panoptic-segmentation/maskformer2_R50_bs16_50ep.yaml",
            metavar="FILE",
            help="path to config file",
        )
        parser.add_argument("--webcam", action="store_true", help="Take inputs from webcam.")
        parser.add_argument("--video-input", help="Path to video file.")
        parser.add_argument(
            "--input",
            nargs="+",
            help="A list of space separated input images; "
            "or a single glob pattern such as 'directory/*.jpg'",
        )
        parser.add_argument(
            "--output",
            help="A file or directory to save output visualizations. "
            "If not given, will show output in an OpenCV window.",
        )

        parser.add_argument(
            "--confidence-threshold",
            type=float,
            default=0.5,
            help="Minimum score for instance predictions to be shown",
        )
        parser.add_argument(
            "--opts",
            help="Modify config options using the command-line 'KEY VALUE' pairs",
            default=[],
            nargs=argparse.REMAINDER,
        )
        return parser


if __name__ == "__main__":
    seg_predictor = SegPredictor()
    # ... (existing setup_cfg and get_parser functions) ...


## used arg: --config-file ../configs/ade20k/semantic-segmentation/maskformer2_R50_bs16_160k.yaml  --opt MODEL.WEIGHTS ../checkpoint_files/ade20k/R50.pkl 
##classes
# 0:wall
# 1:building
# 2:sky
# 3:floor
# 4:tree
# 5:ceiling
# 6:road, route
# 7:bed
# 8:window
# 9:grass
# 10:cabinet
# 11:sidewalk, pavement
# 12:person
# 13:earth, ground
# 14:door
# 15:table
# 16:mountain, mount
# 17:plant
# 18:curtain
# 19:chair
# 20:car
# 21:water
# 22:painting, picture
# 23:sofa
# 24:shelf
# 25:house
# 26:sea
# 27:mirror
# 28:rug
# 29:field
# 30:armchair
# 31:seat
# 32:fence
# 33:desk
# 34:rock, stone
# 35:wardrobe, closet, press
# 36:lamp
# 37:tub
# 38:rail
# 39:cushion
# 40:base, pedestal, stand
# 41:box
# 42:column, pillar
# 43:signboard, sign
# 44:chest of drawers, chest, bureau, dresser
# 45:counter
# 46:sand
# 47:sink
# 48:skyscraper
# 49:fireplace
# 50:refrigerator, icebox
# 51:grandstand, covered stand
# 52:path
# 53:stairs
# 54:runway
# 55:case, display case, showcase, vitrine
# 56:pool table, billiard table, snooker table
# 57:pillow
# 58:screen door, screen
# 59:stairway, staircase
# 60:river
# 61:bridge, span
# 62:bookcase
# 63:blind, screen
# 64:coffee table
# 65:toilet, can, commode, crapper, pot, potty, stool, throne
# 66:flower
# 67:book
# 68:hill
# 69:bench
# 70:countertop
# 71:stove
# 72:palm, palm tree
# 73:kitchen island
# 74:computer
# 75:swivel chair
# 76:boat
# 77:bar
# 78:arcade machine
# 79:hovel, hut, hutch, shack, shanty
# 80:bus
# 81:towel
# 82:light
# 83:truck
# 84:tower
# 85:chandelier
# 86:awning, sunshade, sunblind
# 87:street lamp
# 88:booth
# 89:tv
# 90:plane
# 91:dirt track
# 92:clothes
# 93:pole
# 94:land, ground, soil
# 95:bannister, banister, balustrade, balusters, handrail
# 96:escalator, moving staircase, moving stairway
# 97:ottoman, pouf, pouffe, puff, hassock
# 98:bottle
# 99:buffet, counter, sideboard
# 100:poster, posting, placard, notice, bill, card
# 101:stage
# 102:van
# 103:ship
# 104:fountain
# 105:conveyer belt, conveyor belt, conveyer, conveyor, transporter
# 106:canopy
# 107:washer, automatic washer, washing machine
# 108:plaything, toy
# 109:pool
# 110:stool
# 111:barrel, cask
# 112:basket, handbasket
# 113:falls
# 114:tent
# 115:bag
# 116:minibike, motorbike
# 117:cradle
# 118:oven
# 119:ball
# 120:food, solid food
# 121:step, stair
# 122:tank, storage tank
# 123:trade name
# 124:microwave
# 125:pot
# 126:animal
# 127:bicycle
# 128:lake
# 129:dishwasher
# 130:screen
# 131:blanket, cover
# 132:sculpture
# 133:hood, exhaust hood
# 134:sconce
# 135:vase
# 136:traffic light
# 137:tray
# 138:trash can
# 139:fan
# 140:pier
# 141:crt screen
# 142:plate
# 143:monitor
# 144:bulletin board
# 145:shower
# 146:radiator
# 147:glass, drinking glass
# 148:clock
# 149:flag