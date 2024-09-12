#!/usr/bin/env python3

import rospy
import tf2_ros
import tf.transformations as tf_trans
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import TransformStamped, Quaternion
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
from time import time


def vlp_pcl_cb(vlp_pcl_msg):
    try:
        t1 = time()
        trans = tf_buffer.lookup_transform('world', 'velodyne', rospy.Time(0), rospy.Duration(1.0))
        q = [
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]

        euler_angles = tf_trans.euler_from_quaternion(q)
        roll = euler_angles[0]
        pitch = euler_angles[1]
        counter_rotation = tf_trans.quaternion_from_euler(-roll, -pitch, 0)

        counter_trans = TransformStamped()
        counter_trans.header.stamp = rospy.Time.now()
        counter_trans.header.frame_id = 'velodyne'
        counter_trans.child_frame_id = 'velodyne_horizontal'
        counter_trans.transform.rotation = Quaternion(*counter_rotation)

        broadcaster.sendTransform(counter_trans)
        trans1 = tf_buffer.lookup_transform('velodyne_horizontal', 'velodyne', rospy.Time(0), rospy.Duration(1.0))

        hrzntl_pcl = do_transform_cloud(vlp_pcl_msg, trans1)
        hrzntl_pcl.header = vlp_pcl_msg.header
        hrzntl_pcl.header.frame_id = 'velodyne_horizontal'

        pub.publish(hrzntl_pcl)
        print("t: ", time()- t1)
    except (tf2_ros.LookupException, tf2_ros.ExtrapolationException, tf2_ros.ConnectivityException, tf2_ros.TransformException) as ex:
        rospy.logerr(ex)



if __name__ == '__main__':
    rospy.init_node('velodyne_horizontal_node', anonymous=True)

    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    broadcaster = tf2_ros.TransformBroadcaster()

    pub = rospy.Publisher('horizontal/mid/points', PointCloud2, queue_size=10)
    # rospy.Subscriber("mid/points", PointCloud2, vlp_pcl_cb)
    rospy.Subscriber("velodyne_points", PointCloud2, vlp_pcl_cb) 

    rospy.spin()