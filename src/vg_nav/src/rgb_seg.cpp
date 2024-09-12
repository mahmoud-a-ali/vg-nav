#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <dynamic_reconfigure/server.h>
#include <vg_nav/ThresholdConfig.h>
#include <std_msgs/Header.h>

ros::Publisher grey_pub;
cv_bridge::CvImagePtr cv_ptr;
std_msgs::Header img_header;

double min_threshold_red = 0.0;
double max_threshold_red = 255.0;

double min_threshold_green = 148.0;
double max_threshold_green = 255.0;

double min_threshold_blue = 45.0;
double max_threshold_blue = 195.0;

void thresholdImage()
{
    if (!cv_ptr)
        return;

    cv::Mat image = cv_ptr->image;

    // Threshold each channel
    cv::Mat thresholded_image;
    cv::inRange(image, cv::Scalar(min_threshold_blue, min_threshold_green, min_threshold_red),
                cv::Scalar(max_threshold_blue, max_threshold_green, max_threshold_red), thresholded_image);


// Apply morphological operations (dilation and erosion) for smoothing
    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(4,4));
    cv::morphologyEx(thresholded_image, thresholded_image, cv::MORPH_CLOSE, kernel);



    // Publish the thresholded image
    grey_pub.publish(cv_bridge::CvImage(img_header, "mono8", thresholded_image).toImageMsg());
}



void callback(vg_gpn::ThresholdConfig &config, uint32_t level)
{
    min_threshold_red = config.min_threshold_red;
    max_threshold_red = config.max_threshold_red;
    min_threshold_green = config.min_threshold_green;
    max_threshold_green = config.max_threshold_green;
    min_threshold_blue = config.min_threshold_blue;
    max_threshold_blue = config.max_threshold_blue;
    thresholdImage();
}



void imageCallback(const sensor_msgs::ImageConstPtr &msg)
{
    try
    {
        // Convert ROS image message to cv::Mat
        img_header = msg->header;
        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        thresholdImage();
    }
    catch (cv_bridge::Exception &e)
    {
        ROS_ERROR("cv_bridge exception: %s", e.what());
    }
}

int main(int argc, char **argv)
{
    ros::init(argc, argv, "rgb_mask_grey");
    ros::NodeHandle nh;

    // Subscribe to the sync_img topic
    ros::Subscriber sub = nh.subscribe("sync_cam_img", 1, imageCallback);

    // Initialize the publisher for the grey_img topic
    grey_pub = nh.advertise<sensor_msgs::Image>("nav_image", 1);

    // Dynamic Reconfigure server
    dynamic_reconfigure::Server<vg_gpn::ThresholdConfig> server;
    dynamic_reconfigure::Server<vg_gpn::ThresholdConfig>::CallbackType f;
    f = boost::bind(&callback, _1, _2);
    server.setCallback(f);

    ros::spin();
    return 0;
}
