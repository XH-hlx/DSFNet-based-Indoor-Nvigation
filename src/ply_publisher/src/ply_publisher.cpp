#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Geometry>

int main(int argc, char** argv) {
    ros::init(argc, argv, "ply_publisher");
    ros::NodeHandle nh;
    ros::Publisher pub = nh.advertise<sensor_msgs::PointCloud2>("/point_cloud", 1);
    ros::Rate rate(10); // 发布频率 10Hz

    // 加载 PLY 文件
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    std::string ply_path = "/mnt/xh/ply2/label_NYU0582_0000.ply";
    if (pcl::io::loadPLYFile<pcl::PointXYZRGB>(ply_path, *cloud) == -1) {
        ROS_ERROR("Couldn't read PLY file: %s", ply_path.c_str());
        return -1;
    }
    ROS_INFO("Loaded PLY file with %d points", (int)cloud->size());

    // 定义缩放因子
    float scale_factor = 0.068; // 缩小 2 倍，增强密度

    // 创建缩放矩阵
    Eigen::Matrix4f scale_matrix = Eigen::Matrix4f::Identity();
    scale_matrix(0, 0) = scale_factor; // X 轴缩放
    scale_matrix(1, 1) = scale_factor; // Y 轴缩放
    scale_matrix(2, 2) = scale_factor; // Z 轴缩放

    // 创建绕 X 轴旋转 90 度的矩阵（逆时针）
    Eigen::Matrix4f rotation_x_matrix = Eigen::Matrix4f::Identity();
    float angle_x = M_PI / 2.0; // 90 度
    rotation_x_matrix(1, 1) = cos(angle_x);
    rotation_x_matrix(1, 2) = -sin(angle_x);
    rotation_x_matrix(2, 1) = sin(angle_x);
    rotation_x_matrix(2, 2) = cos(angle_x);

    // 创建绕 Z 轴顺时针旋转 90 度的矩阵
    Eigen::Matrix4f rotation_z_matrix = Eigen::Matrix4f::Identity();
    float angle_z = -M_PI / 3.0; // 顺时针 90 度
    rotation_z_matrix(0, 0) = cos(angle_z);
    rotation_z_matrix(0, 1) = -sin(angle_z);
    rotation_z_matrix(1, 0) = sin(angle_z);
    rotation_z_matrix(1, 1) = cos(angle_z);

    // 创建平移矩阵
    Eigen::Matrix4f translation_matrix = Eigen::Matrix4f::Identity();
    translation_matrix(0, 3) = 1.26; // X 轴平移 -2 米
    translation_matrix(1, 3) = 2.2; // Y 轴平移 5 米
    translation_matrix(2, 3) = -0.124; // Z 轴平移 -0.6 米


    // 组合变换矩阵（先缩放，后绕 X 轴旋转，再绕 Z 轴旋转，最后平移）
    Eigen::Matrix4f transform_matrix = translation_matrix * rotation_z_matrix * rotation_x_matrix * scale_matrix;

    // 应用变换
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    pcl::transformPointCloud(*cloud, *transformed_cloud, transform_matrix);

    // 滤除地板（假设地板在 Z=0 附近）
    pcl::PassThrough<pcl::PointXYZRGB> pass;
    pass.setInputCloud(transformed_cloud);
    pass.setFilterFieldName("z");
    pass.setFilterLimits(0.1, 10.0); // 保留 Z > 0.1m 的点，移除地板
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_no_floor(new pcl::PointCloud<pcl::PointXYZRGB>);
    pass.filter(*cloud_no_floor);

    // 体素滤波，增加点云均匀性
    pcl::VoxelGrid<pcl::PointXYZRGB> sor;
    sor.setInputCloud(cloud_no_floor);
    sor.setLeafSize(0.05f, 0.05f, 0.05f); // 体素大小 5cm
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZRGB>);
    sor.filter(*cloud_filtered);

    // 滤除机器狗自身点云
    pcl::CropBox<pcl::PointXYZRGB> box;
    box.setMin(Eigen::Vector4f(-0.5, -0.5, -0.5, 1)); // 机器狗周围 0.5m
    box.setMax(Eigen::Vector4f(0.5, 0.5, 0.5, 1));
    box.setInputCloud(cloud_filtered);
    box.setNegative(true); // 移除框内点
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cropped(new pcl::PointCloud<pcl::PointXYZRGB>);
    box.filter(*cloud_cropped);

    // 转换为 ROS 消息
    sensor_msgs::PointCloud2 cloud_msg;
    pcl::toROSMsg(*cloud_cropped, cloud_msg);
    cloud_msg.header.frame_id = "odom"; // 保持 world 坐标系
    cloud_msg.header.stamp = ros::Time::now();

    while (ros::ok()) {
        cloud_msg.header.stamp = ros::Time::now();
        pub.publish(cloud_msg);
        ROS_INFO("Published transformed point cloud to /point_cloud with %d points", (int)cloud_cropped->size());
        rate.sleep();
    }

    return 0;
}