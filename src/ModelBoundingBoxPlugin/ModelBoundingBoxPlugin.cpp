#include <gazebo/gazebo.hh>
#include <gazebo/physics/physics.hh>
#include <ignition/math/AxisAlignedBox.hh>
#include <ros/ros.h>
#include <geometry_msgs/Point.h>
#include <std_msgs/Float64.h>
#include <std_msgs/String.h>

namespace gazebo {
class ModelBoundingBoxPlugin : public ModelPlugin {
public:
    void Load(physics::ModelPtr _model, sdf::ElementPtr /*_sdf*/) {
        this->model = _model;
        this->rosNode = std::make_unique<ros::NodeHandle>("~");
        this->pub_area = this->rosNode->advertise<std_msgs::Float64>("/gazebo/" + model->GetName() + "/area", 10);
        this->pub_corners = this->rosNode->advertise<geometry_msgs::Point>("/gazebo/" + model->GetName() + "/corners", 40);
        this->pub_name = this->rosNode->advertise<std_msgs::String>("/gazebo/" + model->GetName() + "/name", 10);
        this->updateConnection = event::Events::ConnectWorldUpdateBegin(
            std::bind(&ModelBoundingBoxPlugin::OnUpdate, this));
    }

    void OnUpdate() {
        // 获取边界框
        ignition::math::AxisAlignedBox bbox = this->model->BoundingBox();
        ignition::math::Vector3d min = bbox.Min();
        ignition::math::Vector3d max = bbox.Max();

        // 计算面积 (XY 平面投影)
        double area = (max.X() - min.X()) * (max.Y() - min.Y());

        // 发布物体名称
        std_msgs::String name_msg;
        name_msg.data = this->model->GetName();
        this->pub_name.publish(name_msg);

        // 发布面积
        std_msgs::Float64 area_msg;
        area_msg.data = area;
        this->pub_area.publish(area_msg);

        // 发布四角坐标 (2D XY 平面) 和 3D 角点
        std::vector<geometry_msgs::Point> corners = {
            geometry_msgs::Point(), geometry_msgs::Point(), geometry_msgs::Point(), geometry_msgs::Point(),
            geometry_msgs::Point(), geometry_msgs::Point(), geometry_msgs::Point(), geometry_msgs::Point()
        };
        // 2D 四角 (XY 平面)
        corners[0].x = min.X(); corners[0].y = min.Y(); corners[0].z = 0.0; // 角1
        corners[1].x = min.X(); corners[1].y = max.Y(); corners[1].z = 0.0; // 角2
        corners[2].x = max.X(); corners[2].y = min.Y(); corners[2].z = 0.0; // 角3
        corners[3].x = max.X(); corners[3].y = max.Y(); corners[3].z = 0.0; // 角4
        // 3D 角点
        corners[4].x = min.X(); corners[4].y = min.Y(); corners[4].z = min.Z(); // 角5
        corners[5].x = min.X(); corners[5].y = max.Y(); corners[5].z = min.Z(); // 角6
        corners[6].x = max.X(); corners[6].y = min.Y(); corners[6].z = min.Z(); // 角7
        corners[7].x = max.X(); corners[7].y = max.Y(); corners[7].z = max.Z(); // 角8

        for (const auto& corner : corners) {
            this->pub_corners.publish(corner);
        }
    }

private:
    physics::ModelPtr model;
    std::unique_ptr<ros::NodeHandle> rosNode;
    ros::Publisher pub_area;
    ros::Publisher pub_corners;
    ros::Publisher pub_name;
    event::ConnectionPtr updateConnection;
};

GZ_REGISTER_MODEL_PLUGIN(ModelBoundingBoxPlugin)
}