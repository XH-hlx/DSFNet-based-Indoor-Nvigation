cmake_minimum_required(VERSION 3.5)
project(gazebo_plugins)

find_package(gazebo REQUIRED)
find_package(roscpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

include_directories(${GAZEBO_INCLUDE_DIRS} ${roscpp_INCLUDE_DIRS} ${std_msgs_INCLUDE_DIRS} ${geometry_msgs_INCLUDE_DIRS})
link_directories(${GAZEBO_LIBRARY_DIRS})

add_library(ModelBoundingBoxPlugin SHARED ModelBoundingBoxPlugin.cpp)
target_link_libraries(ModelBoundingBoxPlugin ${GAZEBO_LIBRARIES} ${roscpp_LIBRARIES} ${std_msgs_LIBRARIES} ${geometry_msgs_LIBRARIES})