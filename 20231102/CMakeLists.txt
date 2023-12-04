cmake_minimum_required(VERSION 3.0.0)

project(20231102)

# 设置编译器的位置
set (CMAKE_C_COMPILER "/usr/bin/gcc")
set (CMAKE_CXX_COMPILER "/usr/bin/g++")

#设置opencv
set (OpenCV_DIR "/home/keys/Env/Opencv/opencv/opencv-4.8.0/build/")
find_package(OpenCV)

set(CMAKE_PREFIX_PATH /home/keys/Env/Libtorch/libtorch/share/cmake/Torch/)
find_package(Torch REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


add_executable(test03 test03.cpp)

target_link_libraries(test03 ${OpenCV_LIBS})

target_link_libraries(test03 ${TORCH_LIBRARIES})
