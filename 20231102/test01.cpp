#include  <iostream>
#include  <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <torch/script.h>

int main() 
{
    std::cout << "Hello, World!" << std::endl;
     
    // cv::Mat image = cv::imread("./001.jpg"); // 读取图像 

    // cv::imshow("Image", image); // 显示图像 
    // cv::waitKey(0); // 等待按键 

    // cv::Mat new_img;
    // cv::cvtColor(image, new_img, cv::COLOR_BGR2RGB);
    // cv::imwrite("./000001.jpg",image);
    return 0;
}
