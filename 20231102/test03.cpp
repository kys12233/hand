#include "torch/script.h"
#include "torch/torch.h"

// opencv 头文件
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <memory>

// 本代码来源于：https://zhuanlan.zhihu.com/p/141401062

int main()
{
    // 如果是1.1版本及以下:
    // std::shared_ptr<torch::jit::script::Module> module = torch::jit::load(argv[1]);

    // 如果是1.2版本及以上:
    torch::jit::script::Module module; //Torch Script中的核心数据结构是ScriptModule。
    // 它是Torch的nn.Module的类似物，代表整个模型作为子模块树。 
    //与普通模块一样，ScriptModule中的每个单独模块都可以包含子模块，参数和方法。 
    //在nn.Modules中，方法是作为Python函数实现的，但在ScriptModules方法中通常实现为Torch Script函数，
    //这是一个静态类型的Python子集，包含PyTorch的所有内置Tensor操作。 
    //这种差异允许您运行ScriptModules代码而无需Python解释器。

    try {
        module = torch::jit::load("/home/keys/Project/PythonProjects/HandwrittenNumeralRecognition/mobilenet_025_size_3_224_224_9_0.103646.jit"); //加载jit格式的模型
        std::cout << "model load successful" << std::endl;
    }
    catch (const c10::Error& ) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    // 之前的版本 module->to , 1.4版本以上用 module.to();
    //module.to(at::kCUDA); // 将模型加载到cuda

    cv::Mat input_image; // 初始化变量，c或者c++代码使用变量之前要先初始化
    
    cv::Mat read_image = cv::imread("/home/keys/Project/PythonProjects/HandwrittenNumeralRecognition/DataSet/mnist_test/0/10.png"); // 初始化变量并读取
    
    if (read_image.empty() || !read_image.data) // 如果读取的图片为空或者没有读取到图片
        std::cout << "read image fail" << std::endl;
    
    // resize(224)
    cv::Size scale(224, 224);
    cv::resize(read_image, input_image, scale, 0, 0, cv::INTER_LINEAR); // 参数1是原图，参数2是目标图
    
    // 转换 [unsigned int] to [float]，实现transforms。
    input_image.convertTo(input_image, CV_32FC3, 1.0 / 255.0); // CV_32FC3表示32位的float，也就是单
    // 精度浮点数，C3表示3通道。
    // torch::from_blob的第一个参数就是：数组指针，可以是数组指针类型或者是类似于数组指针的首地址。
    // 第二个参数：Tensor的大小，如果前面设置的为数组的首地址，则索引数组创建Tensor
    // torch::Tensor x：表示创建torch::Tensor对象x
    torch::Tensor tensor_image = torch::from_blob(input_image.data, {1, input_image.rows, input_image.cols,3});
    tensor_image = tensor_image.permute({0,3,1,2}); // 图片的形状变化，将hwc转换为chw，0不动表示batchsize。

    // 实现Normalize
    // transforms.Normalize(mean=[0.485, 0.456, 0.406],
    //                    std=[0.229, 0.224, 0.225])
    // tensor_image[0][0] = tensor_image[0][0].sub_(0.485).div_(0.229);
    // tensor_image[0][1] = tensor_image[0][1].sub_(0.456).div_(0.224);
    // tensor_image[0][2] = tensor_image[0][2].sub_(0.406).div_(0.225);
    //tensor_image = tensor_image.to(torch::kCUDA);
    torch::Tensor output = module.forward({tensor_image}).toTensor(); //创建torch::Tensor对象
    // output来做 forward向前计算
    std::cout << output << std::endl; //打印结果
    std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl; //打印出来
    
    return 0;
}