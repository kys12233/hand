# 将模型进行转换，实现在ubuntu下的libtorch的推理
# 将，将模型从pth格式转化为jit格式

import torch
from Mobilenetv2_025 import MobileNetV2

model = MobileNetV2()
model.load_state_dict(torch.load("F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8_state_dict\9_0.103646.pth", map_location="cpu"))

sample = torch.randn(1, 3, 224, 224)

# PyTorch通过JIT搭建了Python和C++的桥梁，
# 我们可以将模型转成TorchScript Module，将Python运行时的部分运行时包裹进去。

trace_model = torch.jit.trace(model, sample)
trace_model.save("mobilenet_025_size_3_224_224_9_0.103646.jit")