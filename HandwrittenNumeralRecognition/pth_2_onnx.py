import torch
import torchvision


dummy_input = torch.randn(1,3,224,224,device='cuda')
model_path = 'F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8\98_0.011802.pth'
model = torch.load(model_path)


# 给输入输出取个名字
input_name = ["input_1"]
out_name = ["output_1"]

onnx_save = model_path.split("\\")[-1].replace(".pth",".onnx") #replace用后面的参数替换前面的参数
print(onnx_save)
# exit()

save_name = 'mobilenet_025_size_3_224_224_'

torch.onnx.export(model,dummy_input,save_name + onnx_save, verbose=True, input_names=input_name, output_names=out_name)


