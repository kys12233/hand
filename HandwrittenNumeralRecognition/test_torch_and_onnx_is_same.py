import torch.onnx
import torchvision
import torch
import cv2
import onnx
import onnxruntime
import numpy as np
import Mobilenetv2_025


model = Mobilenetv2_025.mobilenetv2()
device = torch.device("cpu")
dummy_input = torch.randn(1, 3, 224, 224).to(device)
model = torch.load('F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8\98_0.011802.pth', map_location=device)
model.eval()

out = model(dummy_input)
print(out[0][:10])


onnx_model = onnx.load('mobilenet_025_size_3_224_224_98_0.011802.onnx')  # load onnx model
session = onnxruntime.InferenceSession("mobilenet_025_size_3_224_224_98_0.011802.onnx", None)
input_name = session.get_inputs()[0].name
orig_result = session.run([], {input_name: dummy_input.data.numpy()})
print(orig_result[:10])


