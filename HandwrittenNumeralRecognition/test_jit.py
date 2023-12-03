# 运行jit格式


import time
import torch
import cv2 as cv
from Mobilenetv2_025 import MobileNetV2
import matplotlib.pyplot as plt
# import seaborn as sns
# import pandas as pd
from torchvision import transforms

def run_model(model, image):
    s = time.time()
    out = model(image)
    pre_lab = torch.argmax(out, dim=1)
    cost_time = round(time.time() - s, 5)
    return cost_time

test_transform = transforms.Compose(
    [transforms.ToTensor()]
)

image = cv.imread("F:/Data_Set/HandwrittenNumeralRecognition/mnist_test/0/10.png")
#读取后进行预处理
image = cv.resize(image, (224, 224))
image = test_transform(image)
# print(image.shape)
# print(type(image))
image = torch.FloatTensor(image).unsqueeze(0)
# print(image.shape)
# print(type(image))
# exit()
# image = torch.FloatTensor(image).unsqueeze(0).unsqueeze(0).contiguous()

origin_model = MobileNetV2()
origin_model.load_state_dict(torch.load("F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8_state_dict\9_0.103646.pth"))
jit_model = torch.jit.load("mobilenet_025_size_3_224_224_9_0.103646.jit")

# init jit
for _ in range(100):
    x1 = run_model(origin_model, image)
    print("x1的值是：",x1)
    x2 = run_model(jit_model, image)
    print("x2的值是：",x2)

test_times = 10

# begin testing
# results = pd.DataFrame({
#     "type" : ["orgin"] * test_times + ["jit"] * test_times,
#     "cost_time" : [run_model(origin_model, image) for _ in range(test_times)] + [run_model(jit_model, image) for _ in range(test_times)]
# })

# plt.figure(dpi=120)
# sns.boxplot(
#     x=results["type"],
#     y=results["cost_time"]
# )
# plt.show()