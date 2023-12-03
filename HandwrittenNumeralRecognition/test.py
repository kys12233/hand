#进行测试
from dataset import dataset
from torchvision import transforms
import torch
import os
from utils import utils


test_transform = transforms.Compose(
    [transforms.ToTensor()]
)


test_dataset = dataset.MNIST_Dataset(
    label_path = r'F:\Vscode_python_programs\HandwrittenNumeralRecognition\data_to\test_label.txt',
    transform = test_transform
)


BatchSize = 16

pre_model = '98_0.011802.pth'



pre_result = 'pre_result'

# if not os.path.exists(pre_result):
#     os.makedirs(pre_result)

use_param = 'F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8'

use_param_name = use_param.split("\\")[-1]
if not os.path.exists('./'+pre_result + '/' + use_param_name):
    os.makedirs('./'+pre_result + '/' + use_param_name)


if __name__ == "__main__":
    test_loader = torch.utils.data.DataLoader(test_dataset,batch_size = BatchSize,shuffle = False)
    #加载pth文件（模型）
    net = torch.load(r'F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8' +'/' + pre_model)
    net.eval() #暂停模型

    #预测正确的个数（包括真预测成真，假预测成假）
    pre_True_number = 0

    list_a_b = []

    with torch.no_grad(): #目的是下面的程序不计算参数梯度，节省空间
        for image,label,img_name in test_loader:
            image = image.cuda()
            label = label.cuda()

            out = net(image)

            pre_num,label_max,out_max = utils.Compare(label,out)

            pre_True_number += pre_num

            for i in range(BatchSize):
                list_img_label_out = [img_name[i]]
                list_img_label_out.append(label_max[i].item())
                list_img_label_out.append(out_max[i].item())
            #将预测的结果进行保留
                list_a_b.append(list_img_label_out)
        with open('./'+pre_result + '/' + use_param_name + '/'+pre_model.replace('.pth','.txt'), 'w') as file_txt:
            for i in range(len(list_a_b)):
                
                file_txt.write(list_a_b[i][0] + ',')
                file_txt.write(str(list_a_b[i][1]) + ',')
                file_txt.write(str(list_a_b[i][2]) + '\n')

        pre_score = pre_True_number / len(test_dataset)
        with open('./'+pre_result + '/' + use_param_name + '/' + pre_model.replace('.pth','_pre_result.txt'), 'w') as file:
            file.write("预测正确的个数：" + str(pre_True_number) + '\n')
            file.write("所有数据个数：" + str(len(test_dataset)) + '\n')
            file.write("准确率为：{}%".format(pre_score * 100) + '\n')
        