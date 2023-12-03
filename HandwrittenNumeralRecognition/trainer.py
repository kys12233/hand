from dataset import dataset
# import Mobilenetv2
import Mobilenetv2_025
import torch
from torch.nn import MSELoss #引入均方差损失函数
from torchvision import transforms
import numpy as np
import os


train_transform = transforms.Compose([
        transforms.ToTensor() #将数据从hwc转化为chw，并放到0到1之间
    ])

train_dataset = dataset.MNIST_Dataset(
        label_path=r'F:\Vscode_python_programs\HandwrittenNumeralRecognition\data_to\train_label.txt',
        transform=train_transform,
)

mseloss = MSELoss()

save_path = 'F:\Vscode_python_programs\HandwrittenNumeralRecognition\save_models\param_2023_0517_Mobilenetv2_025_batchsize_8_state_dict'

if not os.path.exists(save_path):
    os.makedirs(save_path)
#利用一个参数判断是否需要加载预训练模型
pre_train = False

pre_model = '43_0.035513.pth'

if pre_train == False:
    net = Mobilenetv2_025.mobilenetv2().cuda() #模型实例化
    min_loss = 100 #初始化假设最小损失为100
    start_epoch = 0
else:
    net = torch.load(save_path +'/' + pre_model, map_location=lambda storage, loc: storage)
    net = net.cuda()
    min_loss = np.float32(pre_model.split('.pth')[0].split('_')[1])
    start_epoch = int(pre_model.split('.pth')[0].split('_')[0]) +1

# print(start_epoch)
# print(min_loss)
# exit()


BatchSize = 8


if __name__ == '__main__':

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BatchSize, shuffle=True)
    
    net.train()

    opt = torch.optim.Adam(net.parameters()) #定义优化器

    for epoch in range(start_epoch,10):
        sum_loss = 0 #每一轮的初始总损失是0
        i = 0
        for image, label, img_name in train_loader:
            
            image_cuda = image.cuda() 
            label_cuda = label.cuda()
            
            out = net(image_cuda)
            loss = mseloss(out, label_cuda)

            if i % 100 == 0:
                print("已经进行到了第{}轮的第{}批次".format(epoch,i))
            i += 1
            sum_loss += loss.item()

            opt.zero_grad()
            loss.backward()
            opt.step()
                # torch.save(net.state_dict(), 'param04_26_16_17/{}.pt'.format(epoch))
        avg_loss = sum_loss / len(train_dataset) * BatchSize * 100#损失乘以100，好保存数据
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(net.state_dict(), save_path + '\{}_{}.pth'.format(epoch,round(avg_loss,6)))
            print('epoch是：', epoch,'平均损失：', avg_loss)