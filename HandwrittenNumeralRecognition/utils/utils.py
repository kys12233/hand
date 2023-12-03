#进行准确率的比对。

import torch

def Compare(a,b): #传入两个torch.tensor
    # print(a.shape)
    # exit()
    a_max = torch.argmax(a,1)
    b_max = torch.argmax(b,1)
    a_b_eq = a_max.eq(b_max) #判断a_max和b_max的各个元素是否相等 
    
    number = torch.sum(a_b_eq == True).item()
    return number,a_max,b_max