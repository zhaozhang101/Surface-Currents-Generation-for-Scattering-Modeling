import torch.utils.data as data
import numpy as np
from PIL import Image
import os
import os.path
from io import BytesIO
import torch
from torch.utils import data
from torchvision import transforms, utils
import math
import os
import matplotlib.pyplot as plt
from scipy import io
import numpy
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator
import argparse
from torch import optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.optim import lr_scheduler
import warnings
# from modelSelf import A_Decoder, A_Encoder, A_DiscriminatorInput, A_DiscriminatorOutput, A_Generator, SSIM
from lossfun import *
from tools import *

DT_EXTENSIONS = ['.npy']

def loadcurt(file,size):
    data = numpy.loadtxt(fname=file,dtype=float,skiprows=13)
    x_y = (data[:,1:3]*1000+17.5)*size/35
    zlist = data[:,3]
    Ix_rel_list = data[:,4]
    Ix_img_list = data[:,5]
    Iy_rel_list = data[:,6]
    Iy_img_list = data[:,7]
    Iz_rel_list = data[:,8]
    Iz_img_list = data[:,9]
    X = np.linspace(0, size-1, size)
    Y = np.linspace(0, size-1, size)
    X, Y = np.meshgrid(X, Y)

    current = np.zeros(shape=(7,size,size))
    interp = NearestNDInterpolator(x_y, Ix_rel_list)
    Ix_rel_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, Ix_img_list)
    Ix_img_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, Iy_rel_list)
    Iy_rel_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, Iy_img_list)
    Iy_img_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, Iz_rel_list)
    Iz_rel_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, Iz_img_list)
    Iz_img_mat = interp(X, Y)*1000
    interp = NearestNDInterpolator(x_y, zlist)
    Z_mat = interp(X, Y)

    current[0] = Ix_rel_mat
    current[1] = Ix_img_mat
    current[2] = Iy_rel_mat
    current[3] = Iy_img_mat
    current[4] = Iz_rel_mat
    current[5] = Iz_img_mat
    current[6] = Z_mat
    return torch.tensor(current,dtype=torch.float32).unsqueeze(0)

def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def make_dataset(surface_dir, current_dir, extensions):
    arr = []
    for curr_root, _, fnames in sorted(os.walk(current_dir)):
        surf_root = os.path.join(os.path.split(curr_root)[0],'surface')
        mask_root = os.path.join(os.path.split(os.path.split(curr_root)[0])[0],'PyCode/mask')
        for fname in sorted(fnames):
            if has_file_allowed_extension(fname, extensions) and os.path.exists(os.path.join(curr_root,fname)):
                surfname = fname.split('_',4)[0:3]
                surfname = surfname[0]+'_'+surfname[1]+'_'+surfname[2]+'.npy'
                path_surf = os.path.join(surf_root, surfname)
                path_curr = os.path.join(curr_root, fname)
                info = fname[2:-4]
                parts = info.split("_")
                path_mask = os.path.join(mask_root, 'mask_azi'+str(int(parts[3]))+'_zen'+str(int(parts[4]))+'.npy')

                if len(parts)==5:
                    rela_length = float(parts[1])
                    rms_height = float(parts[2])
                    inci_azi = float(parts[3])
                    inci_zen = float(parts[4])
                    rela_length = 64000 / rela_length ** 3   # (已弃用)
                    rms_height = rms_height ** 2 / 6400    # （已弃用）
                    inci_azi = inci_azi / 45              # (重映射：0° 映射为0 ；90° 映射为 1)
                    inci_zen = inci_zen / 10               # (重映射：0 映射为0 ； 10 映射为 1)
                else:
                    raise Exception("File Name Error")

                item = (rela_length ,rms_height, inci_azi, inci_zen, path_surf, path_curr,path_mask)
                # 相当于 编码和目标
                arr.append(item)
    return arr

def npy_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    data = np.load(path)
    return data

def default_loader(path):
    return npy_loader(path)

def transform_train(curt, surf):
    # randseed1,randseed2,randseed3 = random.random(),random.random(),random.random()
    # if randseed1 > 0.5:
    #     curt = F.hflip(curt)
    #     surf = F.hflip(surf)
    # if randseed2 > 0.5:
    #     curt = F.vflip(curt)
    #     surf = F.vflip(surf)
    # curt = F.rotate(curt,round(randseed3*4)*90)
    # surf = F.rotate(surf,round(randseed3*4)*90)
    return curt, surf

def one_hot_encode(number, num_classes):
    encoding = np.zeros(num_classes)  # 创建一个全零的向量
    encoding[number] = 1  # 将指定位置设置为1
    return encoding

def one_hot_encode_7(number, num_classes=7): # 俯仰角
    # number = 1
    encoding = np.zeros(num_classes)
    encoding[number] = 1
    # low,hig = int(number),int(number)+1
    # encoding[low]=hig-number
    # encoding[min(hig,6)]=number-low
    return encoding

def one_hot_encode_8(number, num_classes=8):
    # number=1
    encoding = np.zeros(num_classes)
    encoding[number] = 1
    # low, hig = int(number), int(number) + 1
    # encoding[low] = hig - number
    # encoding[hig] = number - low
    return encoding

def inverse_one_hot_code(tensor):
    tensor=tensor.squeeze()
    indices = np.where(tensor == 1)[0]*10
    if indices ==0:
        return indices
    else:
        return indices-5
def inverse_one_hot_code_8(tensor):
    tensor=tensor.squeeze()
    indices = np.where(tensor == 1)[0]*45
    return indices
def inverse_one_hot_code_7(tensor):
    tensor=tensor.squeeze()
    indices = np.where(tensor == 1)[0]*10+10
    return indices

class DatasetFolder(data.Dataset):
    def __init__(self, root, transform1=None, transform2=None, loader=default_loader, extensions=DT_EXTENSIONS):
        self.root = root
        self.curt = os.path.join(self.root,'current')
        self.surf = os.path.join(self.root, 'surface')
        self.loader = loader
        self.extensions = extensions
        samples = make_dataset(self.surf, self.curt, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.surf + "\n"
                               "Supported extensions are: " + ",".join(extensions)))
        self.samples = samples
        self.transform1 = transform1
        self.transform2 = transform2

    def __getitem__(self, index):

        rela_length ,rms_height, inci_azi, inci_zen, path_surf, path_curr,path_mask = self.samples[index]
        inci_azi_tensor = torch.tensor(one_hot_encode_8(int(inci_azi)),dtype=torch.float32)
        inci_zen_tensor = torch.tensor(one_hot_encode_7(int(inci_zen)-1),dtype=torch.float32)


        sample_surf = torch.Tensor(self.loader(path_surf)).unsqueeze(0)
        sample_curt = torch.Tensor(self.loader(path_curr))
        sample_mask = torch.Tensor(self.loader(path_mask)).unsqueeze(0)
        sample_curt = sample_curt[0:6]
        if self.transform1 is not None:
            sample_surf = self.transform1(sample_surf)
            sample_curt = self.transform2(sample_curt)
        # sample_curr,sample_surf = transform_train(sample_curr,sample_surf)

        return inci_azi_tensor, inci_zen_tensor, sample_surf, sample_curt ,sample_mask

    def __len__(self):
        return len(self.samples)

# t1 = transforms.Compose([
#         transforms.Normalize(mean=(-0.05661479687218054), std=(0.7472850317902101), inplace=True)
#     ])
#
# t2 = transforms.Compose([
#         transforms.Normalize(mean=(-0.0034205766274340607, 0.0036823689061374378, 0.0005890796175792815,0.0005033695343457479,-0.0009203318524131545,0.00035165509500145845),
#                              std=(3.688634489663193, 3.7355843905866717, 0.32717253401117763,0.3346149050832342,0.561754524941943,0.5684935178281643), inplace=True)
#     ])
#
# d_path = os.path.split(os.path.split(os.path.abspath(__file__))[0])[0]
# dataset = DatasetFolder(os.path.join(d_path, 'ViewData'), transform1 = t1, transform2=t2)
# train_ratio = 0.8  # 训练集比例
# test_ratio = 0.2   # 测试集比例
#
# # 计算切分点索引
# train_size = int(len(dataset) * train_ratio)
# test_size = len(dataset) - train_size
#
# # 切分数据集
# train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
# train_dataloader = data.DataLoader(train_dataset, batch_size=1, shuffle=False)
# print(len(train_dataloader))
# data_iter = iter(train_dataloader)
#
# inci, currList = [],[]
# # 获取一个批次的数据
# for i in range(10):
#     code,surf,curt = next(data_iter)
#     colsB = [surf,curt[:,[0],:,:],curt[:,[1],:,:],curt[:,[2],:,:],curt[:,[3],:,:],curt[:,[4],:,:],curt[:,[5],:,:]]
#     colsB = torch.cat(colsB, 3).detach().cpu()
#     inci.append(code.unsqueeze(0))
#     currList.append(colsB)
#
#     # codeList = torch.cat(codeList, 0).resize(args.batch * 2,24).numpy()
# inci =  torch.cat(inci,0)
# currList = torch.cat(currList, 0)
# utils.save_image(currList, f'test.jpg', normalize=True,nrow=1)
# print(inci)
# print()