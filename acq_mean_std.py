import math
import os
import matplotlib.pyplot as plt
from scipy import io
import numpy
import numpy as np
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator
# from skimage import transform

def Curt_mean_std(folder_path):
    inx = 0
    mat_arrays = [[],[],[],[],[],[],[],[],[],[],[],[]]
    stacked_array = np.zeros(shape=(12,1400,328,328))
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                tmp = np.load(os.path.join(root,file))
                for i in range(12):
                    mat_arrays[i].append(tmp[i])
                print(inx)
                inx += 1
    for i in range(12):
        stacked_array[i] = np.stack(mat_arrays[i])
        std = np.std(stacked_array[i], axis=(0,1,2))
        mean = np.mean(stacked_array[i], axis=(0,1,2))
        print(i,'--std:',std,'mean:',mean)
    return 0

def Surf_mean_std(folder_path):
    inx = 0
    mat_arrays = []
    stacked_array = np.zeros(shape=(25, 35, 35))
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".npy"):
                tmp = np.load(os.path.join(root, file))
                mat_arrays.append(tmp)
                print(inx)
                inx += 1
    stacked_array = np.stack(mat_arrays)
    std = np.std(stacked_array, axis=(0, 1, 2))
    mean = np.mean(stacked_array, axis=(0, 1, 2))
    print('--std:', std, 'mean:', mean)
    return 0

# 用法示例
if __name__ == "__main__":
    new_cur_folder = '../ViewData/current'
    # s = Curt_mean_std(new_cur_folder)
    b = Surf_mean_std('../ViewData/surface')

'''
--std: 0.8358827002436682 mean: -0.014061627126862692      (mm)
Ix_rel, Ix_img, Iy_rel, Iy_img, Iz_rel, Iz_img ,  Ix_abs, Iy_abs, Iz_abs, Ix_ang, Iy_ang, Iz_ang
0 --std: 0.8846734754648876 mean: 0.0001081127327315457
1 --std: 0.8859368369975612 mean: -0.0001301721777078952
2 --std: 0.8800443938755093 mean: -8.828423767000199e-05
3 --std: 0.8809101371889487 mean: 6.556262019127316e-06
4 --std: 0.1227683005584715 mean: -1.7480090477601792e-05
5 --std: 0.1227657849244166 mean: -6.484315591228738e-05
6 --std: 1.161873362900301 mean: 0.46645638012172047
7 --std: 1.1561276459936407 mean: 0.46243883811834624
8 --std: 0.16604906855805418 mean: 0.05070704991728521
9 --std: 0.5744456086135658 mean: -0.0001760040370496479
10 --std: 0.5744517128853651 mean: -0.000177027325651663
11 --std: 0.5744543114822112 mean: -0.00023207408299103755
'''