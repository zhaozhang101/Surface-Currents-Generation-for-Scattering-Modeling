import math
import os
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator
from skimage import transform
# from numba import jit
"""
默认35*35 mm
生成训练数据的代码，主要针对 表面电流 和 粗糙表面两个特征
"""
size = 328
theta_max = 90
phi_max = 360


def os2npy_Relimg(root, file, gene_folder):
    data = np.loadtxt(fname=os.path.join(root, file),dtype=float,skiprows=13)
    x_y = (data[:,1:3]*1000+17.5)*size/35

    Ix_rel_list = data[:,4]
    Ix_img_list = data[:,5]
    Iy_rel_list = data[:,6]
    Iy_img_list = data[:,7]
    Iz_rel_list = data[:,8]
    Iz_img_list = data[:,9]

    Ix_abs_list = np.sqrt((data[:, 4] ** 2 + data[:, 5] ** 2))
    Ix_ang_list = np.angle(data[:, 4] + 1j * data[:, 5])
    Iy_abs_list = np.sqrt((data[:, 6] ** 2 + data[:, 7] ** 2))
    Iy_ang_list = np.angle(data[:, 6] + 1j * data[:, 7])
    Iz_abs_list = np.sqrt((data[:, 8] ** 2 + data[:, 9] ** 2))
    Iz_ang_list = np.angle(data[:, 8] + 1j * data[:, 9])

    X = np.linspace(0, size-1, size)
    Y = np.linspace(0, size-1, size)
    X, Y = np.meshgrid(X, Y)

    current = np.zeros(shape=(12,size,size))
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

    interp = NearestNDInterpolator(x_y, Ix_abs_list)
    Ix_abs_mat = interp(X, Y) * 1000
    interp = NearestNDInterpolator(x_y, Ix_ang_list)
    Ix_ang_mat = interp(X, Y) / np.pi
    interp = NearestNDInterpolator(x_y, Iy_abs_list)
    Iy_abs_mat = interp(X, Y) * 1000
    interp = NearestNDInterpolator(x_y, Iy_ang_list)
    Iy_ang_mat = interp(X, Y) / np.pi
    interp = NearestNDInterpolator(x_y, Iz_abs_list)
    Iz_abs_mat = interp(X, Y) * 1000
    interp = NearestNDInterpolator(x_y, Iz_ang_list)
    Iz_ang_mat = interp(X, Y) / np.pi


    current[0] = Ix_rel_mat
    current[1] = Ix_img_mat
    current[2] = Iy_rel_mat
    current[3] = Iy_img_mat
    current[4] = Iz_rel_mat
    current[5] = Iz_img_mat

    current[6] = Ix_abs_mat
    current[7] = Iy_abs_mat
    current[8] = Iz_abs_mat
    current[9] = Ix_ang_mat
    current[10] = Iy_ang_mat
    current[11] = Iz_ang_mat
    savepath = os.path.join(os.getcwd(), gene_folder, 'current')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    zenith = os.path.split(os.path.split(os.path.split(root)[0])[0])[1].split('_',1)[1]
    azimuth = os.path.split(os.path.split(os.path.split(os.path.split(root)[0])[0])[0])[1].split('_',1)[1]
    np.save(os.path.join(savepath, os.path.split(root)[1] +'_' + azimuth + '_' + zenith+ '.npy'),current)
    return 0


def ffe2npy(root, file, gene_folder):
    data = np.loadtxt(fname=os.path.join(root, file), dtype=float, skiprows=18)

    x_y = np.round(data[:, 0:2])
    x_y = x_y[:,[1,0]]
    Eth_abs = np.sqrt((data[:, 4] ** 2 + data[:, 5] ** 2))
    Eth_ang = np.angle(data[:, 4] + 1j*data[:, 5])
    Eph_abs = np.sqrt((data[:, 6] ** 2 + data[:, 7] ** 2))
    Eph_ang = np.angle(data[:,6] + 1j*data[:, 7])

    X = np.linspace(0, theta_max - 1, theta_max)
    Y = np.linspace(0, phi_max - 1, phi_max)
    X, Y = np.meshgrid(Y, X)
    interp = NearestNDInterpolator(x_y, Eth_abs)
    Eth_abs_mat = interp(X, Y)
    interp = NearestNDInterpolator(x_y, Eth_ang)
    Eth_ang_mat = interp(X, Y)
    interp = NearestNDInterpolator(x_y, Eph_abs)
    Eph_abs_mat = interp(X, Y)
    interp = NearestNDInterpolator(x_y, Eph_ang)
    Eph_ang_mat = interp(X, Y)


def mat2npy(root, file, gene_folder):
    data = io.loadmat(os.path.join(root, file))
    data = data['height']
    resize_transform = transform.resize(data, (size, size), order=3)
    # order=0表示使用最近邻插值，order=1表示使用双线性插值，order=2表示使用双三次插值。默认情况下，order=1
    # # 0: Nearest-neighbor
    # # 1: Bi-linear (default)
    # # 2: Bi-quadratic
    # # 3: Bi-cubic
    # # 4: Bi-quartic
    # # 5: Bi-quintic
    # height = np.interp(resize_transform, (resize_transform.min(), resize_transform.max()), (0, 255))
    savepath = os.path.join(os.getcwd(), gene_folder, 'surface')
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    np.save(os.path.join(savepath, file.split('.',1)[0] + '.npy'),resize_transform)

def read_txt_files_in_folder(folder_path,gene_folder):
    inx = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # if file.endswith(".os"):
            #     os2npy_Relimg(root, file, gene_folder)
            #     print(inx)
            #     inx += 1
            # if file.endswith(".ffe"):
            #     ffe2npy(root, file, gene_folder)
            #     print(inx)
            #     inx += 1
            if file.endswith(".mat"):
                mat2npy(root, file, gene_folder)
                print(inx)
                inx += 1
    return inx

# 用法示例
if __name__ == "__main__":
    new_cur_folder = '../Simulation'
    mat_folder = '../MAT'
    gene_folder = '../ViewData'
    # new_cur_folder = 'Simulation/Azimuth_90/Zenith_40/RS35_10'
    idx = read_txt_files_in_folder(mat_folder,gene_folder)
    print(idx)
