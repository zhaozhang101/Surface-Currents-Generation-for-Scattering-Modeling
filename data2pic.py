import math
import os
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
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
    I_abs_list = np.sqrt(Ix_abs_list**2 + Iy_abs_list**2 + Iz_abs_list**2)

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
    interp = NearestNDInterpolator(x_y, I_abs_list)
    I_abs_mat =  interp(X, Y) *1000


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

    # savepath = os.path.join(os.getcwd(), gene_folder, 'current')
    # if not os.path.exists(savepath):
    #     os.makedirs(savepath)
    zenith = os.path.split(os.path.split(os.path.split(root)[0])[0])[1].split('_',1)[1]
    azimuth = os.path.split(os.path.split(os.path.split(os.path.split(root)[0])[0])[0])[1].split('_',1)[1]
    surf = os.path.split(root)[1]
    surf = surf[2:]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Iz_rel_mat, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(
        f'curt_Izrel azi={azimuth}_zen={zenith}')
    # plt.show()
    plt.savefig(f'../DataPic/curt_Izrel/Izrel_{surf}_azi{azimuth}_zen{zenith}.png', bbox_inches='tight')
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(Iz_img_mat, cmap='viridis')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(
        f'curt_Izimg azi={azimuth}_zen={zenith}')
    # plt.show()
    plt.savefig(f'../DataPic/curt_Izimg/Izimg_{surf}_azi{azimuth}_zen{zenith}.png', bbox_inches='tight')
    plt.close(fig)
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
    savepath = os.path.join(os.getcwd(), gene_folder, 'surf')
    if not os.path.exists(savepath):
        os.makedirs(savepath)

    surf = file[2:-4]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(resize_transform, cmap='viridis', interpolation='nearest')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im, cax=cax, orientation='vertical')
    ax.set_title(
        f'Curt_Ixrel {surf}')
    plt.savefig(f'../DataPic/surf/surf_{surf}.png', bbox_inches='tight')
    plt.close(fig)


def read_txt_files_in_folder(folder_path,gene_folder):
    inx = 0
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".os"):
                os2npy_Relimg(root, file, gene_folder)
                print(inx)
                inx += 1
            # if file.endswith(".ffe"):
            #     ffe2npy(root, file, gene_folder)
            #     print(inx)
            #     inx += 1
            # if file.endswith(".mat"):
            #     mat2npy(root, file, gene_folder)
            #     print(inx)
            #     inx += 1
    return inx

# 用法示例
if __name__ == "__main__":
    gene_folder = '../DataPic'
    new_cur_folder = '../Simulation'
    mat_folder = '../MAT'

    # new_cur_folder = '../Simulation/Azimuth_90/Zenith_40/RS35_10'
    idx = read_txt_files_in_folder(new_cur_folder,gene_folder)
    print(idx)
