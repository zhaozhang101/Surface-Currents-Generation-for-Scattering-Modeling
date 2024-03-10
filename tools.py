import math
import numpy as np
import matplotlib.pyplot as plt
import numpy
import os
import math
import os
import matplotlib.pyplot as plt
from scipy import io
import numpy
import numpy as np
from PIL import Image
import cv2
from scipy.interpolate import NearestNDInterpolator
from skimage import transform
import torch

def inverseTransform(item,mean,std):
    if len(item)>1:
        for i in range(len(item)):
            item[i] = (item[i] * std[i] + mean[i])/1000
        return item
    else:
        for i in range(len(item)):
            item[i] = item[i] * std[i] + mean[i]
        return item[0]

def inverseTransform_numpy(item,mean,std):
    if len(item)>1:
        for i in range(len(item)):
            item[i] = numpy.array((item[i] * std[i] + mean[i])/1000)
        return item
    else:
        for i in range(len(item)):
            item[i] = numpy.array(item[i] * std[i] + mean[i])
        return item[0]

pai = math.pi
freq, epsilon0, mu0 = 3e11, 8.854187817e-12, 4 * pai * 1e-7
omega, k0 = 2 * pai * freq, 2 * pai * freq * math.sqrt(mu0 * epsilon0)

def Curt2Efar(phi,outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf):
    matSize = outIxrel.shape[0]
    theta = np.linspace(-90, 90, 181) * pai / 180
    dist = 10
    xFarfield, yFarfield, z_Farfield = dist * np.sin(theta) * np.cos(phi/180*pai), dist * np.sin(theta) * np.sin(phi/180*pai), dist * np.cos(theta)
    # xFarfield, yFarfield, z_Farfield = 10 * np.sin(theta), np.zeros(len(theta)), 10 * np.cos(theta)
    Jx_tmp, Jy_tmp, Jz_tmp = outIxrel + 1j * outIximg, outIyrel + 1j * outIyimg, outIzrel + 1j * outIzimg
    # 面积
    S = 1.225e-3 / matSize ** 2
    Efar, Efar_abs = np.zeros(shape=(len(theta),3)), np.zeros(len(theta))
    for k in range(len(theta)):
        x = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
        y = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
        xx, yy = np.meshgrid(x, y)
        field_mat = np.expand_dims(np.array([xFarfield[k], yFarfield[k], z_Farfield[k]]), axis=(1,2)).repeat(matSize, 1).repeat(matSize, 2)
        source_mat = np.concatenate((np.expand_dims(xx,0),np.expand_dims(yy,0),np.expand_dims(surf,0)),axis=0)
        Rdelta = field_mat-source_mat
        field_abs = np.linalg.norm(field_mat, axis=0)
        Rdelta_abs = np.linalg.norm(Rdelta, axis=0)
        g = np.exp(-1j * k0 * Rdelta_abs) / (4 * pai * Rdelta_abs)
        J_tmp = np.concatenate((np.expand_dims(Jx_tmp,0),np.expand_dims(Jy_tmp,0),np.expand_dims(Jz_tmp,0)),axis=0)

        G_J = g * (1 + 1j / (k0 * Rdelta_abs) - 1 / ((k0 * Rdelta_abs) ** 2)) * J_tmp + g * (
                3 / ((k0 * Rdelta_abs) ** 2) - 3j / (k0 * Rdelta_abs) - 1) * (np.sum(J_tmp * Rdelta,axis=0)) * Rdelta / (Rdelta_abs ** 2)
        Esc_term2 = -1j * omega * mu0 * G_J * S * field_abs / (np.exp(-1j * k0 * field_abs))
        # Esc_term2 = -1j * omega * mu0 * G_J * S
        Esc = np.sum(Esc_term2,axis=(1,2))
        Efar[k,:] = np.abs(Esc)
        Efar_abs[k] = np.linalg.norm(Efar[k,:])

    return theta, Efar_abs

# def Curt2Efar_3D(outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf):
#     matSize = outIxrel.shape[0]
#     theta = np.linspace(0, 90, 91) * pai / 180
#     phi = np.linspace(0,360,361) * pai / 180
#     xFarfield, yFarfield, z_Farfield = 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.cos(phi),0)), 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.sin(phi),0)), 10 * np.dot(np.expand_dims(np.cos(theta),1),np.expand_dims(np.ones_like(phi),0))
#     Jx_tmp, Jy_tmp, Jz_tmp = outIxrel + 1j * outIximg, outIyrel + 1j * outIyimg, outIzrel + 1j * outIzimg
#     S = 9e-4 / matSize ** 2
#     Efar, Efar_abs = np.zeros(shape=(3,len(theta),len(phi))), np.zeros(shape=(len(theta),len(phi)))
#     for phi_i in range(len(phi)):
#         for theta_i in range(len(theta)):
#             x = np.linspace(-15 / 1000, 15 / 1000, 328)
#             y = np.linspace(-15 / 1000, 15 / 1000, 328)
#             xx, yy = np.meshgrid(x, y)
#             field_mat = np.expand_dims(np.array([xFarfield[theta_i,phi_i], yFarfield[theta_i,phi_i], z_Farfield[theta_i,phi_i]]), axis=(1,2)).repeat(matSize, 1).repeat(matSize, 2)
#             source_mat = np.concatenate((np.expand_dims(xx,0),np.expand_dims(yy,0),np.expand_dims(surf,0)),axis=0)
#             Rdelta = field_mat-source_mat
#             field_abs = np.linalg.norm(field_mat, axis=0)
#             Rdelta_abs = np.linalg.norm(Rdelta, axis=0)
#             g = np.exp(-1j * k0 * Rdelta_abs) / (4 * pai * Rdelta_abs)
#             J_tmp = np.concatenate((np.expand_dims(Jx_tmp,0),np.expand_dims(Jy_tmp,0),np.expand_dims(Jz_tmp,0)),axis=0)
#
#             G_J = g * (1 + 1j / (k0 * Rdelta_abs) - 1 / ((k0 * Rdelta_abs) ** 2)) * J_tmp + g * (
#                     3 / ((k0 * Rdelta_abs) ** 2) - 3j / (k0 * Rdelta_abs) - 1) * (np.sum(J_tmp * Rdelta,axis=0)) * Rdelta / (Rdelta_abs ** 2)
#             Esc_term2 = -1j * omega * mu0 * G_J * S * field_abs / (np.exp(-1j * k0 * field_abs))
#             Esc = np.sum(Esc_term2,axis=(1,2))
#             Efar[:,theta_i,phi_i] = np.abs(Esc)
#             Efar_abs[theta_i,phi_i] = np.linalg.norm(Efar[:,theta_i,phi_i])
#         print("phi:",phi_i)
#     return theta, Efar_abs

def Curt2Efar_3D(outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf):
    matSize = outIxrel.shape[0]
    theta = np.linspace(0, 90, 91) * pai / 180
    phi = np.linspace(0,360,361) * pai / 180
    xFarfield, yFarfield, z_Farfield = 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.cos(phi),0)), 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.sin(phi),0)), 10 * np.dot(np.expand_dims(np.cos(theta),1),np.expand_dims(np.ones_like(phi),0))
    Jx_tmp, Jy_tmp, Jz_tmp = outIxrel + 1j * outIximg, outIyrel + 1j * outIyimg, outIzrel + 1j * outIzimg
    S = 1.225e-3 / matSize ** 2
    Efar, Efar_abs = np.zeros(shape=(3,len(theta),len(phi))), np.zeros(shape=(len(theta),len(phi)))
    for phi_i in range(len(phi)):
        for theta_i in range(len(theta)):
            x = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
            y = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
            xx, yy = np.meshgrid(x, y)
            field_mat = np.expand_dims(np.array([xFarfield[theta_i,phi_i], yFarfield[theta_i,phi_i], z_Farfield[theta_i,phi_i]]), axis=(1,2)).repeat(matSize, 1).repeat(matSize, 2)
            source_mat = np.concatenate((np.expand_dims(xx,0),np.expand_dims(yy,0),np.expand_dims(surf,0)),axis=0)
            Rdelta = field_mat-source_mat
            field_abs = np.linalg.norm(field_mat, axis=0)
            Rdelta_abs = np.linalg.norm(Rdelta, axis=0)
            g = np.exp(-1j * k0 * Rdelta_abs) / (4 * pai * Rdelta_abs)
            J_tmp = np.concatenate((np.expand_dims(Jx_tmp,0),np.expand_dims(Jy_tmp,0),np.expand_dims(Jz_tmp,0)),axis=0)

            G_J = g * (1 + 1j / (k0 * Rdelta_abs) - 1 / ((k0 * Rdelta_abs) ** 2)) * J_tmp + g * (
                    3 / ((k0 * Rdelta_abs) ** 2) - 3j / (k0 * Rdelta_abs) - 1) * (np.sum(J_tmp * Rdelta,axis=0)) * Rdelta / (Rdelta_abs ** 2)
            Esc_term2 = -1j * omega * mu0 * G_J * S * field_abs / (np.exp(-1j * k0 * field_abs))
            Esc = np.sum(Esc_term2,axis=(1,2))
            Efar[:,theta_i,phi_i] = np.abs(Esc)
            Efar_abs[theta_i,phi_i] = np.linalg.norm(Efar[:,theta_i,phi_i])
        print("phi:",phi_i)
    return theta, Efar_abs

def Curt2Efar_3D_FAST(outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf):
    matSize = outIxrel.shape[0]
    theta = np.linspace(0, 90, 91) * pai / 180
    phi = np.linspace(0,360,361) * pai / 180
    xFarfield, yFarfield, zFarfield = 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.cos(phi),0)), 10 * np.dot(np.expand_dims(np.sin(theta),1),np.expand_dims(np.sin(phi),0)), \
        10 * np.dot(np.expand_dims(np.cos(theta), 1), np.expand_dims(np.ones_like(phi), 0))
    if not isinstance(outIxrel, torch.Tensor):
        outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg = \
            torch.Tensor(outIxrel), torch.Tensor(outIyrel), torch.Tensor(outIzrel), torch.Tensor(outIximg), torch.Tensor(outIyimg), torch.Tensor(outIzimg)
    Jx_tmp, Jy_tmp, Jz_tmp = torch.complex(outIxrel,outIximg), torch.complex(outIyrel,outIyimg), \
        torch.complex(outIzrel,outIzimg)
    Jx_tmp, Jy_tmp, Jz_tmp = Jx_tmp.numpy().flatten(), Jy_tmp.numpy().flatten(), Jz_tmp.numpy().flatten()

    S = 1.225e-3 / matSize ** 2
    Efar, Efar_abs = np.zeros(shape=(3,len(theta),len(phi))), np.zeros(shape=(len(theta),len(phi)))
    for phi_i in range(len(phi)):
        x = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
        y = np.linspace(-17.5 / 1000, 17.5 / 1000, 328)
        xx, yy = np.meshgrid(x, y)
        field_mat = np.zeros(shape=(3,matSize*matSize,len(theta)))
        a = np.expand_dims(xFarfield[:, phi_i],0).repeat(matSize*matSize,0)
        field_mat[0, :, :] = np.expand_dims(xFarfield[:, phi_i],0).repeat(matSize*matSize,0)
        field_mat[1, :, :] = np.expand_dims(yFarfield[:, phi_i],0).repeat(matSize*matSize,0)
        field_mat[2, :, :] = np.expand_dims(zFarfield[:, phi_i],0).repeat(matSize*matSize,0)
        # field_mat = np.expand_dims(np.array([xFarfield[theta_i,phi_i], yFarfield[theta_i,phi_i], zFarfield[theta_i,phi_i]]), axis=(1,2)).repeat(matSize, 1).repeat(matSize, 2)
        source_mat = np.concatenate((np.expand_dims(xx,0),np.expand_dims(yy,0),np.expand_dims(surf,0)),axis=0)
        c,h,w = source_mat.shape
        source_mat = np.expand_dims(np.resize(source_mat,(c,h*w)),2).repeat(len(theta),2)
        Rdelta = field_mat-source_mat
        field_abs = np.linalg.norm(field_mat, axis=0)
        Rdelta_abs = np.linalg.norm(Rdelta, axis=0)
        g = np.exp(-1j * k0 * Rdelta_abs) / (4 * pai * Rdelta_abs)
        J_tmp = np.concatenate((np.expand_dims(Jx_tmp,0),np.expand_dims(Jy_tmp,0),np.expand_dims(Jz_tmp,0)),axis=0)
        J_tmp = np.expand_dims(J_tmp,2).repeat(len(theta),2)
        G_J = g * (1 + 1j / (k0 * Rdelta_abs) - 1 / ((k0 * Rdelta_abs) ** 2)) * J_tmp + g * (
                3 / ((k0 * Rdelta_abs) ** 2) - 3j / (k0 * Rdelta_abs) - 1) * (np.sum(J_tmp * Rdelta,axis=0)) * Rdelta / (Rdelta_abs ** 2)
        Esc_term2 = -1j * omega * mu0 * G_J * S * field_abs / (np.exp(-1j * k0 * field_abs))
        Esc = np.sum(Esc_term2,axis=1)
        Efar[:,:,phi_i] = np.squeeze(np.abs(Esc))   # 3,91
        Efar_abs[:,phi_i] = np.linalg.norm(Efar[:,:,phi_i],axis=0)
        print("phi:",phi_i)
    return theta, Efar_abs

def loadffe2D(azi,name):
    data = numpy.loadtxt(fname=name, dtype=float, skiprows=16)
    x_y = np.round(data[:, 0:2])
    x_y = x_y[:, [1, 0]]

    Eth_abs = np.sqrt((data[:, 2] ** 2 + data[:, 3] ** 2))
    Eph_abs = np.sqrt((data[:, 4] ** 2 + data[:, 5] ** 2))
    Eabs = np.sqrt(Eth_abs ** 2 + Eph_abs ** 2)
    # b = E[0:91]
    # a = E[180 * 91 + 1:181 * 91]
    # plt.plot(np.linspace(1, 90, 90) * pai / 180 * 180 / pai, a, 'g', 'linewidth', 2)
    # plt.plot(np.linspace(1, 91, 91) * pai / 180 * 180 / pai, b, 'k', 'linewidth', 2)
    # plt.ylim(0, 0.15)
    # plt.show()
    azi = int(azi)
    if azi >= 180:
        parta_begin, parta_end = 91 * azi + 1, 91 * (azi + 1)
        partb_begin, partb_end = 91 * (azi - 180), 91 * (azi - 179)
    else:
        parta_begin, parta_end = 91 * azi + 1, 91 * (azi + 1)
        partb_begin, partb_end = 91 * (azi + 180), 91 * (azi + 181)

    a = np.flip(Eabs[partb_begin:partb_end])
    b = Eabs[parta_begin:parta_end]
    E_real = np.concatenate((a,b),axis=0)

    # E_real = np.flip(np.concatenate((np.flip(E[1:91]), E[180 * 91:181 * 91])))
    return E_real

theta_max = 91
phi_max = 361

def loadffe3D(name):
    data = numpy.loadtxt(fname=name, dtype=float, skiprows=16)
    x_y = np.round(data[:, 0:2])
    x_y = x_y[:, [1, 0]]
    Eth_abs = np.sqrt((data[:, 2] ** 2 + data[:, 3] ** 2))
    Eth_ang = np.angle(data[:, 2] + 1j * data[:, 3])
    Eph_abs = np.sqrt((data[:, 4] ** 2 + data[:, 5] ** 2))
    Eph_ang = np.angle(data[:, 4] + 1j * data[:, 5])

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
    abc = np.concatenate([np.expand_dims(Eph_abs_mat,0),np.expand_dims(Eth_abs_mat,0)],axis=0)
    # print(abc)
    E_mat = np.linalg.norm(abc,axis=0)
    return E_mat

def inverseTransform_tensor(item,mean,std):
    if len(item)>1:
        for i in range(len(item)):
            item[i] = (item[i] * std[i] + mean[i])/1000
        return item
    else:
        for i in range(len(item)):
            item[i] = item[i] * std[i] + mean[i]
        return item[0]

def RMS_dBCalcu(matA,matB):
    matA = matA.numpy()
    matB = matB.numpy()
    if np.min(matA)<0:  # 判断是不是虚实部
        RMSE_Real = np.sqrt(np.mean((matA - matB) ** 2))*1000
        RMSE = np.std(matA)*1000
    else:
        theshold = min(np.max(matA)*0.05,np.max(matB)*0.05)
        RMSE_Real = np.sqrt(np.mean((10 * np.log10(matA) - 10 * np.log10(matB)) ** 2))
        matA = np.clip(matA,a_min=theshold,a_max=None)
        matB = np.clip(matB, a_min=theshold, a_max=None)
        RMSE = np.sqrt(np.mean((10*np.log10(matA)-10*np.log10(matB))**2))
    print(f'RMSE is::{RMSE_Real},{RMSE}')
    return 0


def MAKECURTPIC(im_path,inciAzi, inciZen, surf, mask, outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outIxrel_R, outIyrel_R, outIzrel_R, outIximg_R, outIyimg_R, outIzimg_R):
    zenAssist = torch.tensor([10, 20, 30, 40, 50, 60, 70], device=inciZen.device)
    aziAssist = torch.tensor([0, 45, 90, 135, 180, 225, 270, 315], device=inciAzi.device)
    azi, zen = torch.sum(inciAzi * aziAssist, dim=1, keepdim=True), torch.sum(inciZen * zenAssist, dim=1, keepdim=True)


    geneCurtX = [outIxrel_R[[0], :, :, :],
                 outIxrel[[0], :, :, :],
                 outIximg_R[[0], :, :, :], outIximg[[0], :, :, :]]
    geneCurtY = [outIyrel_R[[0], :, :, :],
                 outIyrel[[0], :, :, :],
                 outIyimg_R[[0], :, :, :], outIyimg[[0], :, :, :]]
    geneCurtZ = [outIzrel_R[[0], :, :, :],
                 outIzrel[[0], :, :, :],
                 outIzimg_R[[0], :, :, :], outIzimg[[0], :, :, :]]

    geneCurtX = torch.cat(geneCurtX, 3)
    geneCurtY = torch.cat(geneCurtY, 3)
    geneCurtZ = torch.cat(geneCurtZ, 3)


    # utils.save_image(geneCurtX, f'{im_path}/{azi}_{zen}_X.jpg', normalize=True,
    #                  nrow=1)
    # utils.save_image(geneCurtY, f'{im_path}/{azi}_{zen}_Y.jpg', normalize=True,
    #                  nrow=1)
    # utils.save_image(geneCurtZ, f'{im_path}/{azi}_{zen}_Z.jpg', normalize=True,
    #                  nrow=1)
    geneCurtX,geneCurtY,geneCurtZ = geneCurtX.squeeze()*1000,geneCurtY.squeeze()*1000,geneCurtZ.squeeze()*1000
    # fig = plt.figure()
    # ax = fig.add_subplot(312)
    fig,ax = plt.subplots(3,1)
    # ax.plot(theta * 180 / pai, Efar_realfromos, 'b.-', linewidth=1)
    im1 = ax[0].imshow(geneCurtX.numpy(), cmap='jet')
    im2 = ax[1].imshow(geneCurtY.numpy(), cmap='jet')
    im3 = ax[2].imshow(geneCurtZ.numpy(), cmap='jet')

    ax[0].set_xticks([])
    ax[0].set_yticks([])
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[2].set_xticks([])
    ax[2].set_yticks([])

    # ax.set_xlabel(r'$\theta$ of $Efar_{\phi=0}$', fontsize=15, fontfamily='serif')
    # ax.set_ylabel('E [V/m]', fontsize=15, fontfamily='serif')

    cbar = fig.colorbar(im1, ax=ax[0])
    cbar = fig.colorbar(im2, ax=ax[1])
    cbar = fig.colorbar(im3, ax=ax[2])

    subscript_text = 'Subscript$_{example}$'
    # ax.set_title(
    #     'Recovered $Efar_{\phi=0}$ by AI / ground ' + '\n' + 'truth current at ' + f'incident angle={incidentAngle}°')
    plt.savefig(f'img/I_relimg/'+f'{azi}_{zen}'+'.png',bbox_inches='tight')
    # plt.show()
    plt.close(fig)