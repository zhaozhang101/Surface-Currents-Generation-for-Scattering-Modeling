import numpy as np
import matplotlib.pyplot as plt
import os
# 生成Mask的轮廓图



azimeth = np.linspace(0, 315, 8)
print(azimeth)
zenith = np.linspace(10, 70, 7)
distance = 145
for i in range(len(azimeth)):
    for j in range(len(zenith)):
        Sx = distance * np.sin(zenith[j]/ 180 * np.pi) * np.cos(azimeth[i]/ 180 * np.pi)
        Sy = distance * np.sin(zenith[j]/ 180 * np.pi) * np.sin(azimeth[i]/ 180 * np.pi)
        Sz = distance * np.cos(zenith[j]/ 180 * np.pi)

        x = np.linspace(-17.5, 17.5, 328)
        y = np.linspace(-17.5, 17.5, 328)
        xx, yy = np.meshgrid(x, y)
        zz = np.zeros_like(xx)
        Surf_mat = np.concatenate((np.expand_dims(xx, 0), np.expand_dims(yy, 0), np.expand_dims(zz, 0)), axis=0)

        vec1 = np.expand_dims(np.expand_dims(np.array([Sx, Sy, Sz]), 1), 1)
        vec1 = np.tile(vec1, (1, 328, 328))
        vec2 = vec1 - Surf_mat

        dot_product = np.sum(vec1 * vec2, axis=0)
        norm_A = np.linalg.norm(vec1, axis=0)
        norm_B = np.linalg.norm(vec2, axis=0)
        cos_theta = dot_product / (norm_A * norm_B)
        cos_theta[cos_theta < np.cos(2 / 180 * np.pi)] = 0
        cos_theta[cos_theta >= np.cos(2 / 180 * np.pi)] = 1
        # 使用 imshow 绘制矩阵
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.imshow(cos_theta, cmap='viridis', interpolation='nearest')

        if not os.path.exists('mask'):
            os.mkdir('mask')
        if not os.path.exists('maskfig'):
            os.mkdir('maskfig')
        np.save(f'mask/mask_azi{int(azimeth[i])}_zen{int(zenith[j])}.npy', cos_theta)
        ax.set_title(
            f'Mask_azi:{azimeth[i]}_zen:{zenith[j]}')
        plt.savefig(f'maskfig/mask_azi{int(azimeth[i])}_zen{int(zenith[j])}.png', bbox_inches='tight')
        plt.close(fig)
        print()
