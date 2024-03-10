import numpy as np
from scipy.io import savemat
from utils import *
from A_modelSelf_New import *
from tools import *
from matplotlib.ticker import ScalarFormatter, FormatStrFormatter

mean = (
    0.0001081127327315457, -0.0001301721777078952, -8.828423767000199e-05, 6.556262019127316e-06,
    -1.7480090477601792e-05,
    -6.484315591228738e-05)
std = (
    0.8846734754648876, 0.8859368369975612, 0.8800443938755093, 0.8809101371889487, 0.1227683005584715,
    0.1227657849244166)
mean_surf = (-0.014061627126862692,)
std_surf = (0.8358827002436682,)

inciangel = [10,20,30,40,50,60,70]
# inciAzi = [0,45,90,135,180,225,270,315]
inciAzi = [0,45,90]

RMSELIST = np.zeros(shape=(8,7))
RMSELIST_R = np.zeros(shape=(8,7))
for count_a, inci_azi in enumerate(inciAzi):
    for count_z, inci_zen in enumerate(inciangel):
        # inci_azi,inci_zen = 45,40
        # testfilepath_SameSurf = f'../Simulation_GeneTest/Azimuth_{inci_azi}'
        testfilepath_SameSurf = f'../Simulation/Azimuth_{inci_azi}'
        keyinfo = 'Zenith_' + str(inci_zen)
        osname = os.path.join(testfilepath_SameSurf, keyinfo, 'RS35_25', 'RS35_25_025', 'RS35_25_025_Currents1.os')
        surfname = '../MAT/RS35_25_025.mat'
        ffefile = os.path.join(testfilepath_SameSurf, keyinfo, 'RS35_25', 'RS35_25_025', 'RS35_25_025_FarField1.ffe')

        size = 328
        data = io.loadmat(surfname)
        data = data['height']
        resize_transform = transform.resize(data, (size, size), order=3)

        ckptfile = 'ck_model30_noSSIM.pt'
        # ckptfile = 'NewModel03/checkpoint/ck_model10.pt'
        Gnrt = A_Generator().eval()
        if ckptfile is not None:
            ckpt = torch.load(ckptfile, map_location=lambda storage, loc: storage)
            try:
                ckpt_name = os.path.basename(ckptfile)

            except ValueError:
                pass

            Gnrt.load_state_dict(ckpt['Gnrt'])

        with torch.no_grad():
            surf_ori = torch.tensor(resize_transform, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            curt_ori = loadcurt(osname, size)
            # a = curt_ori[0, 6, :, :].numpy()*1000
            # b = surf_ori.squeeze().numpy()*0.2
            surf = (surf_ori - torch.tensor(mean_surf)) / torch.tensor(std_surf)
            curt = (curt_ori[0, 0:6, :, :] - torch.tensor(mean).unsqueeze(1).unsqueeze(1)) / torch.tensor(std).unsqueeze(1).unsqueeze(1)

            inciAzi = torch.tensor(one_hot_encode_8(int(inci_azi/45)), dtype=torch.float32).unsqueeze(0)
            inciZen = torch.tensor(one_hot_encode_7(int(inci_zen/10 - 1)), dtype=torch.float32).unsqueeze(0)

            outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, mask = Gnrt(surf, inciAzi, inciZen)

            AoutIxrel, AoutIximg, AoutIyrel, AoutIyimg, AoutIzrel,  AoutIzimg = inverseTransform_tensor(
                [outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg], mean, std)
            AoutIxrel_R,AoutIximg_R, AoutIyrel_R,AoutIyimg_R, AoutIzrel_R,  AoutIzimg_R = inverseTransform_tensor(
                [curt_ori[[0], [0], :, :].unsqueeze(0), curt_ori[[0], [1], :, :].unsqueeze(0), curt_ori[[0], [2], :, :].unsqueeze(0),
                 curt_ori[[0], [3], :, :].unsqueeze(0), curt_ori[[0], [4], :, :].unsqueeze(0),
                 curt_ori[[0], [5], :, :].unsqueeze(0)], mean, std)

            # 将张量转换为NumPy数组，然后使用Matplotlib显示图像
            # image = AoutIyimg_R[0,:,:,:].permute(1, 2, 0).numpy()  # 调整通道顺序
            # plt.imshow(image)
            # plt.show()
            img_path = None
            # MAKECURTPIC(img_path, inciAzi, inciZen, surf, mask, AoutIxrel, AoutIyrel, AoutIzrel, AoutIximg, AoutIyimg, AoutIzimg,
            #             AoutIxrel_R, AoutIyrel_R, AoutIzrel_R, AoutIximg_R, AoutIyimg_R, AoutIzimg_R)


        # 电流和表面的 逆变换
        outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg = inverseTransform(
            [outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg], mean, std)
        surf = inverseTransform([surf], mean_surf, std_surf)

        outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf = outIxrel.squeeze().numpy(), \
            outIyrel.squeeze().numpy(), outIzrel.squeeze().numpy(), outIximg.squeeze().numpy(), outIyimg.squeeze().numpy(), outIzimg.squeeze().numpy(), surf.squeeze().numpy()

        # 加载os文件
        outIxrel_os, outIximg_os, outIyrel_os, outIyimg_os, outIzrel_os, outIzimg_os = curt_ori[0, 0, :, :]/1000, curt_ori[0, 1, :, :]/1000, curt_ori[0, 2, :, :]/1000, \
            curt_ori[0, 3, :, :]/1000, curt_ori[0, 4, :, :]/1000, curt_ori[0, 5, :, :]/1000

        # 由表面电流 计算远场
        surf_calcu = 0.2 * surf_ori.squeeze().numpy() / 1000
        theta, Efar_fake = Curt2Efar(inci_azi,outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, curt_ori[0, 6, :, :])
        # _, Efar_realfromos = Curt2Efar(inci_azi,outIxrel_os, outIyrel_os, outIzrel_os, outIximg_os, outIyimg_os, outIzimg_os, curt_ori[0, 6, :, :])

        # 直接加载ffe数据
        Efar_real = loadffe2D(inci_azi,ffefile)

        # savemat(f'Efar2D/mat/Efar{inci_azi}_{inci_zen}.mat', {'Ereal': Efar_real, 'Efake': Efar_fake,'Eos':Efar_realfromos})
        print('Done!')

        # 使用savemat函数保存数组到mat文件
        # savemat('fakecurt.mat',
        #         {'outIxrel': outIxrel, 'outIyrel': outIyrel, 'outIzrel': outIzrel, 'outIximg': outIximg, 'outIyimg': outIyimg,
        #          'outIzimg': outIzimg, 'surf': surf})
        # savemat('realcurt.mat',
        #         {'outIxrel_os': outIxrel_os, 'outIyrel_os': outIyrel_os, 'outIzrel_os': outIzrel_os, 'outIximg_os': outIximg_os,
        #          'outIyimg_os': outIyimg_os, 'outIzimg_os': outIzimg_os, 'surf_os': surf})

        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(theta * 180 / pai, Efar_real, 'r^-', linewidth=3)
        ax.plot(theta * 180 / pai, Efar_fake, 'g--*', linewidth=3)
        # ax.plot(theta * 180 / pai, Efar_realfromos, 'b--*', linewidth=1)
        legend = ax.legend(["Ground truth", "Proposed model"])
        legend.get_texts()[0].set_fontsize(12)
        legend.get_texts()[1].set_fontsize(12)
        # ax.set_legend()
        ax.set_xlabel(r'$\theta$ of Efar', fontsize=15, fontfamily='serif')
        ax.set_ylabel('E [V/m]', fontsize=15, fontfamily='serif')
        ax.set_xlim([-90,90])
        ax.set_ylim([-0.001, 0.030])
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        # ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1e'))
        ax.set_xticks([-90,-40,0,40,90])
        ax.grid()
        subscript_text = 'Subscript$_{example}$'
        ax.set_title(
            f'Recovered Efar[phi]={inci_azi} by AI / ground ' + '\n' + 'truth current at ' + f'incident angle=[{inci_azi}][{inci_zen}]°')
        plt.savefig(f'Enear/mat/'+f'Azi{inci_azi}'+f'Zen{inci_zen}'+'.png',bbox_inches='tight')
        # plt.show()
        plt.close(fig)

        # 计算二维的误差
        # idxA, idxB = np.zeros_like(Efar_real), np.zeros_like(Efar_fake)
        # theshold = min(np.max(Efar_real) * 0.05, np.max(Efar_fake) * 0.05)
        # RMSE_Real = np.sqrt(np.mean((10 * np.log10(Efar_real) - 10 * np.log10(Efar_fake)) ** 2))
        # matA = np.clip(Efar_real, a_min=theshold, a_max=None)
        # matB = np.clip(Efar_fake, a_min=theshold, a_max=None)
        # idxA[matA == theshold] = 1
        # idxB[matB == theshold] = 1
        # idx = idxA * idxB
        # staticCount = len(matA) - np.sum(idx)
        # RMSE = np.sqrt(np.sum((10 * np.log10(matA) - 10 * np.log10(matB)) ** 2) / staticCount)
        # RMSELIST[count_a,count_z] = RMSE
        # RMSELIST_R[count_a,count_z] = RMSE_Real
        # print(f'{count_a},{count_z},RMSE_real is::{RMSE_Real},RMSE_threshold is::{RMSE}')
        # print()

savemat(f'Efar2D_ALL.mat', {'RMSELIST': RMSELIST, 'RMSELIST_R': RMSELIST_R})





