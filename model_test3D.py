from A_modelSelf_New import *
from utils import *
from tools import *
from scipy.io import savemat


mean = (
    0.0001081127327315457, -0.0001301721777078952, -8.828423767000199e-05, 6.556262019127316e-06, -1.7480090477601792e-05,
    -6.484315591228738e-05)
std = (
    0.8846734754648876, 0.8859368369975612, 0.8800443938755093, 0.8809101371889487, 0.1227683005584715,
    0.1227657849244166)
mean_surf = (-0.014061627126862692 ,)
std_surf = (0.8358827002436682,)

inciangel = [10,20,30,40,50,60,70]


inci_azi = 225
for inci_zen in inciangel:
    # inci_zen = 70
    testfilepath_SameSurf = '../Simulation/Azimuth_135'
    keyinfo = 'Zenith_' + str(inci_zen)
    osname = os.path.join(testfilepath_SameSurf, keyinfo, 'RS35_25', 'RS35_25_025', 'RS35_25_025_Currents1.os')
    surfname = '../MAT/RS35_25_025.mat'
    ffefile = os.path.join(testfilepath_SameSurf, keyinfo, 'RS35_25', 'RS35_25_025', 'RS35_25_025_FarField1.ffe')

    size = 328
    data = io.loadmat(surfname)
    data = data['height']
    resize_transform = transform.resize(data, (size, size), order=3)

    ckptfile = 'ck_model30_noSSIM.pt'
    Gnrt = A_Generator().eval()
    # ckptfile = 'ck_NewModel05holyNN.pt'

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

        surf = (surf_ori - torch.tensor(mean_surf)) / torch.tensor(std_surf)
        curt = (curt_ori[0, 0:6, :, :] - torch.tensor(mean).unsqueeze(1).unsqueeze(1)) / torch.tensor(
            std).unsqueeze(1).unsqueeze(1)

        inciAzi = torch.tensor(one_hot_encode_8(int(inci_azi / 45)), dtype=torch.float32).unsqueeze(0)
        inciZen = torch.tensor(one_hot_encode_7(int(inci_zen / 10) - 1), dtype=torch.float32).unsqueeze(0)

        outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, mask = Gnrt(surf, inciAzi, inciZen)

        AoutIxrel, AoutIximg, AoutIyrel, AoutIyimg, AoutIzrel, AoutIzimg = inverseTransform_tensor(
            [outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg], mean, std)
        AoutIxrel_R, AoutIximg_R, AoutIyrel_R, AoutIyimg_R, AoutIzrel_R, AoutIzimg_R = inverseTransform_tensor(
            [curt_ori[[0], [0], :, :].unsqueeze(0), curt_ori[[0], [1], :, :].unsqueeze(0),
             curt_ori[[0], [2], :, :].unsqueeze(0),
             curt_ori[[0], [3], :, :].unsqueeze(0), curt_ori[[0], [4], :, :].unsqueeze(0),
             curt_ori[[0], [5], :, :].unsqueeze(0)], mean, std)

        img_path = None
        # MAKECURTPIC(img_path, inciAzi, inciZen, surf, mask, AoutIxrel, AoutIyrel, AoutIzrel, AoutIximg, AoutIyimg,
        #             AoutIzimg,
        #             AoutIxrel_R, AoutIyrel_R, AoutIzrel_R, AoutIximg_R, AoutIyimg_R, AoutIzimg_R)

    # 电流和表面的 逆变换
    outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg = inverseTransform(
        [outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg], mean, std)
    surf = inverseTransform([surf], mean_surf, std_surf)

    outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, surf = outIxrel.squeeze().numpy(), \
        outIyrel.squeeze(), outIzrel.squeeze(), outIximg.squeeze(), outIyimg.squeeze(), outIzimg.squeeze(), surf.squeeze()

    # 加载os文件
    outIxrel_os, outIximg_os, outIyrel_os, outIyimg_os, outIzrel_os, outIzimg_os = curt_ori[0, 0, :,:] / 1000, curt_ori[0, 1, :,:] / 1000, \
        curt_ori[0,2, :,:] / 1000, curt_ori[0, 3, :,:] / 1000, curt_ori[0, 4, :,:] / 1000, curt_ori[0,5, :,:] / 1000

    # 由表面电流 计算远场
    surf_calcu = 0.2 * surf_ori.squeeze().numpy() / 1000

    # thetal, Efar_os = Curt2Efar_3D_FAST(outIxrel_os, outIyrel_os, outIzrel_os, outIximg_os, outIyimg_os, outIzimg_os, curt_ori[0, 6, :, :])
    theta, Efar_fake = Curt2Efar_3D_FAST(outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, curt_ori[0, 6, :, :])
    Efar_os = np.array([0,1,2])
    # outIxABS,outIxABS_OS = np.abs(outIxrel + 1j*outIximg),np.abs(outIxrel_os + 1j*outIximg_os)
    # outIyABS,outIyABS_OS = np.abs(outIyrel + 1j*outIyimg),np.abs(outIyrel_os + 1j*outIyimg_os)
    # outIzABS,outIzABS_OS = np.abs(outIzrel + 1j*outIzimg),np.abs(outIzrel_os + 1j*outIzimg_os)

    Emat = loadffe3D(ffefile)
    print(f'=========================={inci_zen}===============================')
    # RMS_dBCalcu(Emat,Efar_fake)
    # RMS_dBCalcu(outIxrel_os, outIxrel)
    # RMS_dBCalcu(outIyrel_os, outIyrel)
    # RMS_dBCalcu(outIzrel_os, outIzrel)
    # RMS_dBCalcu(outIximg_os, outIximg)
    # RMS_dBCalcu(outIyimg_os, outIyimg)
    # RMS_dBCalcu(outIzimg_os, outIzimg)
    #
    # RMS_dBCalcu(outIxABS_OS,outIxABS)
    # RMS_dBCalcu(outIyABS_OS,outIyABS)
    # RMS_dBCalcu(outIzABS_OS,outIzABS)
    # 直接加载ffe数据


    fig,ax = plt.subplots(2,1)
    ax[0].imshow(Emat,cmap='viridis', interpolation='nearest')
    ax[0].set_title('Ground truth Efar')
    ax[0].axis('off')
    ax[1].imshow(Efar_fake, cmap='viridis', interpolation='nearest')
    ax[1].set_title('Recovered Efar')
    ax[1].axis('off')
    plt.savefig(f'Efar/img/{inci_azi}' + keyinfo + '.png', bbox_inches='tight')
    # plt.show()
    plt.close(fig)
    savemat(f'Enear/mat/Efar{inci_azi}_{keyinfo}_VERIFY.mat',{'Ereal': Emat, 'Efake': Efar_fake , 'Efar_os':Efar_os})
    print()


    # savemat(f'Efar/mat/Efar{keyinfo}.mat',{'Ereal': Emat,'Efake':Efar_fake})
    # 使用savemat函数保存数组到mat文件
    # savemat('fakecurt.mat',
    #         {'outIxrel': outIxrel, 'outIyrel': outIyrel, 'outIzrel': outIzrel, 'outIximg': outIximg, 'outIyimg': outIyimg,
    #          'outIzimg': outIzimg, 'surf': surf})
    # savemat('realcurt.mat',
    #         {'outIxrel_os': outIxrel_os, 'outIyrel_os': outIyrel_os, 'outIzrel_os': outIzrel_os, 'outIximg_os': outIximg_os,
    #          'outIyimg_os': outIyimg_os, 'outIzimg_os': outIzimg_os, 'surf_os': surf})


    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # # ax.plot(theta * 180 / pai, Efar_realfromos, 'b.-', linewidth=1)
    # ax.plot(theta * 180 / pai, Efar_real, 'r.-', linewidth=1)
    # ax.plot(theta * 180 / pai, Efar_fake, 'g--*', linewidth=1)
    #
    # ax.legend(["Ground truth","AI method"])
    # ax.set_xlabel(r'$\theta$ of $Efar_{\phi=0}$', fontsize=15, fontfamily='serif')
    # ax.set_ylabel('E [V/m]', fontsize=15, fontfamily='serif')
    # ax.set_xlim([-90,90])
    # ax.set_ylim([0, 0.15])
    # ax.grid()
    # subscript_text = 'Subscript$_{example}$'
    # ax.set_title(
    #     'Recovered $Efar_{\phi=0}$ by AI / ground ' + '\n' + 'truth current at ' + f'incident angle={incidentAngle}°')
    # plt.savefig(f'img/efar/'+keyinfo+'.png',bbox_inches='tight')
    # plt.close(fig)


