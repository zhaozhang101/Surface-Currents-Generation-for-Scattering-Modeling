import argparse
import matplotlib.pyplot as plt
from utils import *
from torch import optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.optim import lr_scheduler
from model import *
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
import pandas as pd
import warnings
from A_modelSelf_XYZ import A_Generator, SSIM
from lossfun import *
from tools import *

warnings.filterwarnings(action='ignore')

# 0 --std: 0.8846734754648876 mean: 0.0001081127327315457
# 1 --std: 0.8859368369975612 mean: -0.0001301721777078952
# 2 --std: 0.8800443938755093 mean: -8.828423767000199e-05
# 3 --std: 0.8809101371889487 mean: 6.556262019127316e-06
# 4 --std: 0.1227683005584715 mean: -1.7480090477601792e-05
# 5 --std: 0.1227657849244166 mean: -6.484315591228738e-05

mean = (
    0.0001081127327315457, -0.0001301721777078952, -8.828423767000199e-05, 6.556262019127316e-06,
    -1.7480090477601792e-05,
    -6.484315591228738e-05)
std = (
    0.8846734754648876, 0.8859368369975612, 0.8800443938755093, 0.8809101371889487, 0.1227683005584715,
    0.1227657849244166)

mean_surf = (-0.014061627126862692,)
std_surf = (0.8358827002436682,)


# 用于对比生成与真实情况下产生的远场
def ele_far_field_verify(args, Gnrt, test_loader, name, step):
    test_loader = iter(test_loader)
    with torch.no_grad():
        idx = 0
        ffeRMSE, ffeDBRMSE = 0, 0
        for inciAzi, inciZen, surf, curt, mask in (test_loader):
            outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outall = Gnrt(surf, inciAzi, inciZen)
            inci_azi = inverse_one_hot_code_8(numpy.array(inciAzi))
            inci_zen = inverse_one_hot_code_7(numpy.array(inciZen))
            outIxrel, outIximg, outIyrel, outIyimg, outIzrel, outIzimg = inverseTransform_numpy(
                [outIxrel[0, 0, :, :], outIximg[0, 0, :, :], outIyrel[0, 0, :, :],
                 outIyimg[0, 0, :, :], outIzrel[0, 0, :, :], outIzimg[0, 0, :, :]], mean, std)
            outIxrel_R, outIximg_R, outIyrel_R, outIyimg_R, outIzrel_R, outIzimg_R = inverseTransform_numpy(
                [curt[0, 0, :, :], curt[0, 1, :, :], curt[0, 2, :, :], curt[0, 3, :, :], curt[0, 4, :, :],
                 curt[0, 5, :, :]], mean, std)

            surf = inverseTransform([surf], mean_surf, std_surf)

            theta, Efar_fake = Curt2Efar(outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg,
                                         surf[0, 0, :, :] / 1000)
            _, Efar_real = Curt2Efar(outIxrel_R, outIyrel_R, outIzrel_R, outIximg_R, outIyimg_R, outIzimg_R,
                                     surf[0, 0, :, :] / 1000)
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(theta * 180 / pai, Efar_fake, 'g-*', linewidth=1)
            ax.plot(theta * 180 / pai, Efar_real, 'r-*', linewidth=2)
            ax.legend(["AI method", "Ground truth"])
            ax.set_xlabel(r'$\theta$ of E_{far}', fontsize=15, fontfamily='serif')
            ax.set_ylabel('E [V/m]', fontsize=15, fontfamily='serif')
            ax.set_xlim([-90, 90])
            ax.set_ylim([0, 0.03])
            ax.grid()

            ax.set_title(
                'Recovered $Efar_{\phi=0}$ by AI / ground ' + '\n' + 'truth current at ' + f'inciangle={inci_azi}_{inci_zen}°')
            if not os.path.exists('ffeimg'):
                os.mkdir('ffeimg')
            plt.savefig(f'ffeimg/sample_{idx}.png', bbox_inches='tight')
            plt.close(fig)
            idx += 1
            ffermse = np.sqrt(np.mean(np.square(Efar_real - Efar_fake)))
            ffedBrmse = np.sqrt(np.mean(np.square(10 * np.log10(Efar_real) - 10 * np.log10(Efar_fake))))
            # 生成 32 个样本数据
            if idx >= 60:
                break
            ffeRMSE += ffermse / 60
            ffeDBRMSE += ffedBrmse / 60
        print('ffeRMSE of Efar is ', ffeRMSE)
        print('ffeDBRMSE of Efar is ', ffeDBRMSE)


# 用于表面电流的 统计误差
def model_error_statistic(args, Gnrt, test_loader, name, step):
    IxABS, IyABS, IzABS = 0, 0, 0
    IxRMSE, IyRMSE, IzRMSE = 0, 0, 0
    IxMEAN, IyMEAN, IzMEAN = 0, 0, 0
    IDBx, IDBy, IDBz = 0, 0, 0
    len_data = len(test_loader)
    with torch.no_grad():
        for inciAzi, inciZen, surf, curt, mask in (test_loader):
            outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outall = Gnrt(surf, inciAzi, inciZen)
            outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg = inverseTransform_numpy(
                [outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg], mean, std)
            outIxrel_R, outIyrel_R, outIzrel_R, outIximg_R, outIyimg_R, outIzimg_R = inverseTransform_numpy(
                [curt[0, 0, :, :], curt[0, 2, :, :], curt[0, 4, :, :], curt[0, 1, :, :], curt[0, 3, :, :],
                 curt[0, 5, :, :]], mean, std)
            outIx, outIy, outIz = outIxrel + 1j * outIximg, outIyrel + 1j * outIyimg, outIzrel + 1j * outIzimg
            outIx_R, outIy_R, outIz_R = outIxrel_R + 1j * outIximg_R, outIyrel_R + 1j * outIyimg_R, outIzrel_R + 1j * outIzimg_R

            IxMean, IyMean, IzMean = np.mean(np.abs(outIx_R)), np.mean(np.abs(outIy_R)), np.mean(np.abs(outIz_R))
            IxMEAN += np.sum(IxMean) / len_data
            IyMEAN += np.sum(IyMean) / len_data
            IzMEAN += np.sum(IzMean) / len_data

            IxAbserr, IyAbserr, IzAbserr = np.abs(np.abs(outIx) - np.abs(outIx_R)), np.abs(
                np.abs(outIy) - np.abs(outIy_R)), np.abs(np.abs(outIz) - np.abs(outIz_R))
            IxABS += np.sum(IxAbserr) / (args.size ** 2) / len_data
            IyABS += np.sum(IyAbserr) / (args.size ** 2) / len_data
            IzABS += np.sum(IzAbserr) / (args.size ** 2) / len_data

            IxRmseerr, IyRmseerr, IzRmseerr = (np.abs(outIx) - np.abs(outIx_R)) ** 2, (
                    np.abs(outIy) - np.abs(outIy_R)) ** 2, (np.abs(outIz) - np.abs(outIz_R)) ** 2
            IxRMSE += np.sqrt(np.sum(IxRmseerr) / (args.size ** 2)) / len_data
            IyRMSE += np.sqrt(np.sum(IyRmseerr) / (args.size ** 2)) / len_data
            IzRMSE += np.sqrt(np.sum(IzRmseerr) / (args.size ** 2)) / len_data

            IxDB_R, IxDB = np.clip(np.abs(outIx_R), a_min=10e-8, a_max=None), np.clip(np.abs(outIx), a_min=10e-8,
                                                                                      a_max=None)
            IxDB_R, IxDB = 10 * np.log10(IxDB_R), 10 * np.log10(IxDB)
            IDBx += np.sqrt(np.sum((IxDB_R - IxDB) ** 2) / (args.size ** 2)) / len_data
            IyDB_R, IyDB = np.clip(np.abs(outIy_R), a_min=10e-8, a_max=None), np.clip(np.abs(outIy), a_min=10e-8,
                                                                                      a_max=None)
            IyDB_R, IyDB = 10 * np.log10(IyDB_R), 10 * np.log10(IyDB)
            IDBy += np.sqrt(np.sum((IyDB_R - IyDB) ** 2) / (args.size ** 2)) / len_data
            IzDB_R, IzDB = np.clip(np.abs(outIz_R), a_min=10e-8, a_max=None), np.clip(np.abs(outIz), a_min=10e-8,
                                                                                      a_max=None)
            IzDB_R, IzDB = 10 * np.log10(IzDB_R), 10 * np.log10(IzDB)
            IDBz += np.sqrt(np.sum((IzDB_R - IzDB) ** 2) / (args.size ** 2)) / len_data

        print()
        print(f"MEAN of Ix,Iy,Iz is {IxMEAN:.6f},{IyMEAN:.6f},{IzMEAN:.6f}")
        print(f"MAE of Ix,Iy,Iz is {IxABS:.6f},{IyABS:.6f},{IzABS:.6f}")
        print(f"RMSE of Ix,Iy,Iz is {IxRMSE:.6f},{IyRMSE:.6f},{IzRMSE:.6f}")
        print(f"RMSE_DB of Ix,Iy,Iz is {IDBx:.6f},{IDBy:.6f},{IDBz:.6f}")

        data = pd.DataFrame([['IxABS', 'IyABS', 'IzABS', 'IxRMSE', 'IyRMSE', 'IzRMSE']])
        data.to_csv(f'{im_path}/A{name}_TEST.csv', mode='a', header=None, index=False)
        data = pd.DataFrame([[IxABS, IyABS, IzABS, IxRMSE, IyRMSE, IzRMSE]])
        data.to_csv(f'{im_path}/A{name}_TEST.csv', mode='a', header=None, index=False)

# 输出电流图像
def model_test(args, Gnrt, test_loader, name, step):
    test_loader = iter(test_loader)
    with torch.no_grad():
        test_sample_num = 16
        Gnrt.eval()
        currListX, currListY, currListZ = [], [], []

        for i in range(test_sample_num):
            inciAzi, inciZen, surf, curt, mask = next(test_loader)
            inciAzi, inciZen, surf, curt, mask = inciAzi.to(device), inciZen.to(device), surf.to(device), curt.to(
                device), mask.to(device)

            outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outall = Gnrt(surf, inciAzi, inciZen)

            # 4 1 280 280
            geneCurtX = [surf[[0], :, :, :], mask[[0], :, :, :], curt[[0], [0], :, :].unsqueeze(0),
                         outIxrel[[0], :, :, :],
                         curt[[0], [1], :, :].unsqueeze(0), outIximg[[0], :, :, :]]
            geneCurtY = [surf[[0], :, :, :], mask[[0], :, :, :], curt[[0], [2], :, :].unsqueeze(0),
                         outIyrel[[0], :, :, :],
                         curt[[0], [3], :, :].unsqueeze(0), outIyimg[[0], :, :, :]]
            geneCurtZ = [surf[[0], :, :, :], mask[[0], :, :, :], curt[[0], [4], :, :].unsqueeze(0),
                         outIzrel[[0], :, :, :],
                         curt[[0], [5], :, :].unsqueeze(0), outIzimg[[0], :, :, :]]

            geneCurtX = torch.cat(geneCurtX, 3).detach().cpu()
            geneCurtY = torch.cat(geneCurtY, 3).detach().cpu()
            geneCurtZ = torch.cat(geneCurtZ, 3).detach().cpu()

            currListX.append(geneCurtX)
            currListY.append(geneCurtY)
            currListZ.append(geneCurtZ)

        currListX = torch.cat(currListX, 0)
        currListY = torch.cat(currListY, 0)
        currListZ = torch.cat(currListZ, 0)
        utils.save_image(currListX, f'{im_path}/{name}_{str(step).zfill(6)}_X.jpg', normalize=True,
                         nrow=1)
        utils.save_image(currListY, f'{im_path}/{name}_{str(step).zfill(6)}_Y.jpg', normalize=True,
                         nrow=1)
        utils.save_image(currListZ, f'{im_path}/{name}_{str(step).zfill(6)}_Z.jpg', normalize=True,
                         nrow=1)
        Gnrt.train()


def model_train(args, train_loader, test_loader, test_error_loader, Gnrt, ssim_loss, G_optim, device, name):
    Gnrt.train()
    train_loader = sample_data(train_loader)

    G_scheduler = lr_scheduler.StepLR(G_optim, step_size=100000, gamma=0.5)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.1)

    loss_dict = {}
    mse_loss = nn.MSELoss(reduction='mean')
    L1_loss = nn.L1Loss(reduction='mean')
    if args.distributed:
        Gnrt_module = Gnrt.module

    else:
        Gnrt_module = Gnrt

    # 创建空CSV文件
    with open(f'{im_path}/A{name}_loss.csv', 'w') as file:
        pass
    data = pd.DataFrame([['G_loss', 'SSIM', 'L1_Loss', 'G_loss']])
    data.to_csv(f'{im_path}/A{name}_loss.csv', mode='a', header=None, index=False)

    # 迭代训练
    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')
            break

        inciAzi, inciZen, surf, curt, mask = next(train_loader)
        inciAzi, inciZen, surf, curt, mask = inciAzi.to(device), inciZen.to(device), surf.to(device), curt.to(
            device), mask.to(device)

        if i % args.d_reg_every == 0:
            curt.requires_grad = True
            inciAzi.requires_grad = True
            inciZen.requires_grad = True

        outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg, outall = Gnrt(surf, inciAzi, inciZen)
        # random_number = random.randint(0, 9)
        # if random_number > 4:
        #     outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg = outIxrel * mask, outIyrel * mask, \
        #                                                                  outIzrel * mask, outIximg * mask, outIyimg * mask, outIzimg * mask
        #     curt = curt * mask
        # else:
        #     pass
        Img_reloss = 3 - ssim_loss(outIxrel, outIximg) - ssim_loss(outIyrel, outIyimg) - ssim_loss(outIzrel, outIzimg)
        MSE_loss = mse_loss(torch.cat([outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg], dim=1), curt) ** 2
        SSIM = 6 - (ssim_loss(outIxrel, curt[:, [0], :, :]) + ssim_loss(outIximg, curt[:, [1], :, :]) + ssim_loss(
            outIyrel, curt[:, [2], :, :]) + ssim_loss(outIyimg, curt[:, [3], :, :]) + ssim_loss(outIzrel,
                                                                                                curt[:, [4], :, :])
                    + ssim_loss(outIzimg, curt[:, [5], :, :]))
        L1_Loss = L1_loss(torch.cat([outIxrel, outIyrel, outIzrel, outIximg, outIyimg, outIzimg], dim=1), curt) ** 2
        loss_dict['Img_reloss'] = Img_reloss
        loss_dict['MSE_loss'] = MSE_loss
        loss_dict['SSIM'] = SSIM
        loss_dict['L1_Loss'] = L1_Loss
        loss_dict['G_loss'] = MSE_loss + SSIM + L1_Loss

        G_loss = MSE_loss + SSIM * 10 + L1_Loss + 10 * Img_reloss ** 2
        G_optim.zero_grad()
        G_loss.backward()
        G_optim.step()
        G_scheduler.step()
        accumulate(Gnrt_ema, Gnrt_module)

        loss_reduced = reduce_loss_dict(loss_dict)
        Img_relos_final = loss_reduced['Img_reloss'].mean().item()
        G_loss_final = loss_reduced['G_loss'].mean().item()
        MSE_loss_final = loss_reduced['MSE_loss'].mean().item()
        SSIM_final = loss_reduced['SSIM'].mean().item()
        L1_Loss_final = loss_reduced['L1_Loss'].mean().item()

        if idx % 100 == 0:
            data = pd.DataFrame([[G_loss_final, MSE_loss_final, SSIM_final, L1_Loss_final]])
            data.to_csv(f'{im_path}/A{name}_loss.csv', mode='a', header=None, index=False)
            print('\n')
            print(
                f"Epoch : {idx + 1} -> Img_rel:{Img_relos_final:.2f}; G_loss: {G_loss_final:.2f};  MSE_loss: {MSE_loss_final:.2f}; SSIM: {SSIM_final:.2f};  L1_Loss: {L1_Loss_final:.2f};\n"
            )
            print("====================================================")
        if get_rank() == 0:
            pbar.set_description(
                (
                    f"Epoch : {idx + 1} -> Img_rel:{Img_relos_final:.2f}; G_loss: {G_loss_final:.2f};  MSE_loss: {MSE_loss_final:.2f}; SSIM: {SSIM_final:.2f};  L1_Loss: {L1_Loss_final:.2f};"
                )
            )

            if i % 300 == 0:
                with torch.no_grad():
                    Gnrt_copy = copy.deepcopy(Gnrt).eval().cpu()
                    ele_far_field_verify(args, Gnrt_copy, test_error_loader, 'verify', i)
                    # 模型测试
                    model_test(args, Gnrt, test_loader, 'normal', i)
                    model_test(args, Gnrt_ema, test_loader, 'ema', i)
                    model_error_statistic(args, Gnrt_copy, test_error_loader, 'error', i)

            if (i + 1) % 1000 == 0:
                torch.save(
                    {
                        'Gnrt': Gnrt_module.state_dict(),
                        'Gnrt_ema': Gnrt_ema.state_dict(),
                        'G_optim': G_optim.state_dict(),
                        'iter': i,
                    },
                    os.path.join(model_path, f'ck_{name}.pt'),
                )


if __name__ == '__main__':
    seed = 42
    torch.manual_seed(seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU，为所有GPU设置随机种子
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--iter', type=int, default=300000)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--n_sample', type=int, default=64)
    parser.add_argument('--size', type=int, default=280)
    parser.add_argument('--r1', type=float, default=10)

    # 模型加载或者不加载
    parser.add_argument('--ckpt', type=str, default=None)
    # parser.add_argument('--ckpt', type=str, default='NewModel04/checkpoint/ck_NewModel04.pt')
    # 学习率
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--num_down', type=int, default=3)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--d_path', type=str, required=True)
    # 名称 数据集位置
    args = parser.parse_args(['--name', 'NewModel04', '--d_path',
                              os.path.join(os.path.split(os.path.split(os.path.abspath(__file__))[0])[0],
                                           'ViewData')])

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = False
    # SSIM损失函数
    ssim_loss = SSIM().to(device)
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

    save_path = f'./{args.name}'
    im_path = os.path.join(save_path, 'sample')
    model_path = os.path.join(save_path, 'checkpoint')
    os.makedirs(im_path, exist_ok=True)
    os.makedirs(model_path, exist_ok=True)

    args.n_mlp = 5
    args.start_iter = 0

    # 定义模型框架
    Gnrt = A_Generator().to(device)
    Gnrt_ema = copy.deepcopy(Gnrt).to(device).eval()

    # 优化函数
    G_optim = optim.Adam(list(Gnrt.parameters()), lr=args.lr, betas=(0, 0.99))

    # 加载参数
    if args.ckpt is not None:
        # 模型参数
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        Gnrt.load_state_dict(ckpt['Gnrt'])
        G_optim.load_state_dict(ckpt['G_optim'])
        args.start_iter = ckpt['iter']

    if args.distributed:
        Gnrt = nn.parallel.DistributedDataParallel(
            Gnrt,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    # 表面纹理 的标准化
    t1 = transforms.Compose([
        transforms.Normalize(mean=mean_surf, std=std_surf, inplace=True)
    ])

    # 表面电流 的标准化
    t2 = transforms.Compose([
        transforms.Normalize(mean=mean,
                             std=std, inplace=True)
    ])

    dataset = DatasetFolder(args.d_path, transform1=t1, transform2=t2)

    train_ratio = 0.8  # 训练集比例
    test_ratio = 0.2  # 测试集比例

    # 计算切分点索引
    train_size = int(len(dataset) * train_ratio)
    test_size = len(dataset) - train_size

    # 切分数据集
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = data.DataLoader(train_dataset, batch_size=args.batch,
                                   sampler=data_sampler(train_dataset, shuffle=True, distributed=args.distributed),
                                   drop_last=True, pin_memory=True, num_workers=5)

    test_loader = data.DataLoader(test_dataset, batch_size=4, shuffle=True)
    test_error_loader = data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    # 模型训练
    model_train(args, train_loader, test_loader, test_error_loader, Gnrt, ssim_loss, G_optim, device, args.name)



