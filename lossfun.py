import torch
import torch.nn.functional as F
from torch import nn, autograd
from math import exp


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)

def d_logistic_loss(real_pred, fake_pred):
    loss = 0
    for real, fake in zip(real_pred, fake_pred):
        real_loss = F.softplus(-real)
        fake_loss = F.softplus(fake)
        loss += real_loss.mean() + fake_loss.mean()

    return loss


def d_r1_loss(real_pred, real_img):
    grad_penalty = 0
    for real in real_pred:
        grad_real, = autograd.grad(
            outputs=real.mean(), inputs=real_img, create_graph=True, only_inputs=True
        )
        grad_penalty += grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty

def g_nonsaturating_loss(fake_pred, weights):
    loss = 0
    for fake, weight in zip(fake_pred, weights):
        loss += weight*F.softplus(-fake).mean()

    return loss / len(fake_pred)



