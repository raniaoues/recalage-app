# registration_model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from skimage.transform import pyramid_gaussian
from skimage.color import rgb2gray
from skimage.filters import gaussian
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class HomographyNet(nn.Module):
    def __init__(self):
        super(HomographyNet, self).__init__()
        self.B = torch.zeros(6, 3, 3, dtype=torch.float32).to(device)
        self.B[0, 0, 2] = 1.0  # Translation x
        self.B[1, 1, 2] = 1.0  # Translation y
        self.B[2, 0, 1] = 1.0  # Shearing y
        self.B[3, 1, 0] = 1.0  # Shearing x
        self.B[4, 0, 0], self.B[4, 1, 1] = 1.0, -1.0  # Flipping
        self.B[5, 1, 1], self.B[5, 2, 2] = -1.0, 1.0  # Flipping on diagonal

        self.v1 = nn.Parameter(torch.zeros(6, 1, 1, dtype=torch.float32).to(device), requires_grad=True)
        self.vL = nn.Parameter(torch.zeros(6, 1, 1, dtype=torch.float32).to(device), requires_grad=True)

    def forward(self, s):
        C = torch.sum(self.B * self.vL, 0)
        if s == 0:
            C += torch.sum(self.B * self.v1, 0)

        A = torch.eye(3, dtype=torch.float32).to(device)
        H = A.clone()

        for i in torch.arange(1, 10):
            A = torch.mm(A / i, C)
            H += A

        return H


class MINE(nn.Module):
    def __init__(self, nChannel, n_neurons=300, bsize=20, dropout_rate=0.5):
        super(MINE, self).__init__()
        self.nChannel = nChannel
        self.fc1 = nn.Linear(2 * nChannel, n_neurons)
        self.fc2 = nn.Linear(n_neurons, n_neurons)
        self.fc3 = nn.Linear(n_neurons, 1)
        self.dropout = nn.Dropout(dropout_rate)
        self.bsize = bsize

    def forward(self, x, ind):
        x = x.view(x.size()[0] * x.size()[1], x.size()[2])
        MI_lb = 0.0
        for _ in range(self.bsize):
            ind_perm = ind[torch.randperm(len(ind))]
            z1 = self.fc3(F.relu(self.dropout(self.fc2(F.relu(self.fc1(x[ind, :]))))))
            z2 = self.fc3(F.relu(self.dropout(self.fc2(F.relu(
                self.fc1(torch.cat((x[ind, 0:self.nChannel], x[ind_perm, self.nChannel:2 * self.nChannel]), 1))
            )))))
            MI_lb += torch.mean(z1) - torch.log(torch.mean(torch.exp(z2)))
        return MI_lb / self.bsize


def create_pyramids(I, J, downscale=2.0):
    if I.ndim == 2:  # Grayscale
        nChannel = 1
        I_blur = gaussian(I, sigma=1, channel_axis=None)
        J_blur = gaussian(J, sigma=1, channel_axis=None)
        pyramid_I = tuple(pyramid_gaussian(I_blur, downscale=downscale, channel_axis=None))
        pyramid_J = tuple(pyramid_gaussian(J_blur, downscale=downscale, channel_axis=None))
    else:  # RGB
        nChannel = I.shape[2]
        I_blur = gaussian(I, sigma=1, channel_axis=-1)
        J_blur = gaussian(J, sigma=1, channel_axis=-1)
        pyramid_I = tuple(pyramid_gaussian(I_blur, downscale=downscale, channel_axis=-1))
        pyramid_J = tuple(pyramid_gaussian(J_blur, downscale=downscale, channel_axis=-1))
    return pyramid_I, pyramid_J, nChannel



def prepare_multi_resolution_data(pyramid_I, pyramid_J, nChannel, L=6):
    I_lst, J_lst, h_lst, w_lst, xy_lst, ind_lst = [], [], [], [], [], []

    for s in range(L):
        I_ = torch.tensor(cv2.normalize(pyramid_I[s].astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)).to(device)
        J_ = torch.tensor(cv2.normalize(pyramid_J[s].astype(np.float32), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)).to(device)

        if nChannel > 1:
            I_lst.append(I_.permute(2, 0, 1))
            J_lst.append(J_.permute(2, 0, 1))
            h_, w_ = I_lst[s].shape[1], I_lst[s].shape[2]
            if pyramid_I[s].ndim == 3 and pyramid_I[s].shape[2] == 3:
                gray = (rgb2gray(pyramid_I[s]) * 255).astype(np.uint8)
            else:
                gray = (pyramid_I[s] * 255).astype(np.uint8)

        else:
            I_lst.append(I_)
            J_lst.append(J_)
            h_, w_ = I_lst[s].shape[0], I_lst[s].shape[1]
            if pyramid_I[s].ndim == 3 and pyramid_I[s].shape[2] == 3:
                gray = (rgb2gray(pyramid_I[s]) * 255).astype(np.uint8)
            else:
                gray = (pyramid_I[s] * 255).astype(np.uint8)

        blurred = cv2.GaussianBlur(gray, (21, 21), 0)
        edges_grayscale = cv2.dilate(cv2.Canny(blurred, 0, 30), np.ones((5, 5), np.uint8), iterations=1)

        ind_ = torch.nonzero(torch.tensor(edges_grayscale, dtype=torch.bool).view(h_ * w_)).squeeze().to(device)[:1000000]
        ind_lst.append(ind_)
        h_lst.append(h_)
        w_lst.append(w_)

        y_, x_ = torch.meshgrid(torch.arange(0, h_, dtype=torch.float32).to(device),
                                torch.arange(0, w_, dtype=torch.float32).to(device),
                                indexing='ij')
        y_, x_ = 2.0 * y_ / (h_ - 1) - 1.0, 2.0 * x_ / (w_ - 1) - 1.0
        xy_ = torch.stack([x_, y_], 2)
        xy_lst.append(xy_)

    return I_lst, J_lst, h_lst, w_lst, xy_lst, ind_lst


def AffineTransform(I, H, xv, yv):
    xvt = (xv * H[0, 0] + yv * H[0, 1] + H[0, 2]) / (xv * H[2, 0] + yv * H[2, 1] + H[2, 2])
    yvt = (xv * H[1, 0] + yv * H[1, 1] + H[1, 2]) / (xv * H[2, 0] + yv * H[2, 1] + H[2, 2])
    grid = torch.stack([xvt, yvt], 2).unsqueeze(0)
    J = F.grid_sample(I, grid, mode='bilinear', align_corners=True).squeeze()
    return J


def multi_resolution_loss(I_lst, J_lst, xy_lst, ind_lst, homography_net, mine_net, L=6, nChannel=1):
    loss = 0.0
    for s in np.arange(L - 1, -1, -1):
        if nChannel > 1:
            Jw_ = AffineTransform(J_lst[s].unsqueeze(0), homography_net(s), xy_lst[s][:, :, 0], xy_lst[s][:, :, 1]).squeeze()
            mi = mine_net(torch.cat([I_lst[s], Jw_], 0).permute(1, 2, 0), ind_lst[s])
            loss -= (1. / L) * mi
        else:
            Jw_ = AffineTransform(J_lst[s].unsqueeze(0).unsqueeze(0), homography_net(s), xy_lst[s][:, :, 0], xy_lst[s][:, :, 1]).squeeze()
            mi = mine_net(torch.stack([I_lst[s], Jw_], 2), ind_lst[s])
            loss -= (1. / L) * mi
    return loss
