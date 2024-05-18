#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import sys
from datetime import datetime
import numpy as np
import random


def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    resized_image_PIL = pil_image.resize(resolution)
    resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


class LinearNoise:
    def __init__(
            self,
            lr_init,
            lr_final,
            lr_delay_steps=0,
            lr_delay_mult=1.0,
            max_steps=1000000,
    ):
        self.lr_init = lr_init
        self.lr_final = lr_final
        self.lr_delay_steps = lr_delay_steps
        self.lr_delay_mult = lr_delay_mult
        self.max_steps = max_steps

    def __call__(self, step):
        if step < 0 or (self.lr_init == 0.0 and self.lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if self.lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = self.lr_delay_mult + (1 - self.lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / self.lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / self.max_steps, 0, 1)
        log_lerp = self.lr_init * (1 - t) + self.lr_final * t
        return delay_rate * log_lerp


def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    return LinearNoise(
        lr_init=lr_init,
        lr_final=lr_final,
        lr_delay_steps=lr_delay_steps,
        lr_delay_mult=lr_delay_mult,
        max_steps=max_steps,
    )


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def rotor2quaterion(r):
    x, y, z, w = torch.split(r, 1, dim=-1)
    quat_w = x
    quat_x = -w
    quat_y = z
    quat_z = -y
    return torch.cat([quat_w, quat_x, quat_y, quat_z], dim=-1)

def quaterion2rotor(r):
    w, x, y, z = torch.split(r, 1, dim=-1)
    rotors_x = w
    rotors_y = -z
    rotors_z = y
    rotors_w = -x
    return torch.cat([rotors_x, rotors_y, rotors_z, rotors_w], dim=-1)

def rotornorm(rotors1, rotors2, normalize_pesudo=False):

    rotors = torch.cat([rotors1, rotors2], dim=-1)

    if normalize_pesudo:

        a, bxy, bxz, byz, bxw, byw, bzw, pxyzw = torch.split(rotors, 1, dim=-1)

        eps = pxyzw * a - bxy * bzw + bxz * byw - bxw * byz

        mask = eps.abs()[:, 0] > 1e-7

        if mask.sum():

            rotors_pick = rotors[mask]
            eps = eps[mask]
            a, bxy, bxz, byz, bxw, byw, bzw, pxyzw = torch.split(rotors_pick, 1, dim=-1)

            # float l2 = a * a + bxy * bxy + bxz * bxz + byz * byz + bxw * bxw + byw * byw + bzw * bzw + pxyzw * pxyzw;
            l2 = (rotors_pick ** 2).sum(dim=-1, keepdim=True)
            delta = (torch.sqrt(l2 * l2 - 4 * eps * eps) - l2) / (2 * eps)

            da = +delta * pxyzw
            dpxyzw = +delta * a
            dbxy = -delta * bzw
            dbzw = -delta * bxy
            dbxz = +delta * byw
            dbyw = +delta * bxz
            dbyz = -delta * bxw
            dbxw = -delta * byz

            anew = a + da
            pxyzwnew = pxyzw + dpxyzw
            bxywnew = bxy + dbxy
            bxzwnew = bxz + dbxz
            byzwnew = byz + dbyz
            bxwwnew = bxw + dbxw
            bywwnew = byw + dbyw
            bzwwnew = bzw + dbzw

            rotors_new = torch.cat([anew, bxywnew, bxzwnew, byzwnew, bxwwnew, bywwnew,bzwwnew, pxyzwnew], dim=-1)
            # rotors[mask] = rotors_new
            # cannot be inplace operation
            rotors_new_full = torch.zeros_like(rotors)
            rotors_new_full[mask] = rotors_new
            rotors = torch.where(mask.reshape(-1, 1).expand(-1, 8), rotors_new_full, rotors)
    
    length = rotors.norm(dim=-1, keepdim=True)
    rotors = rotors / (1e-7 + length)

    return rotors[:, :4], rotors[:, 4:]


def build_rotation(r):

    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / (1e-8 + norm[:, None])

    R = torch.zeros((q.size(0), 3, 3), device=r.device)

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def build_rotation_from_rotor8(rotors1, rotors2):
    a = rotors1[:,0]
    s = a
    bxy = rotors1[:,1]
    bxz = rotors1[:,2]
    byz = rotors1[:,3]

    bxw = rotors2[:,0]
    byw = rotors2[:,1]
    bzw = rotors2[:,2]
    pxyzw = rotors2[:,3]

    N = a.shape[0]
    r = torch.zeros(N, 4, 4, device=a.device)

    s2 = a * a
    bxy2 = bxy * bxy
    bxz2 = bxz * bxz
    bxw2 = bxw * bxw
    byz2 = byz * byz
    byw2 = byw * byw
    bzw2 = bzw * bzw
    bxyzw2 = pxyzw * pxyzw

    r[:, 0, 0] = -bxy2 - bxz2 - bxw2 + byz2 + byw2 + bzw2 - bxyzw2 + s2
    r[:, 1, 0] = 2 * (bxy * s - bxz * byz - bxw * byw + bzw * pxyzw)
    r[:, 2, 0] = 2 * (bxy * byz + bxz * s - bxw * bzw - byw * pxyzw)
    r[:, 3, 0] = 2 * (bxy * byw + bxz * bzw + bxw * s + byz * pxyzw)

    r[:, 0, 1] = -2 * (bxy * s + bxz * byz + bxw * byw + bzw * pxyzw)
    r[:, 1, 1] = -bxy2 + bxz2 + bxw2 - byz2 - byw2 + bzw2 - bxyzw2 + s2
    r[:, 2, 1] = 2 * (-bxy * bxz + bxw * pxyzw + byz * s - byw * bzw)
    r[:, 3, 1] = 2 * (-bxy * bxw - bxz * pxyzw + byz * bzw + byw * s)

    r[:, 0, 2] = 2 * (bxy * byz - bxz * s - bxw * bzw + byw * pxyzw)
    r[:, 1, 2] = -2 * (bxy * bxz + bxw * pxyzw + byw * bzw + byz * s)
    r[:, 2, 2] = bxy2 - bxz2 + bxw2 - byz2 + byw2 - bzw2 - bxyzw2 + s2
    r[:, 3, 2] = 2 * (bxy * pxyzw - bxz * bxw - byw * byz + bzw * s)

    r[:, 0, 3] = 2 * (bxy * byw + bxz * bzw - bxw * s - byz * pxyzw)
    r[:, 1, 3] = 2 * (-bxy * bxw + bxz * pxyzw + byz * bzw - byw * s)
    r[:, 2, 3] = -2 * (bxy * pxyzw + bxz * bxw + byz * byw + bzw * s)
    r[:, 3, 3] = bxy2 + bxz2 - bxw2 + byz2 - byw2 - bzw2 - bxyzw2 + s2

    return r.permute(0, 2, 1)


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L

def build_scaling_rotation_4d(s, r1, r2):
    L = torch.zeros((s.shape[0], 4, 4), dtype=torch.float, device=s.device)
    R = build_rotation_from_rotor8(r1, r2)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]
    L[:, 3, 3] = s[:, 3]

    L = R @ L
    return L

def slice_4d(scale, r1, r2):
    L = build_scaling_rotation_4d(scale, r1, r2)
    sigma = L @ L.permute(0, 2, 1)
    
    w = 1 / sigma[:, 3,3]
    alpha = sigma[:, 0, 3] * w
    beta = sigma[:, 1,3] * w
    gamma = sigma[:, 2, 3] * w


    cov_3d_out0 = sigma[:, 0, 0] - sigma[:, 0, 3] * alpha
    cov_3d_out1 = sigma[:, 0, 1] - sigma[:, 0, 3] * beta
    cov_3d_out2 = sigma[:, 0, 2] - sigma[:, 0, 3] * gamma
    cov_3d_out3 = sigma[:, 1, 1] - sigma[:, 1, 3] * beta
    cov_3d_out4 = sigma[:, 1, 2] - sigma[:, 1, 3] * gamma
    cov_3d_out5 = sigma[:, 2, 2] - sigma[:, 2, 3] * gamma
    cov_3d_out = torch.stack([cov_3d_out0, cov_3d_out1, cov_3d_out2, cov_3d_out3, cov_3d_out4, cov_3d_out5], dim=-1)
    
    speed = torch.stack([alpha, beta, gamma], dim=-1)
    
    return cov_3d_out, speed, w


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


if __name__ == '__main__':
    k = 100
    rotors1 = torch.rand(100, 4)
    rotors2 = torch.zeros_like(rotors1)
    rotors2 = torch.rand(100, 4)
    rotors1, rotors2 = rotornorm(rotors1, rotors2, True)

    q1 = rotor2quaterion(rotors1)
    R1 = build_rotation(q1)
    R2 = build_rotation_from_rotor8(rotors1, rotors2)

    print(R1[0])
    print(R2[0])
    print((R1 - R2[:, :3, :3]).abs().max())

    scale = torch.rand(k, 4) + 0.5

    slice_4d(scale, rotors1, rotors2)