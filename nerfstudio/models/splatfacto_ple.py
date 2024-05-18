# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
NeRF implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import os.path

import numpy as np
import torch
#from gsplat._torch_impl import quat_to_rotmat
#from gsplat.project_gaussians import project_gaussians
#from gsplat.project_gaussians_4d import ProjectGaussians4D
#from gsplat.rasterize import RasterizeGaussians
#from gsplat.rasterize import rasterize_gaussians
from simple_knn._C import distCUDA2
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

#from nerfstudio.utils.sh import num_sh_bases
from pytorch_msssim import SSIM
from torch.nn import Parameter
from typing_extensions import Literal

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.data.scene_box import OrientedBox
from nerfstudio.engine.callbacks import TrainingCallback, TrainingCallbackAttributes, TrainingCallbackLocation
from nerfstudio.engine.optimizers import Optimizers

# need following import for background color override
from nerfstudio.model_components import renderers
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation_from_rotor8, strip_symmetric, \
    build_scaling_rotation, quaterion2rotor, rotornorm, slice_4d
from nerfstudio.utils.sh_utils import RGB2SH
from nerfstudio.utils.sh_utils import eval_sh
from nerfstudio.utils.ssim import windowed_pearson
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.ops.ball_query import ball_query
from pdb import set_trace
import knn_ops
from typing import NamedTuple
from plyfile import PlyData, PlyElement
from nerfstudio.data.dataparsers.plenoptic_dataparser import Plenoptic

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array


def fetch_ply(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def num_sh_bases(degree: int):
    if degree == 0:
        return 1
    if degree == 1:
        return 4
    if degree == 2:
        return 9
    if degree == 3:
        return 16
    return 25
def knn_fast(points_rescale_pxk, num_bin_eachdim, num_knn):
    num_points = points_rescale_pxk.shape[0]
    num_dim = points_rescale_pxk.shape[1]
    num_total_bin = num_bin_eachdim ** num_dim

    xyzt_min = points_rescale_pxk.min(dim=0)[0].reshape(1, num_dim)
    xyzt_max = points_rescale_pxk.max(dim=0)[0].reshape(1, num_dim)
    bins = (points_rescale_pxk - xyzt_min) / (xyzt_max - xyzt_min + 1e-5) * num_bin_eachdim
    bins = torch.floor(bins).long()
    bins_base = torch.Tensor([num_bin_eachdim ** k for k in range(num_dim)]).to(bins)
    idx_bin = (bins * bins_base).sum(dim=-1, keepdim=True)
    total_bin_count = torch.histogram(idx_bin[:, 0].cpu().float(), torch.arange(num_bin_eachdim ** num_dim + 1).float())
    num_elements_in_each_bin = total_bin_count.hist.cuda().long()
    num_bin_max = num_elements_in_each_bin.max().item()

    # reorder points into bins
    points_reorder, points_reorder_idx = knn_ops.reorder_data_fw(points_rescale_pxk, idx_bin[:, 0].int(), num_bin_eachdim**num_dim, num_bin_max)
    # run knn in each bin
    knnre4d = knn_points(points_reorder, points_reorder, lengths1=num_elements_in_each_bin, lengths2=num_elements_in_each_bin, norm=2, K=num_knn, return_nn=False, return_sorted=False)
    knnidx_bxpx8 = knnre4d.idx
    knndists_bxpx8 = knnre4d.dists
    # binmax
    bin_max_bx1x1 = num_elements_in_each_bin.reshape(-1, 1, 1)
    # change them back
    knndist_kx8, knnmask_kx8, knnidx_kx8 = knn_ops.reorder_data_bw(knnidx_bxpx8.int(), num_elements_in_each_bin.int(), knndists_bxpx8, points_reorder_idx, num_points)
    torch.cuda.empty_cache()
    return knnidx_kx8, knndist_kx8, knnmask_kx8



def random_quat_tensor(N):
    """
    Defines a random quaternion tensor of shape (N, 4)
    """
    u = torch.rand(N)
    v = torch.rand(N)
    w = torch.rand(N)
    return torch.stack(
        [
            torch.sqrt(1 - u) * torch.sin(2 * math.pi * v),
            torch.sqrt(1 - u) * torch.cos(2 * math.pi * v),
            torch.sqrt(u) * torch.sin(2 * math.pi * w),
            torch.sqrt(u) * torch.cos(2 * math.pi * w),
        ],
        dim=-1,
    )


def RGB2SH(rgb):
    """
    Converts from RGB values [0,1] to the 0th spherical harmonic coefficient
    """
    C0 = 0.28209479177387814
    return (rgb - 0.5) / C0


def SH2RGB(sh):
    """
    Converts from the 0th spherical harmonic coefficient to RGB values [0,1]
    """
    C0 = 0.28209479177387814
    return sh * C0 + 0.5


def projection_matrix(znear, zfar, fovx, fovy, device: Union[str, torch.device] = "cpu"):
    """
    Constructs an OpenGL-style perspective projection matrix.
    """
    t = znear * math.tan(0.5 * fovy)
    b = -t
    r = znear * math.tan(0.5 * fovx)
    l = -r
    n = znear
    f = zfar
    P = torch.tensor(
        [
            [2 * n / (r - l), 0.0, (r + l) / (r - l), 0.0],
            [0.0, 2 * n / (t - b), (t + b) / (t - b), 0.0],
            [0.0, 0.0, f / (f - n), -1.0 * f * n / (f - n)],
            [0.0, 0.0, 1.0, 0.0],
        ],
        device=device,
    )
    projection = torch.transpose(P, 0, 1)
    return projection
    


@dataclass
class Splatfacto_pleModelConfig(ModelConfig):
    """Splatfacto Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: Splatfacto_pleModel)
    warmup_length: int = 500
    """period of steps where refinement is turned off"""
    refine_every: int = 100
    """period of steps where gaussians are culled and densified"""
    resolution_schedule: int = 250
    """training starts at 1/d resolution, every n steps this is doubled"""
    background_color: Literal["random", "black", "white"] = "black"
    """Whether to randomize the background color."""
    num_downscales: int = 1
    """at the beginning, resolution is 1/2^d, where d is this number"""
    cull_alpha_thresh: float = 0.1
    """threshold of opacity for culling gaussians. One can set it to a lower value (e.g. 0.005) for higher quality."""
    cull_scale_thresh: float = 0.5
    """threshold of scale for culling huge gaussians"""
    continue_cull_post_densification: bool = False
    """If True, continue to cull gaussians post refinement"""
    reset_alpha_every: int = 30
    """Every this many refinement steps, reset the alpha"""
    densify_grad_thresh: float = 0.0002
    """threshold of positional gradient norm for densifying gaussians"""
    densify_size_thresh: float = 0.01
    """below this size, gaussians are *duplicated*, otherwise split"""
    n_split_samples: int = 2
    """number of samples to split gaussians into"""
    sh_degree_interval: int = 1000
    """every n intervals turn on another sh degree"""
    cull_screen_size: float = 0.15
    """if a gaussian is more than this percent of screen space, cull it"""
    split_screen_size: float = 0.05
    """if a gaussian is more than this percent of screen space, split it"""
    stop_screen_size_at: int = 3000
    """stop culling/splitting at this step WRT screen size of gaussians"""
    random_init: bool = True
    """whether to initialize the positions uniformly randomly (not SFM points)"""
    num_random: int = 100000
    """Number of gaussians to initialize if random init is used"""
    random_scale: float = 3.0
    "Size of the cube to initialize random gaussians within"
    ssim_lambda: float = 0.2
    """weight of ssim loss"""
    stop_split_at: int = 15000
    """stop splitting at this step"""
    sh_degree: int = 3
    """maximum degree of spherical harmonics to use"""
    use_scale_regularization: bool = False
    """If enabled, a scale regularization introduced in PhysGauss (https://xpandora.github.io/PhysGaussian/) is used for reducing huge spikey gaussians."""
    max_gauss_ratio: float = 10.0
    batch_reg: int = 0
    #batch_mode 
    """threshold of ratio of gaussian max to min scale before applying regularization
    loss from the PhysGaussian paper
    """
    percent_dense_xy : float = 0.01
    percent_dense_t : float = 0.2
    temporal_extent : float = 10.
    grad_threshold : float = 2e-4
    grad_threshold_t : float =2e-4
    output_depth_during_training: bool = False
    densify_from_iter: int = 500
    """If True, output depth during training. Otherwise, only output depth during evaluation."""
    rasterize_mode: Literal["classic", "antialiased"] = "classic"

    path:str = 'data/N3V/$scene_name$'
    """
    Classic mode of rendering will use the EWA volume splatting with a [0.3, 0.3] screen space blurring kernel. This
    approach is however not suitable to render tiny gaussians at higher or lower resolution than the captured, which
    results "aliasing-like" artifacts. The antialiased mode overcomes this limitation by calculating compensation factors
    and apply them to the opacities of gaussians to preserve the total integrated density of splats.

    However, PLY exported with antialiased rasterize mode is not compatible with classic mode. Thus many web viewers that
    were implemented for classic mode can not render antialiased mode PLY properly without modifications.
    """


class Splatfacto_pleModel(Model):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto configuration to instantiate model
    """

    config: SplatfactoModelConfig

    def __init__(
        self,
        *args,
        seed_points: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs,
    ):
        self.seed_points = seed_points
        super().__init__(*args, **kwargs)

    def populate_modules(self):
        if self.seed_points is not None and not self.config.random_init:
            xyz = torch.nn.Parameter(self.seed_points[0])  # (Location, Color)
            t =  torch.nn.Parameter(torch.rand_like(xyz[:, :1]))
        else:
            ply_path = os.path.join(self.config.path, "points3d.ply")
            pcd = fetch_ply(ply_path)
            xyz = torch.nn.Parameter(torch.tensor(pcd.points)).repeat(1, 1)
            fused_color =  RGB2SH(torch.tensor(np.asarray(pcd.colors/255)).float().cuda())
            #xyz = torch.nn.Parameter(torch.tensor(np.random.uniform(low=np.array([-38, -21, 5]), high=np.array([16, 12, 24]), size=(self.config.num_random, 3)), dtype=torch.float32))
            t =  torch.nn.Parameter(torch.rand_like(xyz[:, :1]) * (10  * 1.2 - 1))
        self.max_2Dsize = torch.zeros((xyz.shape[0])).cuda()
        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(xyz.detach().numpy())).float().cuda()), 0.0000001).cuda()
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        scales = scales.repeat(1, 1)
        scales_xyz = torch.nn.Parameter(scales.requires_grad_(True))
        scales_t = torch.nn.Parameter(torch.log(torch.sqrt(torch.ones_like(scales[:, :1]) * 0.2 * 10)))
        num_points = xyz.shape[0]
        rots = torch.zeros((num_points, 4)).cuda()
        rots[:, 0] = 1
        quats1=quaterion2rotor(rots)
        quats2=torch.zeros_like(quats1)
        dim_sh = num_sh_bases(self.config.sh_degree)
        self.active_sh_degree =-1
        if (
            self.seed_points is not None
            and not self.config.random_init
            # We can have colors without points.
            and self.seed_points[1].shape[0] > 0
        ):
            shs = torch.zeros((self.seed_points[1].shape[0], dim_sh, 3)).float().cuda()
            if self.config.sh_degree > 0:
                shs[:, 0, :3] = RGB2SH(self.seed_points[1] / 255)
                shs[:, 1:, 3:] = 0.0
            else:
                CONSOLE.log("use color only optimization with sigmoid activation")
                shs[:, 0, :3] = torch.logit(self.seed_points[1] / 255, eps=1e-10)
            features_dc = torch.nn.Parameter(shs[:, 0, :])
            features_rest = torch.nn.Parameter(shs[:, 1:, :])
        else:
            #rgb = np.ones((num_points, 3), dtype=np.uint8) * 127/255
            #rgb = np.zeros((num_points, 3), dtype=np.uint8)
            #fused_color = RGB2SH(torch.tensor(np.asarray(rgb)).float().cuda())
            features = torch.zeros((fused_color.shape[0], 3, (self.config.sh_degree + 1) ** 2)).float().cuda()
            features[:, :3, 0] = fused_color
            features[:, 3:, 1:] = 0.0
            features = features.repeat(1, 1, 1)
            features_dc = torch.nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
            features_rest = torch.nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))

        opacities = torch.nn.Parameter(inverse_sigmoid(0.1 * torch.ones(num_points, 1)))
        self.entropyloss = 0
        self.knnloss = 0
        self.downsize = 1
        self.xyz_gradient_accum = torch.zeros((num_points, 1)).cuda()
        self.denom = torch.zeros((num_points, 1)).cuda()

        self.t_gradient_accum = torch.zeros((num_points, 1)).cuda()
        self.t_denom = torch.zeros((num_points, 1)).cuda()
        if "salmon" in self.config.path:
            self.config.stop_split_at = 10000


        self.gauss_params = torch.nn.ParameterDict(
            {
                "xyz": xyz,
                "t" : t,
                "scales_xyz": scales_xyz,
                "scales_t" : scales_t,
                "quats1": quats1,
                "quats2": quats2,
                "features_dc": features_dc,
                "features_rest": features_rest,
                "opacities": opacities,
            }
        )

        # metrics
        from torchmetrics.image import PeakSignalNoiseRatio
        from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = SSIM(data_range=1.0, size_average=True, channel=3)
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.crop_box: Optional[OrientedBox] = None
        if self.config.background_color == "random":
            self.background_color = torch.tensor(
                [0.1490, 0.1647, 0.2157]
            )  # This color is the same as the default background color in Viser. This would only affect the background color when rendering.
        else:
            self.background_color = get_color(self.config.background_color)
        self.batchflag = False

    @property
    def colors(self):
        if self.config.sh_degree > 0:
            return SH2RGB(self.features_dc)
        else:
            return torch.sigmoid(self.features_dc)

    @property
    def shs_0(self):
        return self.features_dc

    @property
    def shs_rest(self):
        return self.features_rest

    @property
    def num_points(self):
        return self.xyz.shape[0]

    @property
    def xyz(self):
        return self.gauss_params["xyz"]
    
    @property
    def t(self):
        return self.gauss_params["t"]
    
    @property
    def xyzt(self):
        return torch.cat([self.xyz, self.t], dim=-1)
    

    @property
    def scales_xyz(self):
        return self.gauss_params["scales_xyz"]
    
    @property
    def scales_t(self):
        return self.gauss_params["scales_t"]
    
    @property
    def scales(self):
        scales_xyz=self.scales_xyz
        scales_t=self.scales_t
        return torch.exp(torch.cat([scales_xyz, scales_t], dim=-1))

    @property
    def quats1(self):
        return self.gauss_params["quats1"]
    @property
    def quats2(self):
        return self.gauss_params["quats2"]
    
    @property
    def get_rotors(self):
        return rotornorm(self.quats1, self.quats2, True)

    @property
    def features_dc(self):
        return self.gauss_params["features_dc"]

    @property
    def features_rest(self):
        return self.gauss_params["features_rest"]
    
    @property
    def opacities(self):
        return self.gauss_params["opacities"]

    @property
    def get_opacities(self):
        return torch.sigmoid(self.opacities)
    
    

    def load_state_dict(self, dict, **kwargs):  # type: ignore
        # resize the parameters to match the new number of points
        self.step = 30000
        if "xyz" in dict:
            # For backwards compatibility, we remap the names of parameters from
            # means->gauss_params.means since old checkpoints have that format
            for p in ["xyz","t", "scales_xyz","scales_t" ,"quats1","quats2", "features_dc", "features_rest", "opacities"]:
                dict[f"gauss_params.{p}"] = dict[p]
        newp = dict["gauss_params.xyz"].shape[0]
        for name, param in self.gauss_params.items():
            old_shape = param.shape
            new_shape = (newp,) + old_shape[1:]
            self.gauss_params[name] = torch.nn.Parameter(torch.zeros(new_shape, device=self.device))
        super().load_state_dict(dict, **kwargs)

    def k_nearest_sklearn(self, x: torch.Tensor, k: int):
        """
            Find k-nearest neighbors using sklearn's NearestNeighbors.
        x: The data tensor of shape [num_samples, num_features]
        k: The number of neighbors to retrieve
        """
        # Convert tensor to numpy array
        x_np = x.cpu().numpy()

        # Build the nearest neighbors model
        from sklearn.neighbors import NearestNeighbors

        nn_model = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", metric="euclidean").fit(x_np)

        # Find the k-nearest neighbors
        distances, indices = nn_model.kneighbors(x_np)

        # Exclude the point itself from the result and return
        return distances[:, 1:].astype(np.float32), indices[:, 1:].astype(np.float32)

    def remove_from_optim(self, optimizer, deleted_mask, new_params):
        """removes the deleted_mask from the optimizer provided"""
        assert len(new_params) == 1
        # assert isinstance(optimizer, torch.optim.Adam), "Only works with Adam"

        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        del optimizer.state[param]

        # Modify the state directly without deleting and reassigning.
        if "exp_avg" in param_state:
            param_state["exp_avg"] = param_state["exp_avg"][~deleted_mask]
            param_state["exp_avg_sq"] = param_state["exp_avg_sq"][~deleted_mask]

        # Update the parameter in the optimizer's param group.
        del optimizer.param_groups[0]["params"][0]
        del optimizer.param_groups[0]["params"]
        optimizer.param_groups[0]["params"] = new_params
        optimizer.state[new_params[0]] = param_state

    def remove_from_all_optim(self, optimizers, deleted_mask):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.remove_from_optim(optimizers.optimizers[group], deleted_mask, param)
      
        torch.cuda.empty_cache()

    def dup_in_optim(self, optimizer, dup_mask, new_params, n=2):
        """adds the parameters to the optimizer"""
        param = optimizer.param_groups[0]["params"][0]
        param_state = optimizer.state[param]
        if "exp_avg" in param_state:
            repeat_dims = (n,) + tuple(1 for _ in range(param_state["exp_avg"].dim() - 1))
            param_state["exp_avg"] = torch.cat(
                [
                    param_state["exp_avg"],
                    torch.zeros_like(param_state["exp_avg"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
            param_state["exp_avg_sq"] = torch.cat(
                [
                    param_state["exp_avg_sq"],
                    torch.zeros_like(param_state["exp_avg_sq"][dup_mask.squeeze()]).repeat(*repeat_dims),
                ],
                dim=0,
            )
        del optimizer.state[param]
        optimizer.state[new_params[0]] = param_state
        optimizer.param_groups[0]["params"] = new_params
        del param

    def dup_in_all_optim(self, optimizers, dup_mask, n):
        param_groups = self.get_gaussian_param_groups()
        for group, param in param_groups.items():
            self.dup_in_optim(optimizers.optimizers[group], dup_mask, param, n)

    def after_train(self, step: int):
        assert step == self.step
        # to save some training time, we no longer need to update those stats post refinement
        
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = self.render['visibility_filter']
            visibility_filter_batch = self.render['mask_t']
            if self.batchflag:
                assert(isinstance(self.render['radii'], list))
                radii_batch = torch.stack(self.render['radii'], dim=-1).max(dim=-1)[0]
                self.max_2Dsize[visibility_filter_batch] = torch.max(
                    self.max_2Dsize[visibility_filter_batch],
                    radii_batch[visibility_filter_batch]
                )
            else:
                self.max_2Dsize[visible_mask] = torch.maximum(
                    self.max_2Dsize[visible_mask],
                    self.render['radii'][visible_mask]
                )

            if  self.batchflag:
                xyscreen, scale = self.render['viewspace_points']
                assert isinstance(xyscreen, list)
                assert isinstance(visible_mask, list)
                update_filter_batch = torch.stack(visible_mask, dim=-1).any(dim=-1)

                grads_scale = [d.grad[update_filter_batch, :2].detach() * scale for d in xyscreen]
                grads_norm = [d.norm(dim=-1, keepdim=True) for d in grads_scale]
                grads_max = torch.cat(grads_norm, dim=-1).max(dim=-1)[0]
                grads = grads_max[:, None] 
                gradsnorm = torch.cat([d.norm(dim=-1, keepdim=True) for d in grads_scale],dim = -1)
                self.xyz_gradient_accum[update_filter_batch] = grads + self.xyz_gradient_accum[update_filter_batch]
                self.denom[update_filter_batch] += 1
            #assert self.xys.grad is not None
            else:
                grad_this_iter = self.render['viewspace_points'][0].grad[visible_mask,:2].detach()
                scale=self.render['viewspace_points'][1]
                grad_this_iter = grad_this_iter * scale
                grads = grad_this_iter.norm(dim=-1, keepdim=True)
                self.denom[visible_mask] += 1
                self.xyz_gradient_accum[visible_mask]= grads + self.xyz_gradient_accum[visible_mask]
                    
            if self.batchflag:
                # batched version
                assert isinstance(visible_mask, list)
                visible_mask = torch.stack(visible_mask, dim=-1).any(dim=-1)
            grad_this_iter_t = self.t.grad[visible_mask, :1].detach()
            self.t_gradient_accum[visible_mask] = grad_this_iter_t + self.t_gradient_accum[visible_mask]
            self.t_denom[visible_mask] += 1
        torch.cuda.empty_cache()    
           
            
            # update the max screen size, as a ratio of number of pixels
            
    def set_crop(self, crop_box: Optional[OrientedBox]):
        self.crop_box = crop_box

    def set_background(self, background_color: torch.Tensor):
        assert background_color.shape == (3,)
        self.background_color = background_color

    def refinement_after(self, optimizers: Optimizers, step):
        assert step == self.step
        if self.step <= self.config.warmup_length:
            return
        with torch.no_grad():
            # Offset all the opacity reset logic by refine_every so that we don't
            # save checkpoints right when the opacity is reset (saves every 2k)
            # then cull
            # only split/cull if we've seen every image since opacity reset
            reset_interval = self.config.reset_alpha_every * self.config.refine_every  #3000
            do_densification = (
                self.step < self.config.stop_split_at
                and (self.step % self.config.refine_every ==0)
            )
            
            if do_densification:
                # then we densify
               
                nsamps = self.config.n_split_samples
                
                grads= self.xyz_gradient_accum / self.denom
                grads[grads.isnan()] = 0.0
                grads_t= self.t_gradient_accum / self.t_denom
                grads_t[grads_t.isnan()] = 0.0
                
                n_init_points = self.num_points
                # Extract points that satisfy the gradient condition
                padded_grad = torch.zeros((n_init_points), device=self.xyz.device)
                padded_grad[:grads.shape[0]] = grads.squeeze()

                selected_pts_mask_gradxy = torch.where(padded_grad >= self.config.grad_threshold, True, False)
                selected_pts_mask_scalexyz = torch.max(self.scales[:, :3], dim=1).values > self.config.percent_dense_xy * self.scene_extent
                selected_pts_mask_xy = torch.logical_and(selected_pts_mask_gradxy, selected_pts_mask_scalexyz)
        
                padded_grad_t = torch.zeros((n_init_points)).cuda()
                padded_grad_t[:grads_t.shape[0]] = grads_t.abs().squeeze()
                selected_pts_mask_gradt = torch.where(padded_grad_t >= self.config.grad_threshold_t, True, False)
                selected_pts_mask_scalet = self.scales[:, 3] > self.config.percent_dense_t * self.config.temporal_extent
                selected_pts_mask_t = torch.logical_and(selected_pts_mask_gradt, selected_pts_mask_scalet)

                splits = torch.logical_or(selected_pts_mask_xy, selected_pts_mask_t)
                split_params = self.split_gaussians(splits, nsamps)
                
                selected_pts_mask_scalexyz = torch.max(self.scales[:, :3], dim=1).values <= self.config.percent_dense_xy * self.scene_extent
                selected_pts_mask_xy = torch.logical_and(selected_pts_mask_gradxy, selected_pts_mask_scalexyz)

                selected_pts_mask_gradt = torch.where(grads_t.abs().squeeze() >= self.config.grad_threshold_t, True, False)
                selected_pts_mask_scalet = self.scales[:, 3] <= self.config.percent_dense_t * self.config.temporal_extent
                selected_pts_mask_t = torch.logical_and(selected_pts_mask_gradt, selected_pts_mask_scalet)

                dups = torch.logical_or(selected_pts_mask_xy, selected_pts_mask_t)
                dup_params = self.dup_gaussians(dups)
                
                for name, param in self.gauss_params.items():
                    self.gauss_params[name] = torch.nn.Parameter(
                        torch.cat([param.detach(), split_params[name], dup_params[name]], dim=0)
                    ) 
                # append zeros to the max_2Dsize tensor
                self.max_2Dsize = torch.cat(
                    [
                        self.max_2Dsize,
                        torch.zeros_like(split_params["scales_xyz"][:, 0]),
                        torch.zeros_like(dup_params["scales_xyz"][:, 0]),
                    ],
                    dim=0,
                )
                split_idcs = torch.where(splits)[0]
                self.dup_in_all_optim(optimizers, split_idcs, nsamps)

                dup_idcs = torch.where(dups)[0]
                self.dup_in_all_optim(optimizers, dup_idcs, 1)

                # After a guassian is split into two new gaussians, the original one should also be pruned.
                splits_mask = torch.cat(
                    (
                        splits,
                        torch.zeros(
                            nsamps * splits.sum()+dups.sum(),
                            device=self.device,
                            dtype=torch.bool,
                        ),
                    )
                )

                deleted_mask = self.cull_gaussians(splits_mask)
            else:
                # if we donot allow culling post refinement, no more gaussians will be pruned.
                deleted_mask = None
            if deleted_mask is not None:
                self.remove_from_all_optim(optimizers, deleted_mask)
            
                if self.step % reset_interval == 0 or \
                    (
                        (self.config.background_color == 'white') and self.step == self.config.densify_from_iter
                    ):
                    # Reset value is set to be twice of the cull_alpha_thresh
                   
                    minvalue=torch.min(self.get_opacities, torch.ones_like(self.get_opacities) * 0.01)
                    self.gauss_params['opacities'] = inverse_sigmoid(minvalue)
                    # reset the exp of optimizer
                    optim = optimizers.optimizers["opacities"]
                    param = optim.param_groups[0]["params"][0]
                    param_state = optim.state[param]
                    param_state["exp_avg"] = torch.zeros_like(param_state["exp_avg"])
                    param_state["exp_avg_sq"] = torch.zeros_like(param_state["exp_avg_sq"])
                    torch.cuda.empty_cache()
            self.xyz_gradient_accum = torch.zeros((self.num_points, 1), device=self.xyz.device)
            self.denom = torch.zeros((self.num_points, 1), device=self.xyz.device)

            self.t_gradient_accum = torch.zeros((self.num_points, 1), device=self.xyz.device)
            self.t_denom = torch.zeros((self.num_points, 1), device=self.xyz.device)

            self.max_2Dsize = torch.zeros((self.num_points), device=self.xyz.device)
            torch.cuda.empty_cache()
            


    def cull_gaussians(self, extra_cull_mask: Optional[torch.Tensor] = None):
        """
        This function deletes gaussians with under a certain opacity threshold
        extra_cull_mask: a mask indicates extra gaussians to cull besides existing culling criterion
        """
        n_bef = self.num_points
        # cull transparent ones
        culls = (self.get_opacities < 0.005).squeeze()
        below_alpha_count = torch.sum(culls).item()
        toobigs_count = 0
        if extra_cull_mask is not None:
            culls = culls | extra_cull_mask
        # cull huge ones
        #toobigs_t= self.scales[:, 3] > 0.8 * 1 # all the time scale
        if self.step < self.config.stop_screen_size_at:
            # cull big screen space
            if "coffee" in self.config.path or "steak" in self.config.path or  "salmon" in self.config.path:
                size = 20000
            else:
                size = 10000
            assert self.max_2Dsize is not None
            toobigs = (self.max_2Dsize >size).squeeze()
            toobigs_ws = (self.scales[:, :3].max(dim=1).values > 0.1 * self.scene_extent).squeeze()
            toobigs=toobigs|toobigs_ws
            culls = culls | toobigs
            toobigs_count = torch.sum(toobigs).item()
        for name, param in self.gauss_params.items():
            self.gauss_params[name] = torch.nn.Parameter(param[~culls])
      

        CONSOLE.log(
            f"Culled {n_bef - self.num_points} gaussians "
            f"({below_alpha_count} below alpha thresh, {toobigs_count} too bigs, {self.num_points} remaining)"
        )
        torch.cuda.empty_cache()

        return culls

    def split_gaussians(self, split_mask,samps):
        """
        This function splits gaussians that are too large
        """
        n_splits = split_mask.sum().item()
        CONSOLE.log(f"Splitting {split_mask.sum().item()/self.num_points} gaussians: {n_splits}/{self.num_points}")
        centered_samples = torch.randn((samps * n_splits, 3), device=self.device)  # Nx3 of axis-aligned scales
        stds4d = self.scales[split_mask][:, 0:4].repeat(samps, 1)
        means4d = torch.zeros((stds4d.size(0), 4), device=self.xyz.device)
        scaled_samples = torch.normal(mean=means4d, std=stds4d)
        #quats = self.quats[split_mask] / self.quats[split_mask].norm(dim=-1, keepdim=True)  # normalize them first
        #rots = quat_to_rotmat(quats.repeat(samps, 1))  # how these scales are rotated
        _rotors1, _rotors2 = self.quats1,self.quats2
        rots = build_rotation_from_rotor8 (_rotors1[split_mask], _rotors2[split_mask]).repeat(samps, 1, 1)
        rotated_samples = torch.bmm(rots, scaled_samples.unsqueeze(-1)).squeeze(-1)
        new_xyzt = rotated_samples + self.xyzt[split_mask].repeat(samps, 1)
        new_xyz = new_xyzt[:, :3]
        new_t = new_xyzt[:, 3:4]
        # step 2, sample new colors
        new_features_dc = self.features_dc[split_mask].repeat(samps, 1, 1)
        new_features_rest = self.features_rest[split_mask].repeat(samps, 1, 1)
        # step 3, sample new opacities
        new_opacities = self.opacities[split_mask].repeat(samps, 1)
        # step 4, sample new scales
        size_fac = 1.6
        new_scales = torch.log(self.scales[split_mask] / size_fac).repeat(samps, 1)
        new_scale_xyz = new_scales[:,:3]
        new_scale_t = new_scales[:,3:4]
        # step 5, sample new quats
        new_quats1 = self.quats1[split_mask].repeat(samps, 1)
        new_quats2 = self.quats2[split_mask].repeat(samps, 1)
        out = {
            "xyz": new_xyz,
            "t" : new_t,
            "features_dc": new_features_dc,
            "features_rest": new_features_rest,
            "opacities": new_opacities,
            "scales_xyz": new_scale_xyz,
            "scales_t": new_scale_t,
            "quats1": new_quats1,
            "quats2": new_quats2,
        }
        for name, param in self.gauss_params.items():
            if name not in out:
                out[name] = param[split_mask].repeat(samps, 1)
      
        torch.cuda.empty_cache()
        return out

    def dup_gaussians(self, dup_mask):
        """
        This function duplicates gaussians that are too small
        """
        n_dups = dup_mask.sum().item()
        CONSOLE.log(f"Duplicating {dup_mask.sum().item()/self.num_points} gaussians: {n_dups}/{self.num_points}")
        new_dups = {}
        for name, param in self.gauss_params.items():
            new_dups[name] = param[dup_mask]
        torch.cuda.empty_cache()
        return new_dups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        cbs = []
        cbs.append(TrainingCallback([TrainingCallbackLocation.BEFORE_TRAIN_ITERATION], self.step_cb))
        # The order of these matters
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.after_train,
            )
        )
        cbs.append(
            TrainingCallback(
                [TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                self.refinement_after,
                update_every_num_iters=self.config.refine_every,
                args=[training_callback_attributes.optimizers],
            )
        )
        return cbs

    def step_cb(self, step):
        self.step = step

    def get_gaussian_param_groups(self) -> Dict[str, List[Parameter]]:
        # Here we explicitly use the means, scales as parameters so that the user can override this function and
        # specify more if they want to add more optimizable params to gaussians.
        return {
            name: [self.gauss_params[name]]
            for name in ["xyz", "t","scales_xyz","scales_t", "quats1","quats2", "features_dc", "features_rest", "opacities"]
        }

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        """Obtain the parameter groups for the optimizers

        Returns:
            Mapping of different parameter groups
        """
        gps = self.get_gaussian_param_groups()
        return gps

    def _get_downscale_factor(self):
        if self.training:
            return 2 ** max(
                (self.config.num_downscales - self.step // self.config.resolution_schedule),
                0,
            )
        else:
            return 1

    def _downscale_if_required(self, images):
        #d = self._get_downscale_factor()
        d = self.downsize

        if len(images.shape) == 4:
            if d > 1:
                new_size = [images.shape[1] // d, images.shape[2] // d]

                # torchvision can be slow to import, so we do it lazily.
                import torchvision.transforms.functional as TF

                downscaled_images = []
                for image in images:
                    downscaled_image = TF.resize(image.permute(2, 0, 1), new_size, antialias=None).permute(1, 2, 0)
                    downscaled_images.append(downscaled_image)
                return torch.stack(downscaled_images)
            return images
            
        else:
            if d > 1:
                newsize = [images.shape[0] // d, images.shape[1] // d]

                # torchvision can be slow to import, so we do it lazily.
                import torchvision.transforms.functional as TF

                return TF.resize(images.permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
            return images

        
        
    def forward_1(self,camera,background):
        camera_to_world=camera.camera_to_worlds[0]
        camera_to_world[:3, 1:3] *= -1
        world_to_camera = torch.linalg.inv(camera_to_world).to(torch.float)
        R = world_to_camera[:3, :3]
        #R[0],R[1] = R[1].clone() , R[0].clone()
        T = world_to_camera[:3, 3]
        
        viewmat = torch.zeros((4, 4))
        viewmat[:3,:3] = R
        viewmat[:3,3] = T
        viewmat[3,3] = 1.
        viewmat = torch.transpose(viewmat, 0, 1).cuda()
        
        camera_center = torch.linalg.inv(viewmat)[3, :3]
        self.scene_extent = float(camera.camera_extent)
        # calculate the FOV of the camera given fx and fy, width and height
        cx = camera.cx.item()
        cy = camera.cy.item()
        fovx = 2 * math.atan(camera.width / (2 * camera.fx))
        fovy = 2 * math.atan(camera.height / (2 * camera.fy))
        W, H = int(camera.width.item()), int(camera.height.item())
        self.W = W
        self.H = H
        self.last_size = (H, W)
        projmat = projection_matrix(0.01, 100, fovx, fovy, device=self.device)
        projmat = torch.matmul(viewmat,projmat)
        rotors1,rotors2=self.quats1,self.quats2
        N=self.xyzt.shape[0]
        time=camera.times[0][0]
        means4d = self.xyzt
        means3d,conv3, speed, w= self.slice_4d_to_3d(t_current=time, mask=None)
        opacity_aftersigmopid = self.get_opacities
        delta_t = means4d[:, 3] - time
        delta_t_2 = delta_t*delta_t
        tshift = 0.5 * w* delta_t_2
        #tshift =w* delta_t_2
        temporal_thres = 16
        temporal_mask = tshift < temporal_thres
        opacity_aftertime = torch.exp(-tshift)[:, None] * opacity_aftersigmopid
        shs_view = torch.cat((self.features_dc, self.features_rest), dim=1).transpose(1, 2).view(-1, 3, (self.config.sh_degree + 1) ** 2)
        #dir_pp = (means3d -  torch.linalg.inv(camera.camera_to_worlds[0])[3, :3].repeat(torch.cat((features_dc_crop, features_rest_crop), dim=1).shape[0],1))
        
        dir_pp = means3d - camera_center.repeat(N,1)
        dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
        sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
        colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        

        means4d_masked = means4d[temporal_mask]
        means3d_masked = means3d[temporal_mask]
        conv3_masked = conv3[temporal_mask]
        speed_masked = speed[temporal_mask]
        opacity_aftertime_masked = opacity_aftertime[temporal_mask]
        colors_precomp_masked = colors_precomp[temporal_mask]
        screenspace_points_full = torch.zeros_like(means4d[:, :3], dtype=means4d.dtype, requires_grad=True,
                                              device=self.xyzt.device) + 0
        try:
            screenspace_points_full.retain_grad()
        except:
            pass

        screenspace_points = screenspace_points_full[temporal_mask]
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)
        
        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=background,
            scale_modifier=1.0,
            viewmatrix=viewmat,
            projmatrix=projmat,
            sh_degree=self.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
            confidence=torch.ones_like(screenspace_points[:, 0:1])
        )
        
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        

        means3D = means3d_masked
        means2D = screenspace_points
        opacity = opacity_aftertime_masked

        scales = None
        rotations = None
        cov3D_precomp = conv3_masked
        
        shs = None
        colors_precomp = colors_precomp_masked
        rendered_image, radii, depth, alpha = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp,
        )
        
        rendered_image=torch.clamp_max(rendered_image,1.0)
        if self.step % 100 ==0:
            import torchvision.transforms.functional as TF
            from PIL import Image
            image_pil = TF.to_pil_image( rendered_image)
            image_pil.save("image.jpg")

        radii_full = torch.zeros_like(means4d[:, 0], dtype=radii.dtype, requires_grad=False,
                                                            device=self.xyzt.device) + 0                                  
        radii_full[temporal_mask] = radii
        spatial_mask = radii > 0

        totalmask = radii_full > 0

        render =  {
            "render": rendered_image,
            "depth": depth,
            "alpha": alpha,
            
            "mean4d_full": means4d,
            "mean3d_full": means3d,
            "speed_full": speed,

            "mask_t": totalmask,
            
            "mean4d_masked": means4d_masked[spatial_mask],
            "mean3d_masked": means3d_masked[spatial_mask],
            "speed_masked": speed_masked[spatial_mask],

            "viewspace_points": [screenspace_points_full, torch.Tensor([1, 1]).to(screenspace_points_full).reshape(-1, 2)],
            "visibility_filter": totalmask,
            "radii": radii_full,
        }
        output = {"rgb": rendered_image.permute(1,2,0), "depth": depth.permute(1,2,0), "accumulation": alpha.permute(1,2,0), "background": background}
        self.render = render
        torch.cuda.empty_cache()
        return output
  
        
    def forward_3d(self,
                   temporal_mask,
                   means3D,
                          opacity,
                          cov3D_precomp,
                          colors_precomp,
                        camera: Camera,
                        bg_color: torch.Tensor,
                        viewmat,
                        camera_center,
                        projmat,
                        tanfovx,tanfovy,
                        scaling_modifier=1.0,
                        override_color=None,
                        downsize = 1):
        
        
        ###############################################################
        # render 3d
        ###############################################################

        pnum = temporal_mask.shape[0]

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points_full = torch.zeros((pnum, 3), dtype=means3D.dtype, requires_grad=True,
                                              device=bg_color.device) + 0
        try:
            screenspace_points_full.retain_grad()
        except:
            pass

        screenspace_points = screenspace_points_full[temporal_mask]

        # Set up rasterization configuration
        self.scene_extent = float(camera.camera_extent)
        # calculate the FOV of the camera given fx and fy, width and height
        W, H = int(camera.width.item()), int(camera.height.item())
        self.W = W
        self.H = H
        self.last_size = (H, W)
        

        raster_settings = GaussianRasterizationSettings(
            image_height=int(H/downsize),
            image_width=int(W/downsize),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=1.0,
            viewmatrix=viewmat,
            projmatrix=projmat,
            sh_degree=self.active_sh_degree,
            campos=camera_center,
            prefiltered=False,
            debug=False,
            confidence=torch.ones_like(screenspace_points[:, 0:1])
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        means2D = screenspace_points

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None

        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image, radii, depth, alpha  = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        rendered_image=torch.clamp_max(rendered_image,1.0)

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        radii_full = (torch.zeros(pnum, dtype=radii.dtype, requires_grad=False,
                                              device=bg_color.device) + 0).contiguous()
        radii_full[temporal_mask] = radii.contiguous()

        totalmask = radii_full > 0
        if self.step % 100 ==0:
            import torchvision.transforms.functional as TF
            from PIL import Image
            image_pil = TF.to_pil_image( rendered_image)
            image_pil.save("image.jpg")
        
        torch.cuda.empty_cache()
        return {
            "render": rendered_image,
            "depth": depth,
            "alpha": alpha,
            "viewspace_points": [screenspace_points_full, torch.Tensor([1, 1]).to(screenspace_points_full).reshape(-1, 2)],
            "visibility_filter": totalmask,
            "radii": radii_full,
        }
    
    
    def forward_2(self,camera,background):
        means4d, conv3, speed, w = self.slice_4d_to_3d_without_T()
        opacity_aftersigmopid = self.get_opacities
        shs_view = torch.cat((self.features_dc, self.features_rest), dim=1).transpose(1, 2).view(-1, 3, (self.config.sh_degree + 1) ** 2)

        reg = self.config.batch_reg
        if reg:
            _, delta_t_center = self.slice_4d_to_3d_with_T(means4d, speed, camera[1].times[0][0])
            delta_t_center_2 = delta_t_center ** 2
            tshift_center = w * delta_t_center_2 
            opacity_aftertime_center = torch.exp(-tshift_center)[:, None] * opacity_aftersigmopid

        outputs = []

        for cam in camera:
            downsize = 1
            self.downsize = downsize
            camera_to_world=cam.camera_to_worlds[0]
            camera_to_world[:3, 1:3] *= -1
            world_to_camera = torch.linalg.inv(camera_to_world).to(torch.float)
            R = world_to_camera[:3, :3]
            T = world_to_camera[:3, 3]
            viewmat = torch.zeros((4, 4))
            viewmat[:3,:3] = R
            viewmat[:3,3] = T
            viewmat[3,3] = 1.
            viewmat = torch.transpose(viewmat, 0, 1).cuda()
            camera_center = torch.linalg.inv(viewmat)[3, :3]
            fovx = 2 * math.atan(cam.width / (2 * cam.fx))
            fovy = 2 * math.atan(cam.height / (2 * cam.fy))
            tanfovx = math.tan(fovx * 0.5)
            tanfovy = math.tan(fovy * 0.5)
            projmat = projection_matrix(0.01, 100, fovx, fovy, device=self.device)
            projmat = torch.matmul(viewmat,projmat)
            means3d, delta_t = self.slice_4d_to_3d_with_T(means4d, speed, cam.times[0][0])
            dir_pp = (means3d - camera_center.repeat(self.features_dc.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(self.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)

            if reg:
                tshift = tshift_center
                opacity_aftertime = opacity_aftertime_center
            else:
                delta_t_2 = delta_t ** 2
                tshift =  0.5*w * delta_t_2 
                opacity_aftertime = torch.exp(-tshift)[:, None] * opacity_aftersigmopid

            temporal_thres = 16
            temporal_mask = tshift < temporal_thres
            
            means3d_masked = means3d[temporal_mask]
            conv3_masked = conv3[temporal_mask]
            opacity_aftertime_masked = opacity_aftertime[temporal_mask]
            colors_precomp_masked = colors_precomp[temporal_mask]
            
            output = self.forward_3d(temporal_mask,
                                            means3d_masked,
                                            opacity_aftertime_masked,
                                            conv3_masked,
                                            colors_precomp_masked,
                                            cam,
                                            background,
                                            viewmat,
                                            camera_center,
                                            projmat,
                                            tanfovx,tanfovy,downsize = downsize)
           
            outputs.append(output)
           
        self.render = {}
        #self.render_low = {}
        self.render["mean4d_full"] = means4d
        self.render["speed_full"] = speed
        
        for k, _ in outputs[0].items():

            if k == "viewspace_points":
                self.render[k] = [[output[k][0] for output in outputs], output[k][1]]
            elif k in ["visibility_filter", "radii"]:
                self.render[k] = [output[k] for output in outputs]
            elif k in ['render', "depth", "alpha"]:
                self.render[k] = torch.stack([output[k] for output in outputs], dim=0)
        self.render['mask_t'] = torch.stack(self.render['visibility_filter'], dim=-1).any(dim=1)
        out = {"rgb": self.render['render'].permute(0,2,3,1), 
               'depth':self.render["depth"].permute(0,2,3,1), 
               "accumulation": self.render['alpha'].permute(0,2,3,1), 
               "background": background}
        
        #return outputs_dict
        return out
        
        
        


    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        

        
        #assert camera.shape[0] == 1, "Only one camera at a time"

        # get the background color
        if self.training:
            if self.config.background_color == "random":
                background = torch.rand(3, device=self.device)
            elif self.config.background_color == "white":
                background = torch.ones(3, device=self.device)
            elif self.config.background_color == "black":
                background = torch.zeros(3, device=self.device)
            else:
                background = self.background_color.to(self.device)
        else:
            if renderers.BACKGROUND_COLOR_OVERRIDE is not None:
                background = renderers.BACKGROUND_COLOR_OVERRIDE.to(self.device)
            else:
                background = self.background_color.to(self.device)

        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.xyz).squeeze()
            if crop_ids.sum() == 0:
                rgb = background.repeat(int(camera.height.item()), int(camera.width.item()), 1)
                depth = background.new_ones(*rgb.shape[:2], 1) * 10
                accumulation = background.new_zeros(*rgb.shape[:2], 1)
                return {"rgb": rgb, "depth": depth, "accumulation": accumulation, "background": background}
        else:
            crop_ids = None
        if self.step % 1000 ==0:
            if self.active_sh_degree <self.config.sh_degree:
                self.active_sh_degree += 1
        #if self.config.batch_reg != None:
        if isinstance(camera,list):
            output = self.forward_2(camera,background)
            self.batchflag = True
        else:
            output = self.forward_1(camera,background)
        if True:
            points4d, speed = self.render["mean4d_full"], self.render["speed_full"]
            mask_t = self.render['mask_t']
            points4d_msked, speed_masked = points4d[mask_t], speed[mask_t]
            
            K = 8

            # 4d knn
            #scale_4d = torch.Tensor([1, 1, 1, 5]).to(points4d).reshape(1, -1)
            # knn_tscale = 5 #min(self.cameras_extent, 22)
            if isinstance(camera,list):
                knn_tscale = min(camera[0].camera_extent, 30)
            else:
                knn_tscale = min(camera.camera_extent, 30)
            scale_4d = torch.Tensor([[self.config.temporal_extent, self.config.temporal_extent, self.config.temporal_extent, knn_tscale]]).to(points4d_msked)
            scale_4d = torch.ones_like(scale_4d)
            
            K_knn = K
            points = (points4d_msked * scale_4d)
            pnum = points.shape[0]
            with torch.no_grad():
                # less than 60000, use knn, ~0.01sec
                #if pnum < 6e4: 
                if True:
                    re = knn_points(points.unsqueeze(0), points.unsqueeze(0), lengths1=None, lengths2=None, norm=2, K=K_knn, return_nn=False, return_sorted=False)  # 1 x k x 10
                    knnre4d = re.idx.reshape(-1,)
                    knnmask_pxk = (re.dists[0] > 1e-7).float() # if it is <1e-7, it is the same point
                    # dists = re.dists.max(dim=-1)[0]
            
                else:
                    # 4 means 256 bins, 5 is 625 bins
                    # assume each bin has 1e2 points
                    num_rough_bins = 4
                    bin_length = num_rough_bins + (self.step % 2)
                    re_dix_pxk, dists_pxk, knnmask_pxk = knn_fast(points, bin_length, K_knn)
                    knnre4d = re_dix_pxk.reshape(-1,)
            
            knn_speed4d_pxkx3 = speed_masked[knnre4d].reshape(-1, K_knn, 3)
            knn_mask_pxkx1 = knnmask_pxk.unsqueeze(-1)
            knn_validnum_px1 = knnmask_pxk.sum(dim=-1, keepdim=True)
            knn_mask_p  = (knn_validnum_px1[:, 0] > 0).float()

            knn_speed4d_average = (knn_speed4d_pxkx3 * knn_mask_pxkx1).sum(dim=1) / (knn_validnum_px1 + 1e-7)
            loss_speed_4d = (knn_speed4d_average - speed_masked).abs().sum(dim=-1) * (K_knn - 1) / K_knn
            loss_speed_4d = (loss_speed_4d * knn_mask_p).sum() / (knn_mask_p.sum() + 1e-7)

            self.knnloss=0.05 * loss_speed_4d
        if False:
            opacity = self.get_opacities
            opacity_filter = self.render['mask_t']
            # opacity_filter_2 = torch.stack(visibility_filter, dim=-1).any(dim=-1)
            # assert torch.all(opacity_filter == opacity_filter_2)

            visible_ocpaicty = opacity[opacity_filter]
            visible_ocpaicty_clip = torch.clamp(visible_ocpaicty, 1e-3, 1-1e-3)
            loss_opacity_entropy = -visible_ocpaicty_clip * torch.log(visible_ocpaicty_clip)
            loss_opacity_entropy = loss_opacity_entropy.mean()

            non_visib_opacity = opacity[~opacity_filter]
            non_visib_ocpaicty_clip = torch.clamp(non_visib_opacity, 1e-3, 1-1e-3)
            loss_opacity_entropy_nonvisib = -non_visib_ocpaicty_clip * torch.log(non_visib_ocpaicty_clip)
            loss_opacity_entropy_nonvisib = 0.01 * loss_opacity_entropy_nonvisib.mean()
            loss_opacity_entropy = loss_opacity_entropy + loss_opacity_entropy_nonvisib
    
            # self.is_opacity_reset = False
            if self.step < self.config.stop_split_at:
                # we have clip
                self.entropyloss=0.001 * loss_opacity_entropy 
            else:
                # let's also add it even we don't split
                self.entropyloss=0.001 * loss_opacity_entropy
        torch.cuda.empty_cache()
        return output  # type: ignore
        #return {"rgb": rendered_image.permute(2,1,0)}
        
        

    def slice_4d_to_3d(self, t_current=None, mask=None):
        xyzt = self.xyzt
        scale = self.scales
        r1, r2 = self.quats1,self.quats2
        if mask is not None:
            xyzt = xyzt[mask]
            scale = scale[mask]
            r1 = r1[mask]
            r2 = r2[mask]
        conv3, speed, w = slice_4d(scale, r1, r2)
        if t_current is not None:
            deltat = t_current - xyzt[:, 3:4]
            mean3d = xyzt[:, :3] + speed * deltat
        else:
            mean3d = None
        return mean3d, conv3, speed, w 
    def slice_4d_to_3d_without_T(self, mask=None):
        xyzt = self.xyzt
        scale = self.scales
        r1, r2 = self.quats1,self.quats2
        if mask is not None:
            xyzt = xyzt[mask]
            scale = scale[mask]
            r1 = r1[mask]
            r2 = r2[mask]
        
        conv3, speed, w = slice_4d(scale, r1, r2)
        return xyzt, conv3, speed, w
    
    def slice_4d_to_3d_with_T(self, xyzt, speed, t_current):
        deltat = t_current - xyzt[:, 3]
        mean3d = xyzt[:, :3] + speed * deltat[:, None]
        return mean3d, deltat

    def get_gt_img(self, image: torch.Tensor):
        """Compute groundtruth image with iteration dependent downscale factor for evaluation purpose

        Args:
            image: tensor.Tensor in type uint8 or float32
        """
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        gt_img = self._downscale_if_required(image)
        return gt_img.to(self.device)

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        if image.shape[2] == 4:
            alpha = image[..., -1].unsqueeze(-1).repeat((1, 1, 3))
            return alpha * image[..., :3] + (1 - alpha) * background
        else:
            return image

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        if gt_rgb.dim() == 3:
            metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb.permute(0,1,2))
        else:
            metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb.permute(0,1,2,3))
        metrics_dict["gaussian_count"] = self.num_points
        # print(metrics_dict)
        return metrics_dict

    def get_loss_dict(self, outputs, batch, metrics_dict=None) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        pred_img = outputs["rgb"]

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        
        if len(gt_img.shape)==3:
            simloss = self.ssim(gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...])
        else:
            simloss = self.ssim(pred_img.permute(0, 3, 1,2).contiguous(),gt_img.permute(0, 3, 1,2).contiguous())
        rgbloss = (1 - self.config.ssim_lambda) * Ll1 + self.config.ssim_lambda * (1-simloss) 
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = self.scales
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)
        
        
        return {
            "main_loss":rgbloss+self.entropyloss+self.knnloss,
            'scale_reg':scale_reg
        }
        
        

    @torch.no_grad()
    def get_outputs_for_camera(self, camera: Cameras, obb_box: Optional[OrientedBox] = None) -> Dict[str, torch.Tensor]:
        """Takes in a camera, generates the raybundle, and computes the output of the model.
        Overridden for a camera-based gaussian model.

        Args:
            camera: generates raybundle
        """
        assert camera is not None, "must provide camera to gaussian model"
        self.set_crop(obb_box)
        outs = self.get_outputs(camera.to(self.device))
        return outs  # type: ignore

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        """Writes the test image outputs.

        Args:
            image_idx: Index of the image.
            step: Current step.
            batch: Batch of data.
            outputs: Outputs of the model.

        Returns:
            A dictionary of metrics.
        """
        gt_rgb = self.composite_with_background(self.get_gt_img(batch["image"]), outputs["background"])
        d = self._get_downscale_factor()
        if d > 1:
            # torchvision can be slow to import, so we do it lazily.
            import torchvision.transforms.functional as TF

            newsize = [batch["image"].shape[0] // d, batch["image"].shape[1] // d]
            predicted_rgb = TF.resize(outputs["rgb"].permute(2, 0, 1), newsize, antialias=None).permute(1, 2, 0)
        else:
            predicted_rgb = outputs["rgb"]

        combined_rgb = torch.cat([gt_rgb, predicted_rgb], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        gt_rgb = torch.moveaxis(gt_rgb, -1, 0)[None, ...]
        predicted_rgb = torch.moveaxis(predicted_rgb, -1, 0)[None, ...]

        psnr = self.psnr(gt_rgb, predicted_rgb)
        ssim = self.ssim(gt_rgb, predicted_rgb)
        lpips = self.lpips(gt_rgb, predicted_rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim)}  # type: ignore
        metrics_dict["lpips"] = float(lpips)

        images_dict = {"img": combined_rgb}

        return metrics_dict, images_dict
