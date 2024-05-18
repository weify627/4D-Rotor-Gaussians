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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

from glob import glob
from PIL import Image
import os.path

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import DataParser, DataParserConfig, DataparserOutputs
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json



@dataclass
class PlenDataParserConfig(DataParserConfig):

    _target: Type = field(default_factory=lambda:Plenoptic )
    """target class to instantiate"""
    data: Path = Path("data/dnerf/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    alpha_color: str = "black"
    """alpha color of background"""
    downsample: int = 1
    batch_size: int = 2
    

@dataclass
class Plenoptic(DataParser):

    config: PlenDataParserConfig
    includes_time: bool = True
    
    def __init__(self, config: PlenDataParserConfi):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color
        self.downsample = config.downsample
        self.batch_size = config.batch_size



    def _generate_dataparser_outputs(self, split="train"):

        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None
        if split == 'test':
            self.batch_size = 1
        meta = load_from_json(self.data / f"transforms_{split}.json")
        image_filenames = []
        poses = []
        times = []
        for frame in meta["frames"]:  
            fname = self.data / Path(frame["file_path"].replace("./", "") + ".png")
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
            times.append(frame["time"])
        poses = np.array(poses).astype(np.float32)
        times = torch.tensor(times, dtype=torch.float32)
        
        image_height, image_width = 1014,1352
        
        focal_length_x = float(meta["fl_x"])
        focal_length_y = float(meta["fl_y"])
        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses) 
        camera_to_world_=camera_to_world.clone()
        camera_to_world_[:,:3, 1:3] *= -1
        world_to_camera_ = torch.linalg.inv(camera_to_world_).to(torch.float)
        R_ = world_to_camera_[:,:3, :3]
        R_[0],R_[1] = R_[0].clone() , R_[1].clone()
        T_ = world_to_camera_[:, :3, 3]
        viewmat_ = torch.zeros((T_.shape[0],4, 4))
        viewmat_[:,:3,:3] = R_
        viewmat_[:,:3,3] = T_
        viewmat_[:,3,3] = 1.
        viewmat_ = torch.transpose(viewmat_, 1, 2).cuda()
        
        camera_centers = torch.linalg.inv(viewmat_)[:,3, :3]
        
        
        # camera to world transform
        average_camera_center = torch.mean(camera_centers, dim=0)
        camera_distance = torch.linalg.norm(camera_centers - average_camera_center, dim=-1)
        max_distance = torch.max(camera_distance)
        camera_extent = float(max_distance) 
        camera_extent = torch.ones_like(times) * camera_extent
        batch_size = torch.ones_like(times) * self.batch_size
        
        scene_box = SceneBox(aabb=torch.tensor([[-16, -20, 5], [16, 20, 24]], dtype=torch.float32))
        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length_x/self.downsample,
            fy=focal_length_y/self.downsample,
            cx=cx/self.downsample,
            cy=cy/self.downsample,
            camera_type=CameraType.PERSPECTIVE,
            times=times,
            camera_extent = camera_extent,
            batch_size = batch_size,
        )

        if self.batch_size == 1:
            dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            )
            return dataparser_outputs
        
        
        indices_all = [torch.randperm(times.shape[0]).tolist() for _ in range(self.batch_size)]
        indices = []
        for i in list(range(times.shape[0])):
            random_indices = [d[i] for d in indices_all]
            indices.append(random_indices)
        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
            dataparser_scale=self.scale_factor,
            indices = indices,
        )

        return dataparser_outputs


