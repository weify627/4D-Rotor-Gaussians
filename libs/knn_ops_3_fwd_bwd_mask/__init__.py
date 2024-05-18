# Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

# Pre-load dynamic torch dependencies, otherwise runtime-lookup will fail for torch-specific .so's
import torch

from . import libknn_ops_cc as knn_ops  # type: ignore
