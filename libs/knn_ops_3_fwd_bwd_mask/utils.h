// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#pragma once

#include <torch/extension.h>

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<torch::Tensor, torch::Tensor> reorder_data_fw_cu(
    const torch::Tensor data_kxd,
    const torch::Tensor bin_k,
    const int num_bin,
    const int max_bin_element
);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reorder_data_bw_cu(
    const torch::Tensor data_bin_idx_bxexm,
    const torch::Tensor data_binlen_b, // length of each bin
    const torch::Tensor data_knndist_bxexm,
    const torch::Tensor data_ori_idx_bxe,
    const int num_points
);
