// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"
#include <vector>

std::tuple<torch::Tensor, torch::Tensor> reorder_data_fw(
    const torch::Tensor data_kxd,
    const torch::Tensor bin_k,
    const int num_bin,
    const int max_bin_element
){
    CHECK_INPUT(data_kxd);
    CHECK_INPUT(bin_k);

    return reorder_data_fw_cu(data_kxd, bin_k, num_bin, max_bin_element);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reorder_data_bw(
    const torch::Tensor data_bin_idx_bxexm,
    const torch::Tensor data_binlen_b, // length of each bin
    const torch::Tensor data_knndist_bxexm,
    const torch::Tensor data_ori_idx_bxe,
    const int num_points
){
    CHECK_INPUT(data_bin_idx_bxexm);
    CHECK_INPUT(data_binlen_b);
    CHECK_INPUT(data_knndist_bxexm);
    CHECK_INPUT(data_ori_idx_bxe);

    return reorder_data_bw_cu(data_bin_idx_bxexm, data_binlen_b, data_knndist_bxexm, data_ori_idx_bxe, num_points);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("reorder_data_fw", &reorder_data_fw);
    m.def("reorder_data_bw", &reorder_data_bw);   
}
