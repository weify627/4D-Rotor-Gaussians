// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"

template<typename scalar_t>
__global__ void reorder_data_bw_kernel(
    // Inputs
    const int32_t max_bin_num, // b
    const int32_t max_element_in_each_bin,  // e
    const int32_t num_knn,  // m
    const int32_t num_point,  // p
    const scalar_t* __restrict__ data_knndist_bxexm,  // we have k points, each point has d dimension feature
    const int32_t* __restrict__ data_bin_idx_bxexm, // we know the bin index of each point
    const int32_t* __restrict__ data_bin_length_b, // we know the bin index of each point
    const int32_t* __restrict__ data_ori_idx_bxe, // we know the bin index of each point
    // Outputs
    scalar_t* __restrict__ outdist_pxm, // we will put each point in the corresponding cell
    scalar_t* __restrict__ outmask_pxm, // we will put each point in the corresponding cell
    int32_t* __restrict__ outidx_pxm
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= max_bin_num * max_element_in_each_bin) {
        return;
    }

    // first, what's the index of this point
    const int32_t original_index_of_this_point = data_ori_idx_bxe[tidx];
    if (original_index_of_this_point < 0) {
        return;
    }
    int32_t original_point_shift = original_index_of_this_point * num_knn;

    // second, what's the knn point index of this point
    int32_t pointidx = tidx % max_element_in_each_bin;
    int32_t binidx = (tidx - pointidx) / max_element_in_each_bin;
    int32_t binshift = binidx * max_element_in_each_bin;

    int32_t valid_pnum = min(num_knn, data_bin_length_b[binidx]);

    int32_t shift = tidx * num_knn;
    for (int32_t j=0; j<valid_pnum; ++j){
        int32_t knnidx_in_this_bin = data_bin_idx_bxexm[shift + j];
        int32_t knnidx = binshift + knnidx_in_this_bin;
        int32_t knnidx_original = data_ori_idx_bxe[knnidx];
        outidx_pxm[original_point_shift + j] = knnidx_original;

        scalar_t knn_dist = data_knndist_bxexm[shift+j];
        outdist_pxm[original_point_shift + j] = knn_dist;

        scalar_t pvalid = (scalar_t)(knnidx_original!= original_index_of_this_point);
        outmask_pxm[original_point_shift + j] = pvalid;
    } 
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> reorder_data_bw_cu(
    torch::Tensor data_bin_idx_bxexm, 
    torch::Tensor data_binlen_b, // length of each bin
    torch::Tensor data_knndist_bxexm, 
    torch::Tensor data_ori_idx_bxe, 
    const int num_points

) {
    int32_t num_bin = data_bin_idx_bxexm.size(0);
    int32_t num_element_in_each_bin = data_bin_idx_bxexm.size(1);
    int32_t num_knn = data_bin_idx_bxexm.size(2);
    
    torch::Tensor data_dist_out = -torch::ones({num_points, num_knn}, data_knndist_bxexm.options());
    torch::Tensor data_mask_out = torch::zeros({num_points, num_knn}, data_knndist_bxexm.options());
    torch::Tensor data_index_out = torch::zeros({num_points, num_knn}, data_bin_idx_bxexm.options());

    const int total_thread = num_bin * num_element_in_each_bin;
    const int threads = 512, blocks = (total_thread+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data_knndist_bxexm.scalar_type(), "reorder_data_bw_cu", ([&] {
            reorder_data_bw_kernel<scalar_t><<<blocks, threads>>>(
                num_bin, 
                num_element_in_each_bin, 
                num_knn, 
                num_points,
                data_knndist_bxexm.data_ptr<scalar_t>(), 
                data_bin_idx_bxexm.data_ptr<int32_t>(), 
                data_binlen_b.data_ptr<int32_t>(), 
                data_ori_idx_bxe.data_ptr<int32_t>(),
                data_dist_out.data_ptr<scalar_t>(),
                data_mask_out.data_ptr<scalar_t>(),
                data_index_out.data_ptr<int32_t>()
            );
        }));

    return {data_dist_out, data_mask_out, data_index_out};
}