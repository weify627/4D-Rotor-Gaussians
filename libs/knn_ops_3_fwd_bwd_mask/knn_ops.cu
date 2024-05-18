// Copyright (c) 2023 NVIDIA CORPORATION.  All rights reserved.

#include "utils.h"

template<typename scalar_t>
__global__ void reorder_data_fw_kernel(
    // Inputs
    const int32_t max_bin_num,
    const int32_t max_element_in_each_bin,
    const int32_t num_point, 
    const int32_t num_dim, 
    const scalar_t* __restrict__ data_kxd,  // we have k points, each point has d dimension feature
    const int32_t* __restrict__ binidx_k, // we know the bin index of each point
    // Outputs
    scalar_t* __restrict__ out_bxexd, // we will put each point in the corresponding cell
    int32_t* __restrict__ outidx_bxe
) {
    int32_t tidx = blockDim.x * blockIdx.x + threadIdx.x;
    if (tidx >= max_bin_num) {
        return;
    }

    int32_t shift = tidx * max_element_in_each_bin;
    int32_t num_sample = 0;
    for (int32_t j=0; j<num_point; ++j){
        if(binidx_k[j] == tidx)
        {
            for (int32_t k=0; k<num_dim; ++k){
                out_bxexd[(shift + num_sample) * num_dim + k] = data_kxd[j * num_dim + k];
            }
            outidx_bxe[shift + num_sample] = j;
            num_sample ++;
        }
    } 
}

std::tuple<torch::Tensor, torch::Tensor> reorder_data_fw_cu(
    torch::Tensor data_kxd, 
    torch::Tensor binidx_k, 
    const int max_bin_num,
    const int max_element_in_each_bin
) {
    int32_t num_points = data_kxd.size(0);
    int32_t num_dim = data_kxd.size(1);
    
    torch::Tensor data_out = -torch::ones({max_bin_num, max_element_in_each_bin, num_dim}, data_kxd.options());
    torch::Tensor data_index_out = -torch::ones({max_bin_num, max_element_in_each_bin}, binidx_k.options());

    const int threads = 512, blocks = (max_bin_num+threads-1)/threads;

    AT_DISPATCH_ALL_TYPES_AND_HALF(data_kxd.scalar_type(), "reorder_data_fw_cu", ([&] {
            reorder_data_fw_kernel<scalar_t><<<blocks, threads>>>(
                max_bin_num, 
                max_element_in_each_bin, 
                num_points, 
                num_dim,
                data_kxd.data_ptr<scalar_t>(), 
                binidx_k.data_ptr<int32_t>(), 
                data_out.data_ptr<scalar_t>(),
                data_index_out.data_ptr<int32_t>()
            );
        }));

    return {data_out, data_index_out};
}