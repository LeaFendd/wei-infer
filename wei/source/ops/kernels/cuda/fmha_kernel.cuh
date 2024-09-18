#ifndef FMHA_KERNEL_CU_CUH
#define FMHA_KERNEL_CU_CUH
#include "../kernels_interface.h"
#include "tensor/tensor.h"

namespace kernel {
void fmha_kernel_cu(
    const int32_t pos,
    const int32_t head_num,
    const int32_t layer_index,
    const int32_t seq_len,
    const int32_t dim,
    const int32_t group_size,
    const int32_t head_size,
    const tensor::Tensor &query_tensor,
    const tensor::Tensor &key_cache_tensor,
    const tensor::Tensor &value_cache_tensor,
    const CudaConfig *config,
    tensor::Tensor &mha_out
);
} // namespace kernel

#endif // MATMUL_KERNEL_CU_CUH
