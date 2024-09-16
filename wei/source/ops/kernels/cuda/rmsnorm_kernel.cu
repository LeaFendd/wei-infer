#include "rmsnorm_kernel.cuh"
#include <cub/cub.cuh>

namespace kernel {

template <int32_t BLOCK_DIM>
__global__ void row_rmsnorm_fp32_kernel(
    float *inputs, float *weights, const int size, const int eps, float *outputs
) {
    const int32_t tid = threadIdx.x;
    // vectorized load
    constexpr int pack_size = 4;
    const int num_packs = size / pack_size;
    const int pack_offset = num_packs * pack_size; // 最后不足4个元素的偏移量

    // 每个线程的本地和，每个线程负责处理一个float4
    float sum = 0.f;
    float4 *inputs_packed = reinterpret_cast<float4 *>(inputs);
    for (size_t i = tid; i < num_packs; i += blockDim.x) {
        float4 in = inputs_packed[i];
        sum += in.x * in.x + in.y * in.y + in.z * in.z + in.w * in.w;
    }
    // 处理剩余不足4个的元素
    for (size_t i = pack_offset + tid; i < size; i += blockDim.x) {
        sum += inputs[i] * inputs[i];
    }

    // 每个block做reduce
    using BlockReduceT = cub::BlockReduce<float, BLOCK_DIM>;
    __shared__ typename BlockReduceT::TempStorage temp_storage;
    __shared__ float shared_val;
    sum = BlockReduceT(temp_storage).Sum(sum);
    if (threadIdx.x == 0) {
        shared_val = sum;
    }
    __syncthreads();
    sum = shared_val;
    const float scale = rsqrtf(sum / static_cast<float>(size) + eps);

    // norm
    float4 *weights_packed = reinterpret_cast<float4 *>(weights);
    float4 *outputs_packed = reinterpret_cast<float4 *>(outputs);
    for (size_t i = tid; i < num_packs; i += blockDim.x) {
        float4 in = inputs_packed[i];
        float4 weight = weights_packed[i];
        outputs_packed[i] = make_float4(
            scale * in.x * weight.x,
            scale * in.y * weight.y,
            scale * in.z * weight.z,
            scale * in.w * weight.w
        );
    }
    for (size_t i = pack_offset + tid; i < size; i += blockDim.x) {
        outputs[i] = scale * inputs[i] * weights[i];
    }
    __syncthreads();
}

} // namespace kernel
