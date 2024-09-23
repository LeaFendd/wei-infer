#include "base/base.h"
#include "base/cuda_config.h"
#include "tensor/tensor.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>

namespace kernel {
__global__ void flash_attention_kernel(
    const int32_t pos,
    const int32_t seq_len,
    const float *query,       // [1, D]
    const float *key_cache,   // [tile_size, D]
    const float *value_cache, // [tile_size, D]
    const int32_t dim,
    const int32_t group_size,
    const int32_t head_num,
    const int32_t head_size,
    const int32_t layer_offset,
    const int num_tiles,
    const int tile_size,
    float *l,     // [num_tiles]
    float *m,     // [num_tiles]
    float *output //[1, D]
) {
    // grid[H], block(128)
    // smem: [kv_dim * 3]
    int tid = threadIdx.x;
    int head = blockIdx.x;

    if (head >= head_num) {
        return;
    }

    // 索引到当前head的q k v s l m o
    const float *q_head = query + head * head_size;
    const float *k_head = key_cache + layer_offset + head * head_size;
    const float *v_head = value_cache + layer_offset + head * head_size;
    float *l_head = l + head * tile_size;
    float *m_head = m + head * tile_size;
    float *out_head = output + head * head_size;

    // Smem layout: q_smem, k_smem, v_smem, s_smem, o_smem
    extern __shared__ float smem[];
    float *q_smem = smem;                     // [dim]
    float *k_smem = q_smem + dim;             // [tile_size * dim]
    float *v_smem = k_smem + tile_size * dim; // [tile_size * dim]
    float *s_smem = v_smem + tile_size * dim; // [tile_size]
    float *o_smem = s_smem + tile_size;       // [dim]

#pragma unroll
    // 加载q到smem，所有线程同时加载
    for (int i = tid; i < dim; i += blockDim.x) {
        q_smem[i] = q_head[i];
    }
    __syncthreads();

    float tile_m = -INFINITY;
    const float scale = 1.0f / sqrtf(dim);
#pragma unroll
    for (size_t tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        // 索引到当前tile的k v
        const float *k_tile = k_head + tile_idx * dim;
        const float *v_tile = v_head + tile_idx * dim;
#pragma unroll
        // 加载 k_tile 和 v_tile 到共享内存，每个线程加载[dim]
        for (int idx = tid; idx < tile_size * dim; idx += blockDim.x) {
            k_smem[idx] = k_tile[idx];
            v_smem[idx] = v_tile[idx];
        }
        __syncthreads();

        // 计算S_ij
        float s = 0.0f;
        if (tid < tile_size) {
            for (int i = 0; i < dim; i++) {
                s += q_smem[i] * k_smem[tid * dim + i];
            }
            s *= scale;
            s_smem[tid] = s;
        }
        __syncthreads();

        // 计算 tile_m = max(s_smem[tid])
        float val = (tid < tile_size) ? s_smem[tid] : -INFINITY;

        // 并行归约求最大值
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
        }
        if ((tid % warpSize) == 0) {
            smem[tid / warpSize] = val; // 复用 smem 的前部分空间
        }
        __syncthreads();

        if (tid < warpSize) {
            val = (tid < (blockDim.x / warpSize)) ? smem[tid] : -INFINITY;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
            }
            if (tid == 0) {
                tile_m = val;
            }
        }
        __syncthreads();

        // 计算 exp(s - tile_m)
        float exp_s = 0.0f;
        if (tid < tile_size) {
            s_smem[tid] = expf(s_smem[tid] - tile_m);
            exp_s = s_smem[tid];
        }
        __syncthreads();

        // 规约求和得到l_i
        val = (tid < tile_size) ? exp_s : 0.0f;
        for (int offset = warpSize / 2; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if ((tid % warpSize) == 0) {
            smem[tid / warpSize] = val; // 复用 smem 的前部分空间
        }
        __syncthreads();

        float l_tile = 0.0f;
        if (tid < warpSize) {
            val = (tid < (blockDim.x / warpSize)) ? smem[tid] : 0.0f;
            for (int offset = warpSize / 2; offset > 0; offset /= 2) {
                val += __shfl_down_sync(0xffffffff, val, offset);
            }
            if (tid == 0) {
                l_tile = val;
            }
        }
        __syncthreads();

        // 更新 l 和 m
        float row_m_prev = (tile_idx > 0) ? m_head[tile_idx - 1] : -INFINITY;
        float row_l_prev = (tile_idx > 0) ? l_head[tile_idx - 1] : 0.0f;

        float new_m = fmaxf(row_m_prev, tile_m);
        float exp_m_diff_prev = expf(row_m_prev - new_m);
        float exp_m_diff_tile = expf(tile_m - new_m);
        float new_l = exp_m_diff_prev * row_l_prev + exp_m_diff_tile * l_tile;

        if (tid == 0) {
            m_head[tile_idx] = new_m;
            l_head[tile_idx] = new_l;
        }
        __syncthreads();

        if (tile_idx == 0) {
            for (int i = tid; i < dim; i += blockDim.x) {
                o_smem[i] = 0.0f;
            }
        }
        __syncthreads();

        // 每个线程计算部分和
        for (int i = tid; i < dim; i += blockDim.x) {
            float v_sum = 0.0f;
            for (int t = 0; t < tile_size; t++) {
                v_sum += s_smem[t] * v_smem[t * dim + i];
            }

            // 更新 o_smem
            float o_prev = o_smem[i];
            o_smem[i] = ((exp_m_diff_prev * row_l_prev * o_prev) +
                         (exp_m_diff_tile * v_sum)) /
                        new_l;
        }
        __syncthreads();
    } // 遍历tiles

    // 写回输出到全局内存
    for (int i = tid; i < dim; i += blockDim.x) {
        out_head[i] = o_smem[i];
    }
}

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
) {
    int32_t tile_size = 128; // 每个分块的大小
    int32_t num_tiles = (seq_len + tile_size - 1) / tile_size;
    dim3 block(tile_size);
    dim3 grid(head_num);
    int32_t layer_offset = layer_index * seq_len * dim;

    // Q K V S O
    const int32_t smem_size =
        ((2 + 2 * tile_size) * dim + tile_size) * sizeof(float);

    // l, m
    auto dtype = base::DataType::kDataTypeFp32;
    auto alloc_cu = base::CUDADeviceAllocatorFactory::getInstance();
    tensor::Tensor l_tensor(dtype, head_num, num_tiles, true, alloc_cu);
    tensor::Tensor m_tensor(dtype, head_num, num_tiles, true, alloc_cu);
    m_tensor.set_value(-INFINITY);

    // call kernel
    flash_attention_kernel<<<grid, block, smem_size, config->stream>>>(
        pos,
        seq_len,
        query_tensor.ptr<float>(),
        key_cache_tensor.ptr<float>(),
        value_cache_tensor.ptr<float>(),
        dim,
        group_size,
        head_num,
        head_size,
        layer_offset,
        num_tiles,
        tile_size,
        l_tensor.ptr<float>(),
        m_tensor.ptr<float>(),
        mha_out.ptr<float>()
    );
}
} // namespace kernel
