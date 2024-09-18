#include "../source/ops/kernels/kernels_interface.h"
#include "../utils.cuh"
#include "base/buffer.h"
#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

TEST(test_fmha, fmha1) {
    using namespace base;
    auto alloc = CUDADeviceAllocatorFactory::getInstance();

    const int L = 512;
    const int D = 128;
    const int H = 8;

    tensor::Tensor q(DataType::kDataTypeFp32, 1, H, D, true, alloc);
    tensor::Tensor k(DataType::kDataTypeFp32, L, H, D, true, alloc);
    tensor::Tensor v(DataType::kDataTypeFp32, L, H, D, true, alloc);
    tensor::Tensor o(DataType::kDataTypeFp32, 1, H, D, true, alloc);

    q.set_value(0.01);
    k.set_value(0.01);
    v.set_value(0.01);

    kernel::get_fmha_kernel(DeviceType::kDeviceCUDA)(
        0, H, 0, L, D * H, 1, D, q, k, v, nullptr, o
    );
    cudaDeviceSynchronize();
}