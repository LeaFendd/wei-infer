#include <cuda_runtime_api.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "base/buffer.h"

TEST(test_buffer, allocate) {
    using namespace base;
    auto allocator =
        base::DeviceAllocatorFactory::getInstance(DeviceType::kDeviceCPU);
    Buffer buffer(32, allocator);
    ASSERT_NE(buffer.ptr(), nullptr);
}

TEST(test_buffer, use_external) {
    using namespace base;
    auto allocator =
        base::DeviceAllocatorFactory::getInstance(DeviceType::kDeviceCPU);
    float *ptr = new float[32];
    Buffer buffer(32, allocator, ptr, true);
    ASSERT_EQ(buffer.ptr(), ptr);
    ASSERT_EQ(buffer.is_external(), true);
    delete[] ptr;
}

TEST(test_buffer, cuda_memcpy1) {
    using namespace base;
    auto alloc_cpu =
        base::DeviceAllocatorFactory::getInstance(DeviceType::kDeviceCPU);
    auto alloc_cuda =
        base::DeviceAllocatorFactory::getInstance(DeviceType::kDeviceCUDA);

    int32_t size = 32;
    float *ptr = new float[size];
    for (int i = 0; i < size; ++i) {
        ptr[i] = float(i);
    }
    Buffer buffer(size * sizeof(float), nullptr, ptr, true);
    buffer.set_device_type(DeviceType::kDeviceCPU);
    ASSERT_EQ(buffer.is_external(), true);

    Buffer cu_buffer(size * sizeof(float), alloc_cuda);
    cu_buffer.copy_from(buffer);

    float *ptr2 = new float[size];
    cudaMemcpy(
        ptr2, cu_buffer.ptr(), sizeof(float) * size, cudaMemcpyDeviceToHost
    );
    for (int i = 0; i < size; ++i) {
        // ptr[i] = float(i);
        ASSERT_EQ(ptr2[i], float(i));
    }

    delete[] ptr;
    delete[] ptr2;
}
