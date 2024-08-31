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