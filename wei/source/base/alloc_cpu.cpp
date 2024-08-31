#include "base/alloc.h"
#include <cstdlib>
#include <glog/logging.h>

namespace base {
CPUDeviceAllocator::CPUDeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCPU) {}

void *CPUDeviceAllocator::allocate(size_t byte_size) const {
    if (byte_size == 0) {
        return nullptr;
    } else {
        return malloc(byte_size);
    }
}

void CPUDeviceAllocator::release(void *ptr) const {
    if (ptr != nullptr) {
        free(ptr);
    }
}

} // namespace base