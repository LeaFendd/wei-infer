#include <cuda_runtime_api.h>

#include "base/alloc.h"

namespace base {

CUDADeviceAllocator::CUDADeviceAllocator()
    : DeviceAllocator(DeviceType::kDeviceCUDA) {}

void *CUDADeviceAllocator::allocate(size_t byte_size) const {
    int device_id = 0;
    cudaError_t cuda_status = cudaGetDevice(&device_id);
    CHECK(cuda_status == cudaSuccess);
    if (byte_size >= CUDA_LARGE_MEMORY_THRESHOLD) {
        return cudaAllocateLargeMemory(byte_size, device_id);
    } else {
        return cudaAllocateSmallMemory(byte_size, device_id);
    }
}

void *CUDADeviceAllocator::cudaAllocateLargeMemory(
    size_t byte_size, int device_id
) const {
    cudaError_t cuda_status = cudaGetDevice(&device_id);
    CHECK(cuda_status == cudaSuccess);
    // 获取当前device上的表
    auto &buffers_table = large_buffers_map_[device_id];
    // 遍历表，找到合适的内存
    int sel_id = -1;
    for (size_t i = 0; i < buffers_table.size(); i++) {
        if (buffers_table[i].byte_size >= byte_size && !buffers_table[i].busy &&
            buffers_table[i].byte_size - byte_size <
                CUDA_LARGE_MEMORY_THRESHOLD) {
            // 选择空间最小的buffer
            if (sel_id = -1 || (buffers_table[i].byte_size <
                                buffers_table[sel_id].byte_size)) {
                sel_id = i;
            }
        }
    }
    if (sel_id != -1) {
        buffers_table[sel_id].busy = true;
        return buffers_table[sel_id].data;
    }

    // 若没找到空闲且合适的内存，则分配新的内存
    void *data = nullptr;
    cuda_status = cudaMalloc(&data, byte_size);
    if (cudaSuccess != cuda_status) {
        char buf[256];
        snprintf(
            buf,
            256,
            "Error: CUDA error when allocating %lu MB memory! maybe there's no "
            "enough memory left on  device.",
            byte_size >> 20
        );
        LOG(ERROR) << buf;
        return nullptr;
    }
    buffers_table.emplace_back(data, byte_size, true);
    return data;
}

void *CUDADeviceAllocator::cudaAllocateSmallMemory(
    size_t byte_size, int device_id
) const {
    cudaError_t cuda_status = cudaGetDevice(&device_id);
    CHECK(cuda_status == cudaSuccess);
    // 获取当前device上的表
    auto &buffers_table = small_buffers_map_[device_id];
    // 遍历表，找到合适的内存
    for (auto &buffer : buffers_table) {
        if (buffer.byte_size >= byte_size && !buffer.busy) {
            buffer.busy = true;
            return buffer.data;
        }
    }
    // 若没找到空闲且合适的内存，则分配新的内存
    void *data = nullptr;
    cuda_status = cudaMalloc(&data, byte_size);
    if (cudaSuccess != cuda_status) {
        char buf[256];
        snprintf(
            buf,
            256,
            "Error: CUDA error when allocating %lu MB memory! maybe there's no "
            "enough memory left on  device.",
            byte_size >> 20
        );
        LOG(ERROR) << buf;
        return nullptr;
    }
    buffers_table.emplace_back(data, byte_size, true);
    return data;
}

void CUDADeviceAllocator::release(void *ptr) const {
    if (ptr == nullptr) {
        return;
    }
    // 空闲内存过多时统一释放内存
    cudaError_t cuda_status = cudaSuccess;
    for (auto &it : small_buffers_map_) {
        if (free_size_cnt_[it.first] > CUDA_MAX_FREE_SMALL_BUFFER_SIZE) {
            std::vector<CudaMemoryBuffer> temp;
            auto &buffers = it.second;
            for (size_t i = 0; i < buffers.size(); i++) {
                if (!buffers[i].busy) {
                    cuda_status = cudaSetDevice(it.first);
                    cuda_status = cudaFree(buffers[i].data);
                    CHECK(cuda_status == cudaSuccess)
                        << "Error: CUDA error when release memory on device!";
                } else {
                    temp.emplace_back(buffers[i]);
                }
            }
            buffers.clear();
            it.second = temp;
            free_size_cnt_[it.first] = 0;
        }
    }

    // 对ptr取消busy状态，使其可以用于其他内存分配
    for (auto &it : small_buffers_map_) {
        for (auto &buffer : it.second) {
            if (buffer.data == ptr) {
                buffer.busy = false;
                free_size_cnt_[it.first] += buffer.byte_size;
                return;
            }
        }
    }
    for (auto &it : large_buffers_map_) {
        for (auto &buffer : it.second) {
            if (buffer.data == ptr) {
                buffer.busy = false;
                return;
            }
        }
    }
    // default处理：ptr不在两个内存池中，直接释放内存
    cuda_status = cudaFree(ptr);
    CHECK(cuda_status == cudaSuccess)
        << "Error: CUDA error when release memory on device";
}

std::shared_ptr<CUDADeviceAllocator> CUDADeviceAllocatorFactory::instance_ =
    std::make_shared<CUDADeviceAllocator>();
} // namespace base
