#ifndef WEI_INFER_BASE_ALLOC_H_
#define WEI_INFER_BASE_ALLOC_H_

#include "base.h"
#include <map>
#include <memory>

namespace base {
enum class MemcpyKind {
    kMemcpyCPU2CPU = 0,
    kMemcpyCPU2CUDA = 1,
    kMemcpyCUDA2CPU = 2,
    kMemcpyCUDA2CUDA = 3,
};

class DeviceAllocator {
  public:
    explicit DeviceAllocator(DeviceType device_type)
        : device_type_(device_type) {}

    virtual DeviceType device_type() const { return device_type_; }

    virtual void release(void *ptr) const = 0;

    virtual void *allocate(size_t byte_size) const = 0;

    virtual void memcpy(
        const void *src_ptr,
        void *dest_ptr,
        size_t byte_size,
        MemcpyKind memcpy_kind = MemcpyKind::kMemcpyCPU2CPU,
        void *stream = nullptr,
        bool need_sync = false
    ) const;

    virtual void memset_zero(
        void *ptr, size_t byte_size, void *stream, bool need_sync = false
    );

  private:
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
};

// CPU device allocator
class CPUDeviceAllocator : public DeviceAllocator {
  public:
    explicit CPUDeviceAllocator();

    void *allocate(size_t byte_size) const override;

    void release(void *ptr) const override;
};

class CPUDeviceAllocatorFactory {
  public:
    static std::shared_ptr<base::DeviceAllocator> getInstance() {
        return instance_;
    }

  private:
    static std::shared_ptr<base::CPUDeviceAllocator> instance_;
};

// CUDA device allocator
#define CUDA_LARGE_MEMORY_THRESHOLD (1 * MB)
#define CUDA_MAX_FREE_SMALL_BUFFER_SIZE (100 * MB)
#define CUDA_MAX_FREE_LARGE_BUFFER_SIZE (1 * GB)

struct CudaMemoryBuffer {
    void *data;
    size_t byte_size;
    bool busy;

    CudaMemoryBuffer() = default;

    CudaMemoryBuffer(void *data, size_t byte_size, bool busy)
        : data(data), byte_size(byte_size), busy(busy) {}
};

class CUDADeviceAllocator : public DeviceAllocator {
  public:
    explicit CUDADeviceAllocator();

    void *allocate(size_t byte_size) const override;

    void release(void *ptr) const override;

  private:
    // 每块GPU上设置两个buffer，一个用于存放大内存，一个用于存放小内存
    mutable std::map<int, size_t> free_size_cnt_; // 当前空闲内存大小
    mutable std::map<int, std::vector<CudaMemoryBuffer>> large_buffers_map_;
    mutable std::map<int, std::vector<CudaMemoryBuffer>> small_buffers_map_;

    void *cudaAllocateLargeMemory(size_t byte_size, int device_id) const;

    void *cudaAllocateSmallMemory(size_t byte_size, int device_id) const;
};

class CUDADeviceAllocatorFactory {
  public:
    static std::shared_ptr<base::DeviceAllocator> getInstance() {
        return instance_;
    }

  private:
    static std::shared_ptr<base::CUDADeviceAllocator> instance_;
};

// United device allocator factory
class DeviceAllocatorFactory {
  public:
    static std::shared_ptr<base::DeviceAllocator>
    getInstance(DeviceType device_type) {
        if (device_type == DeviceType::kDeviceCPU) {
            return CPUDeviceAllocatorFactory::getInstance();
        } else if (device_type == DeviceType::kDeviceCUDA) {
            return CUDADeviceAllocatorFactory::getInstance();
        } else {
            LOG(FATAL) << "Unsupported device type!";
            return nullptr;
        }
    }
};

} // namespace base

#endif // WEI_INFER_BASE_ALLOC_H_