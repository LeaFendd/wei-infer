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

class CPUDeviceAllocator : public DeviceAllocator {
  public:
    explicit CPUDeviceAllocator();

    void *allocate(size_t byte_size) const override;

    void release(void *ptr) const override;
};

} // namespace base

#endif // WEI_INFER_BASE_ALLOC_H_