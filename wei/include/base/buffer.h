#ifndef WEI_INFER_BASE_BUFFER_H_
#define WEI_INFER_BASE_BUFFER_H_

#include "base/alloc.h"
#include <memory>

namespace base {
class Buffer : std::enable_shared_from_this<Buffer> {
  private:
    size_t byte_size_ = 0;
    std::shared_ptr<DeviceAllocator> allocator_;
    void *ptr_ = nullptr;
    DeviceType device_type_ = DeviceType::kDeviceUnknown;
    bool use_external_ = false;

  protected:
    Buffer(const Buffer &) = delete; // 禁止复制构造函数

    Buffer &operator=(const Buffer &) = delete; // 禁止赋值构造函数

  public:
    explicit Buffer() = default;

    explicit Buffer(
        size_t byte_size,
        std::shared_ptr<DeviceAllocator> allocator = nullptr,
        void *ptr = nullptr,
        bool use_external = false
    );

    virtual ~Buffer();

    bool allocate();

    void copy_from(const Buffer &buffer) const;

    void copy_from(const Buffer *buffer) const;

    void *ptr();

    const void *ptr() const;

    size_t byte_size() const;

    std::shared_ptr<DeviceAllocator> allocator() const;

    DeviceType device_type() const;

    void set_device_type(DeviceType device_type);

    std::shared_ptr<Buffer> get_shared_from_this();

    bool is_external() const;
};
} // namespace base

#endif // WEI_INFER_BASE_BUFFER_H_