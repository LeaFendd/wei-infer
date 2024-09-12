#ifndef WEI_INFER_TENSOR_TENSOR_H_
#define WEI_INFER_TENSOR_TENSOR_H_

#include "base/base.h"
#include "base/buffer.h"
#include <armadillo>
#include <driver_types.h>
#include <glog/logging.h>
#include <memory>
#include <vector>

namespace tensor {

class Tensor {
  public:
    explicit Tensor() = default;

    explicit Tensor(
        base::DataType data_type,
        std::vector<int32_t> dims,
        bool need_alloc = false,
        std::shared_ptr<base::DeviceAllocator> alloc = nullptr,
        void *ptr = nullptr
    );

    void init_buffer(
        std::shared_ptr<base::DeviceAllocator> alloc,
        base::DataType data_type,
        bool need_alloc,
        void *ptr
    );

    // index
    template <typename T> T &index(int64_t offset);

    template <typename T> const T &index(int64_t offset) const;

    // data ptr
    template <typename T> T *ptr();

    template <typename T> const T *ptr() const;

    std::shared_ptr<base::Buffer> get_buffer() const;

    bool assign(std::shared_ptr<base::Buffer> buffer);

    void reset(base::DataType data_type, const std::vector<int32_t> &dims);

    bool allocate(
        std::shared_ptr<base::DeviceAllocator> allocator,
        bool need_realloc = false
    );

    //
    size_t size() const;

    size_t byte_size() const;

    int32_t dims_size() const;

    int32_t get_dim(int32_t idx) const;

    const std::vector<int32_t> &dims() const;

    std::vector<size_t> strides() const;

    base::DataType data_type() const;

    bool is_empty() const;

    void reshape(const std::vector<int32_t> &dims);

    tensor::Tensor clone() const;

    //
    void to_cpu();

    void to_cuda(cudaStream_t stream = nullptr);

    void set_device_type(base::DeviceType device_type) const;

    base::DeviceType device_type() const;

  private:
    size_t size_ = 0;
    std::vector<int32_t> dims_;
    std::shared_ptr<base::Buffer> buffer_;
    base::DataType data_type_ = base::DataType::kDataTypeUnknown;
};
} // namespace tensor

#endif // WEI_INFER_TENSOR_TENSOR_H_