#include "tensor/tensor.h"
#include <cuda_device_runtime_api.h>
#include <cuda_runtime.h>
#include <glog/logging.h>
#include <numeric>

namespace tensor {
template <typename T, typename Tp>
static size_t reduce_dimension(T begin, T end, Tp init) {
    if (begin >= end) {
        return 0;
    }
    size_t size = std::accumulate(begin, end, init, std::multiplies<>());
    return size;
}

Tensor::Tensor(
    base::DataType data_type,
    std::vector<int32_t> dims,
    bool need_alloc,
    std::shared_ptr<base::DeviceAllocator> alloc,
    void *ptr
)
    : dims_(std::move(dims)), data_type_(data_type) {
    size_ = reduce_dimension(dims_.begin(), dims_.end(), 1);
    if (need_alloc && alloc) {
        allocate(alloc);
    } else {
        init_buffer(alloc, data_type, need_alloc, ptr);
    }
}

size_t Tensor::byte_size() const {
    return this->size() * DataTypeSize(data_type_);
}

/**
 * @brief allocate逻辑
 * @param allocator
 * @param need_realloc
 * @return true
 * @return false
 */
bool Tensor::allocate(
    std::shared_ptr<base::DeviceAllocator> allocator, bool need_realloc = false
) {
    if (!allocator) {
        LOG(ERROR
        ) << "The allocator parameter in the allocate function is null "
             "pointer!";
        return false;
    }
    const size_t byte_size = this->byte_size();
    if (byte_size == 0) {
        LOG(ERROR) << "The byte_size parameter in the allocate function is "
                      "equal to zero!";
        return false;
    }
    if (buffer_ && byte_size <= buffer_->byte_size()) {
        if (!need_realloc) {
            return true;
        }
    }
    buffer_ = std::make_shared<base::Buffer>(byte_size, allocator, nullptr);
    if (!buffer_->ptr()) {
        LOG(ERROR) << "The memory allocated is a null pointer!";
        return false;
    }
    return true;
}

const std::vector<int32_t> &Tensor::dims() const { return this->dims_; }

void Tensor::set_device_type(base::DeviceType device_type) const {
    if (buffer_) {
        buffer_->set_device_type(device_type);
    }
}

void Tensor::reshape(const std::vector<int32_t> &dims) {
    size_t size = reduce_dimension(dims.begin(), dims.end(), 1);
    if (!buffer_) {
        this->dims_ = dims;
        this->size_ = size;
        return;
    }

    if (size > size_) {
        auto new_buffer = std::make_shared<base::Buffer>(
            size * base::DataTypeSize(this->data_type_), buffer_->allocator()
        );
        CHECK(new_buffer->allocate());
        new_buffer->copy_from(buffer_.get());
        this->buffer_ = new_buffer;
    }
    this->dims_ = dims;
    this->size_ = size;
}

std::shared_ptr<base::Buffer> Tensor::get_buffer() const { return buffer_; }

Tensor Tensor::clone() const {
    Tensor new_tensor = *this;
    size_t byte_size = this->byte_size();

    auto allocator = buffer_->allocator();
    new_tensor.buffer_ = std::make_shared<base::Buffer>(byte_size, allocator);
    new_tensor.buffer_->copy_from(buffer_.get());
    return new_tensor;
}

size_t Tensor::byte_size() const {
    return this->size() * DataTypeSize(data_type_);
}

std::vector<size_t> Tensor::strides() const {
    std::vector<size_t> strides;
    if (!dims_.empty()) {
        for (int32_t i = 0; i < dims_.size() - 1; ++i) {
            size_t stride =
                reduce_dimension(dims_.begin() + i + 1, dims_.end(), 1);
            strides.push_back(stride);
        }
        strides.push_back(1);
    }
    return strides;
}

bool Tensor::is_empty() const {
    return size_ == 0 || buffer_ == nullptr || buffer_->ptr() == nullptr;
}

void Tensor::init_buffer(
    std::shared_ptr<base::DeviceAllocator> alloc,
    base::DataType data_type,
    bool need_alloc,
    void *ptr
) {
    if (!alloc && !need_alloc) {
        std::shared_ptr<base::Buffer> buffer = std::make_shared<base::Buffer>(
            DataTypeSize(data_type) * size_, nullptr, ptr, true
        );
        this->buffer_ = buffer;
    } else {
        allocate(alloc, true);
    }
}
} // namespace tensor