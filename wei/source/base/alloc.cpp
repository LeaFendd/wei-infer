#include "base/alloc.h"
#include <cuda_runtime_api.h>

namespace base {

cudaMemcpyKind _getCudaMemcpyKind(MemcpyKind memcpy_kind) {
    switch (memcpy_kind) {
    case MemcpyKind::kMemcpyCPU2CUDA:
        return cudaMemcpyHostToDevice;
    case MemcpyKind::kMemcpyCUDA2CPU:
        return cudaMemcpyDeviceToHost;
    case MemcpyKind::kMemcpyCUDA2CUDA:
        return cudaMemcpyDeviceToDevice;
    default:
        LOG(FATAL) << "Unknown memcpy kind: " << static_cast<int>(memcpy_kind);
        return cudaMemcpyDefault;
    }
}

cudaError_t _cudaMemcpyWraper(
    const void *src,
    void *dst,
    size_t n,
    MemcpyKind memcpy_kind,
    void *cuda_stream
) {
    cudaMemcpyKind kind = _getCudaMemcpyKind(memcpy_kind);
    if (cuda_stream) {
        CUstream_st *_stream = static_cast<CUstream_st *>(cuda_stream);
        return cudaMemcpyAsync(dst, src, n, kind, _stream);
    } else {
        return cudaMemcpy(dst, src, n, kind);
    }
}

// DeviceAllocator
void DeviceAllocator::memcpy(
    const void *src_ptr,
    void *dest_ptr,
    size_t byte_size,
    MemcpyKind memcpy_kind,
    void *stream,
    bool need_sync
) const {
    // input check
    CHECK_NE(src_ptr, nullptr);
    CHECK_NE(dest_ptr, nullptr);
    if (byte_size == 0) {
        return;
    }

    // memcpy
    if (memcpy_kind == MemcpyKind::kMemcpyCPU2CPU) {
        std::memcpy(dest_ptr, src_ptr, byte_size);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCPU2CUDA) {
        _cudaMemcpyWraper(src_ptr, dest_ptr, byte_size, memcpy_kind, stream);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CPU) {
        _cudaMemcpyWraper(src_ptr, dest_ptr, byte_size, memcpy_kind, stream);
    } else if (memcpy_kind == MemcpyKind::kMemcpyCUDA2CUDA) {
        _cudaMemcpyWraper(src_ptr, dest_ptr, byte_size, memcpy_kind, stream);
    } else {
        LOG(FATAL) << "Unknown memcpy kind: " << static_cast<int>(memcpy_kind);
    }
    if (need_sync) {
        cudaDeviceSynchronize();
    }
}

void DeviceAllocator::memset_zero(
    void *ptr, size_t byte_size, void *stream, bool need_sync
) {
    CHECK(device_type_ != base::DeviceType::kDeviceUnknown);
    if (device_type_ == base::DeviceType::kDeviceCPU) {
        std::memset(ptr, 0, byte_size);
    } else {
        if (stream) {
            cudaStream_t stream_ = static_cast<cudaStream_t>(stream);
            cudaMemsetAsync(ptr, 0, byte_size, stream_);
        } else {
            cudaMemset(ptr, 0, byte_size);
        }
        if (need_sync) {
            cudaDeviceSynchronize();
        }
    }
}

} // namespace base
