#ifndef WEI_INFER_BASE_BASE_H_
#define WEI_INFER_BASE_BASE_H_

#include <cstdint>
#include <glog/logging.h>
#include <string>

#define UNUSED(expr)                                                           \
    do {                                                                       \
        (void)(expr);                                                          \
    } while (0)

namespace base {
enum class DeviceType : uint8_t {
    kDeviceUnknown = 0,
    kDeviceCPU = 1,
    kDeviceCUDA = 2,
};

enum class DataType : uint8_t {
    kDataTypeUnknown = 0,
    kDataTypeFp32 = 1,
    kDataTypeInt8 = 2,
    kDataTypeInt32 = 3,
};

inline size_t DataTypeSize(DataType data_type) {
    if (data_type == DataType::kDataTypeFp32) {
        return sizeof(float);
    } else if (data_type == DataType::kDataTypeInt8) {
        return sizeof(int8_t);
    } else if (data_type == DataType::kDataTypeInt32) {
        return sizeof(int32_t);
    } else {
        return 0;
    }
}

#define KB (1024)
#define MB (1024 * 1024)
#define GB (1024 * 1024 * 1024)

} // namespace base

#endif // WEI_INFER_BASE_BASE_H_