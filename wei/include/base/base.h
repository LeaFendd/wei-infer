#ifndef WEI_INFER_BASE_BASE_H_
#define WEI_INFER_BASE_BASE_H_

#include <cstdint>
#include <glog/logging.h>
#include <string>

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
} // namespace base

#endif // WEI_INFER_BASE_BASE_H_