#ifndef WEI_INFER_OPS_MATMUL_H_
#define WEI_INFER_OPS_MATMUL_H_

#include "layer.h"
#include <base/cuda_config.h>
namespace op {
class MatmulLayer : public LayerParam {
  public:
    explicit MatmulLayer(
        base::DeviceType device_type,
        int32_t dim0,
        int32_t dim1,
        bool is_quant_layer = false
    );

    base::Status check() const override;

    base::Status forward() override;

  private:
    int32_t dim0_ = 0;
    int32_t dim1_ = 0;
};
} // namespace op
#endif // WEI_INFER_OPS_MATMUL_H_
