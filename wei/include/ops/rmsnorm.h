#ifndef WEI_INFER_OPS_RMSNORM_H_
#define WEI_INFER_OPS_RMSNORM_H_

#include "layer.h"
namespace op {
class RmsNormLayer : public LayerParam {
  public:
    explicit RmsNormLayer(base::DeviceType device_type, int32_t dim);

    base::Status check() const override;

    base::Status forward() override;

  private:
    int32_t dim_ = 0;
};
} // namespace op
#endif // WEI_INFER_OPS_RMSNORM_H_
