#ifndef WEI_INFER_SAMPLER_ARGMAX_SAMPLER_H_
#define WEI_INFER_SAMPLER_ARGMAX_SAMPLER_H_

#include "sampler.h"
#include <base/base.h>
namespace sampler {
class ArgmaxSampler : public Sampler {
  public:
    explicit ArgmaxSampler(base::DeviceType device_type)
        : Sampler(device_type) {}

    size_t sample(const float *logits, size_t size, void *stream) override;
};
} // namespace sampler
#endif // WEI_INFER_SAMPLER_ARGMAX_SAMPLER_H_
