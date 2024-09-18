#ifndef WEI_INFER_MODEL_CONFIG_H_
#define WEI_INFER_MODEL_CONFIG_H_

namespace model {
struct ModelConfig {
    int dim = 0;
    int hidden_dim = 0;
    int layer_num = 0;
    int head_num = 0;
    int kv_head_num = 0;
    int vocab_size = 0;
    int seq_len = 0;
};

struct TransformerConfig {
    int kv_dim_ = 0;
    int kv_mul_ = 0;
    int head_size_ = 0;
    int vocab_size_ = 0;
    int dim_ = 0;
    int hidden_dim_ = 0;
    int layer_num_ = 0;
    int head_num_ = 0;
    int kv_head_num_ = 0;
    int seq_len_ = 0;
    bool is_shared_weight_ = false;
};
} // namespace model
#endif // WEI_INFER_MODEL_CONFIG_H_
