# Layer的实现
Layer分为无参数网络层和有参数网络层，网络层需要管理网络的输入和输出，有参数网络层需要额外考虑网络权重。

我们先来考虑一个`Layer`类应该有哪些成员变量。

与PyTorch中写`nn.Module`不同的是，我们在`Layer`实例化时，就需要为输入输出设置好容器，这里我们使用`vector`作为多个输入输出张量的容器。

```cpp
// 首先是基础设置，如名称、类型、数据类型什么的
std::string layer_name_;
LayerType layer_type_;
base::DataType data_type_;
base::DeviceType device_type_;
// 之后是网络层的输入输出以及计算所需的配置
// 输入输出使用vector作为容器，cuda_config_是全局共用一份，所以是shared_ptr
std::vector<tensor::Tensor> inputs_;
std::vector<tensor::Tensor> outputs_;
std::shared_ptr<kernel::CudaConfig> cuda_config_;
// 有参数的层还要保存权重
int32_t group_size_ = 0;
bool is_quant_layer_ = false;
tensor::Tensor scales_;
std::vector<tensor::Tensor> weights_;
```

我们先考虑无参数层的`Layer`，构造函数需要实现的功能应该是：设置好输入输出的容器。对于不同的`Layer`，其输入输出张量的数量肯定是不一样的，比如一个加法层，`inputs_`的长度应该是2，`outputs_`的长度应该是1。

接着我们需要保证在前向传播时，输入输出的维度应当是正确的，因此需要写一个`check()`函数用于这些检查。

最后就是最重要的`forward()`，它的基本功能应该包括`check()`和调用算子进行计算，并返回状态码。