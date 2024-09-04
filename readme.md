# wei-infer: 自制LLM推理框架实现LLAMA2推理
本仓库是本人对于[KuiperLlama](https://github.com/zjhellofss/KuiperLLama)的学习与复现，并尝试加入更多自己的东西，感谢原仓库作者提供如此完善的学习项目。

## 复现进度
- [x] 统一内存资源管理
  - [x] CPU端
  - [x] CUDA端
- [ ] Tensor类的实现
- [ ] 算子注册管理
- [ ] 算子类
  - [ ] Flash Attention
  - [ ] SGEMV
  - [ ] RMSNorm
  - [ ] SwiGLU
  - [ ] 其他
- [ ] Sampler
- [ ] 权重量化
- [ ] 权重加载
- [ ] 对话demo