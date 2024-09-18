# wei-infer: 自制LLM推理框架实现LLAMA2推理
在RTX3090上，推理LLAMA2-7B Int8量化的结果为59Token/s：
![chat-7b](note/img/recording.gif)

<!-- 本仓库是本人对于[KuiperLlama](https://github.com/zjhellofss/KuiperLLama)的学习与复现，并尝试加入更多自己的东西，比如Flash Attention/Decoding等，感谢原仓库作者提供如此完善的学习项目。 -->

同时仓库中也记录了本人在学习该项目时的笔记和个人见解，欢迎交流。[我的笔记](./note/00_index.md)

## 复现进度
- [x] 统一内存资源管理
  - [x] CPU端
  - [x] CUDA端
- [x] Tensor类的实现
- [x] 算子注册管理
- [x] 算子类
  - [x] Flash Attention
  - [x] SGEMV
  - [x] RMSNorm
  - [x] SwiGLU
  - [x] 其他
- [x] Sampler
- [x] 权重量化
- [x] 权重加载
- [x] 对话demo