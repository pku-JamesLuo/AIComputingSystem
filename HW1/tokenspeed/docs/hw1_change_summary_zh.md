# 作业一变更摘要

| 文件 | 变更 | 必要性 |
|---|---|---|
| `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py` | 新增 Ascend GDN Prefill/Decode，按 8 token 分段递推，正确读写 slot 状态 | CANN 长序列原生递推会出现 NaN，且状态精度接口不同；模型需要完整 GDN 路径 |
| `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/triton.py` | 注册 NPU GDN 实现和能力信息 | 让统一 attention/GDN 调度接口能够选择 Ascend 算子 |
| `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/mha.py` | Ascend MHA Prefill 支持 256 维 head 的 BSH 回退路径，并传入 `[1,1,S,S]` 因果 mask | CANN 9.0 TND 路径不接受 Qwen3.5 的 256 维 head；单 token Prefill 还要求 BSH mask 有显式 batch/head 维 |
| `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/layernorm.py`、`tokenspeed-kernel/python/tokenspeed_kernel/ops/layernorm/ascend.py` | 增加 gated RMSNorm 并注册 NPU 实现 | GDN 层需要门控归一化，不能退回未支持的 Triton 路径 |
| `python/tokenspeed/runtime/models/qwen3_5.py` | NPU 下不创建 CUDA stream | 模型初始化必须兼容没有 CUDA 的 Ascend 设备 |
| `python/tokenspeed/runtime/layers/layernorm.py` | Gemma RMSNorm 和融合 Add+Norm 在 NPU 上走统一 NPU 算子 | 避免调用仅适用于 CUDA 的 FlashInfer Gemma kernel |
| `tokenspeed-kernel-npu/test/test_gdn.py`、`test_mha.py`、`test_layernorm.py` | 新增 GDN 状态、多请求隔离、256 维及单 token 注意力和 NPU LayerNorm 数值测试 | 覆盖新增路径、边界形状和请求状态正确性 |
| `docs/recipes/models.md`、`tokenspeed-kernel-npu/README.md` | 添加 Qwen3.5-0.8B Ascend 配置与使用边界 | 记录可复用的环境和启动方式 |
| `docs/hw1_experiment_report_zh.md` | 添加中文实验报告：环境、算子表、数值验证、性能和限制 | 满足作业报告要求 |

**验证结果：**Triton-Ascend 设备验证通过；当前合并运行 GDN、MHA、LayerNorm 共 19/19 项通过。当前代码服务健康检查通过；4096 上下文配置下完成 1673-token 输入并回答 `cobalt`，2048 配置下连续完成短输入、长输入和重复请求。性能 benchmark 的四种输入/输出长度组合另有各 12 次热身后样本；其 HBM 数值是共享设备级采样。
