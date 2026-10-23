# 作业一实验报告：Qwen3.5-0.8B Ascend 计算路径适配

> 更新说明：本版依据 `hw1.md` 更新了当前代码验证结果，补充 4096 上下文服务配置、长输入请求结果、单 token 注意力修复及测试记录。工作区未找到单独的 `作业1.md` 模板。

## 小组成员贡献

- 小组成员姓名及分工：当前工作区未提供名单与分工记录，请提交前由小组补全。
- 本轮已完成的工程工作：环境检查与配置、Ascend 算子适配、数值测试、模型服务与性能测量、文档整理。

## 1. 目标与结论

目标是在 Ascend 910C 单个 chip 上运行 Qwen3.5-0.8B，建立计算路径正确性与性能基线。模型包含 18 个 Gated DeltaNet（GDN）层和 6 个全注意力层。当前代码已在单个 Ascend chip 上按 4096 最大上下文配置通过服务健康检查，并成功完成 1673-token 长输入请求；此前的 2048 配置验证还连续完成了短输入、503-token 长输入和重复短请求。算子测试与 Triton-Ascend 设备验证通过。性能基线另有四组输入/输出长度测量记录。

## 2. 环境与启动配置

| 项目 | 配置/版本 |
|---|---|
| 硬件 | Ascend 910，目标逻辑设备 5（物理卡 2、chip 1）；PyTorch 设备名为 Ascend910_9362 |
| CANN | 9.0.0 |
| Python | 3.11，本地 `.venv` 使用系统站点包 |
| PyTorch / torch_npu | 2.9.0+cpu / 2.9.0.post2 |
| Transformers | 5.12.0 |
| Triton / Triton-Ascend | 3.2.0 / 3.2.1 |
| 模型 | Qwen3.5-0.8B，本地 safetensors 共 488 个权重键 |
| 调度器 | 本地构建 tokenspeed-scheduler 0.1.41 |

运行前加载 CANN 环境，启用本地环境与 NPU 内核：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source .venv/bin/activate
export TOKENSPEED_KERNEL_BACKEND=npu
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python"
export ASCEND_RT_VISIBLE_DEVICES=5
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost
```

本轮模型服务采用 BF16 权重/激活、FP32 Mamba 状态、MHA 注意力、贪心采样；禁用 Prefill 图、PDL 和 KVStore。4096 配置复核使用 `max_num_seqs=4`、`max_total_tokens=16384` 和 `chunked_prefill_size=4096`；`--retraction-snapshot-ratio 0` 关闭快照池，便于单机验证。

本次 4096 配置服务启动命令如下（模型路径为本地已下载权重）：

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
source .venv/bin/activate
export TOKENSPEED_KERNEL_BACKEND=npu
export PYTHONPATH="$PWD/python:$PWD/tokenspeed-kernel/python:$PWD/tokenspeed-kernel-npu/python"
export ASCEND_RT_VISIBLE_DEVICES=5
export NO_PROXY=127.0.0.1,localhost
export no_proxy=127.0.0.1,localhost

python -m tokenspeed.cli serve ../models/Qwen3.5-0.8B \
  --language-model-only \
  --served-model-name qwen35-0.8b \
  --device npu \
  --dtype bfloat16 \
  --mamba-ssm-dtype float32 \
  --kv-cache-dtype auto \
  --attention-backend mha \
  --sampling-backend greedy \
  --disable-prefill-graph \
  --disable-pdl \
  --disable-kvstore \
  --retraction-snapshot-ratio 0 \
  --max-model-len 4096 \
  --max-num-seqs 4 \
  --max-total-tokens 16384 \
  --chunked-prefill-size 4096 \
  --prefix-granularity 128 \
  --max-cudagraph-capture-size 4 \
  --cudagraph-capture-sizes 1 2 4 \
  --disable-autotune \
  --host 127.0.0.1 \
  --port 31992
```

4096 端到端复核配置为 `max_model_len=4096`、`max_total_tokens=16384`、`chunked_prefill_size=4096`。`NO_PROXY` 必须包含 localhost，否则本地 gRPC 健康探测可能被转发到环境代理。一次 `max_total_tokens=4096` 的试运行只分配 5 个 device pages，调度器拒绝接纳请求；改用仓库配方的 16384 后启动和健康检查成功。

## 3. 计算路径和算子支持情况

| 功能 | 主要输入/输出 | 类型 | 实现来源与本轮适配 | 状态与测试 |
|---|---|---|---|---|
| 线性层/矩阵乘 | hidden `[T,H]`、权重 `[H,O]` -> `[T,O]` | BF16 | TokenSpeed 线性层和 CANN 矩阵算子 | 已通过模型 Prefill/Decode |
| RMSNorm / Gemma RMSNorm | `[T,H]` -> `[T,H]` | BF16，统计 FP32 | NPU 内核；Gemma 权重语义、残差融合走 NPU 路径 | 数值用例通过 |
| RoPE | Q/K 及位置 -> Q/K | BF16 | `torch_npu`/NPU 算子 | 设备基础验证和模型路径通过 |
| 全注意力 Prefill | Q `[T,Hq,D]`、K/V `[T,Hkv,D]` -> `[T,Hq,D]` | BF16 | CANN `npu_fused_infer_attention_score`；D=256 时逐请求切到 BSH，其他维度使用 TND | MHA 对照测试通过；D=256 的因果 Prefill 有专项测试 |
| 全注意力 Decode | Q `[B,Hq,D]`、分页 KV -> `[B,Hq,D]` | BF16 | CANN 分页注意力 | 分页 KV 数值对照通过 |
| GDN Prefill | Q/K/V `[1,T,H,K]`、状态 `[N,H,K,V]` -> 输出及最终状态 | BF16 输入/输出，状态 FP32 | 新增 Ascend recurrent GDN 适配；以最多 8 token 为一段调用原生递推算子并传递段间状态 | 与 FP32 参考递推对照，含长度 1、9、多序列不等长 |
| GDN Decode | 单步 Q/K/V、状态池和 slot 索引 -> 输出及更新状态 | BF16 输入/输出，状态 FP32 | 新增 NPU decode；临时转换为原生算子要求的 BF16 状态，写回对应请求 slot | 多步更新、无效 slot 填充和请求隔离测试通过 |
| GDN 门控/归一化 | 门控向量与投影 -> 门控输出 | BF16 | 新增 NPU gated RMSNorm 路径 | 门控前后归一化和 sigmoid 两类用例通过 |
| 因果卷积 | 每层输入及卷积状态 -> 输出及更新状态 | BF16 | TokenSpeed 已有 NPU Triton 实现 | 端到端运行记录中完成 Prefill/Decode |
| 采样 | logits -> token id | BF16 logits / 整数 id | TokenSpeed greedy sampler | 多次完整生成请求成功 |

### 关键路径：GDN 状态更新

模型 GDN 层先计算投影、Q/K 归一化与 RoPE，再生成门控参数，最终进入 GDN 核心递推。新增适配接收 TokenSpeed 统一接口中的序列边界和 slot 状态池；Prefill 对不同请求按 `cu_seqlens` 切片，并将长序列分成不超过 8 token 的段，逐段将上一段最终状态作为下一段初态。这样绕过当前 CANN 原生递推对较长序列输出 NaN 的限制，同时保持完整递推，不截断 token。Decode 根据 slot 索引读取和写回请求状态；`-1` 等无效 slot 不写入有效请求状态。因 CANN 原生算子仅接受 BF16 状态，算子调用边界转换状态精度，TokenSpeed 状态池对外仍保持 FP32。

## 4. 正确性验证

### 参考实现和判据

- MHA 参考：PyTorch 显式 QK 点积、softmax 和 AV；BF16 结果使用 `atol=0.02, rtol=0.02`，256 维专项使用 `atol=0.03, rtol=0.03`。
- GDN 参考：FP32 逐 token recurrent 更新；比较 Prefill 输出、最终状态以及 Decode 多步输出/状态。测试覆盖单 token、超过单段限制的 9 token、多请求不等长 `[3,11]`，以及两个有效 slot 和一个无效 slot。
- RMSNorm/Gemma 与 gated RMSNorm：对照 PyTorch 公式並检查残差更新和两种门控配置。
- 算子测试命令：`pytest -q tokenspeed-kernel-npu/test/test_gdn.py tokenspeed-kernel-npu/test/test_mha.py tokenspeed-kernel-npu/test/test_layernorm.py`。
- 设备基础验证：`python test/ci_system/verify_triton_ascend.py`，检测到 Ascend910_9362，Triton-Ascend 验证成功。

本轮新增/修改的 NPU 测试记录：GDN 4 项通过；MHA 6 项通过（含单 token、256 维 Prefill 回归）；LayerNorm 9 项通过。当前合并重跑共 **19 项通过**。Triton-Ascend 设备验证也通过。测试环境有 CANN 文件所有者不一致、NPU 内部格式回退等 warning，不影响测试结果。

当前代码的服务健康检查通过，启动预热请求成功。随后在同一个服务进程中按顺序执行以下请求，均收到 HTTP 成功响应，输出与预期答案一致：

| 顺序 | 输入 token | 输出 token | 提示与检查结果 |
|---:|---:|---:|---|
| 1 | 26 | 2 | `What is 2 + 2?`，输出 `4` |
| 2 | 24 | 2 | `What is the capital of France?`，输出 `Paris` |
| 3 | 503 | 3 | 重复上下文中询问 reference color，输出 `cobalt` |
| 4 | 26 | 2 | 再次询问 `2 + 2`，仍输出 `4` |

四次请求固定 `temperature=0`、`max_tokens=16`，设置 `enable_thinking=false`。该检查证明当前模型路径能在短/长输入间连续 Prefill/Decode，并对不同请求保持独立状态；它是功能级答案检查，不是与 Transformers 参考模型逐元素比较 logits。服务日志保存在本次运行的 `/tmp/assignment-e2e-current.log`。

随后使用仓库配方的 4096 上下文配置，在同一单 chip 服务上执行 1673-token 输入，回答 `cobalt`；之后服务日志还记录了一次 26-token 短输入请求完成。长输入请求结果及配置日志保存在 `/tmp/assignment-e2e-4096.log`。本次验证的是 4096 上下文配置能够启动并处理长输入，不代表已将 Prefill 输入长度推至 4096 token 边界。

## 5. 性能基线

配置为 BF16、贪心、`ignore_eos=true`，每组 12 次热身后请求；输入分别为 128、2048 token，输出分别固定为 32、128 token。TTFT 为流式响应首个输出 chunk 的客户端计时；Decode 每 token 取 `(总耗时 - TTFT) / (输出 token 数 - 1)`。表内为 12 次测量中位数。

| 输入 token | 输出 token | TTFT (ms) | 总耗时 (s) | Decode (ms/token) |
|---:|---:|---:|---:|---:|
| 128 | 32 | 159.6 | 1.583 | 45.91 |
| 128 | 128 | 157.2 | 5.851 | 44.83 |
| 2048 | 32 | 1032.2 | 2.426 | 44.95 |
| 2048 | 128 | 1031.4 | 6.719 | 44.74 |

HBM 总容量为 65,536 MB。`npu-smi info` 在记录时显示目标 chip 5 的占用约 14,416 MB；该设备由多进程共享，因此此值是设备级占用，不能全部归因于本模型。服务日志的模型加载内存估计约 13.94 GiB，和设备级采样处于相同量级。

Ascend Profiler 采样中，128 token Prefill 的单个冷启动 step 约 70.6 ms kernel 累计时长，其中因果卷积约 49.6 ms、GDN 递推约 9.6 ms；热身后的 Decode 单步约 7.1 ms kernel 累计时长，其中因果卷积约 2.9 ms、矩阵乘约 1.6 ms。2048 token Prefill TTFT 约为 1.03 s，主要随输入增长；Decode 则稳定约 45 ms/token。主要优化方向是因果卷积的 NPU kernel/调用开销和 Decode 阶段的调度与小算子开销。Profiler 未采集 HBM 带宽计数器，不能把 kernel 时长直接解释为访存带宽或算术强度。

性能数据来自先前四组 benchmark 运行，每组取 12 个热身后样本；该基线不是在本次单 token mask 形状修复后重新采集的。修复仅改变 256 维 BSH Prefill 的 mask 维度表达，性能数据可作为当前实现的参考基线，不应视为该修复后的严格回归测量。HBM 数值是共享设备级采样，也不是模型进程独占占用。

## 6. 主要适配障碍与待办

1. CANN 9.0 的 TND 融合注意力不支持 Qwen3.5 全注意力层的 256 维 head；当前 Prefill 对该 head size 改用逐请求 BSH，Decode 使用分页 BSH。当前 mask 缓存上限为 2048，故超过 2048 的 256 维 Prefill 尚未验证。
2. CANN recurrent GDN 对长序列的原生调用会产生 NaN，且状态接口要求 BF16；本轮通过最多 8 token 分段和接口处 BF16 转换解决数值问题，同时保留 FP32 状态池。
3. 已在 4096 上下文配置下完成 1673-token 输入推理，但尚未用接近 4096 token 的输入验证边界；当前 256 维注意力 mask 的尺寸上限为 2048，继续扩展上下文前需处理并测试该限制。
4. 模型路径按作业要求只验证 Language Model，不包含多模态视觉路径；MTP/speculative decoding 未实现，使用普通贪心 Decode。

## 7. 文件变更索引

核心适配位于 `tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/gdn.py`、`mha.py`、`layernorm.py`，注册位于 `tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/triton.py` 与 `ops/layernorm/ascend.py`；模型 NPU 初始化兼容性修正在 `python/tokenspeed/runtime/models/qwen3_5.py` 和 `runtime/layers/layernorm.py`。测试分别位于 `tokenspeed-kernel-npu/test/test_gdn.py`、`test_mha.py`、`test_layernorm.py`。启动参数和使用说明在 `docs/recipes/models.md`、`tokenspeed-kernel-npu/README.md`。
