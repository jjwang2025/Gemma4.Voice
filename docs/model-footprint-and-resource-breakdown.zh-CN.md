# 模型体量与资源拆解

## 文档范围

这份文档从 `Gemma4.Voice` 项目的实际使用角度，说明 `onnx-community/gemma-4-E2B-it-ONNX` 的参数规模、存储占用和运行时内存特征。

重点讨论的是当前项目实际使用的 **音频转写路径**：

- `audio_encoder`
- `embed_tokens`
- `decoder_model_merged`

视觉编码器会作为参考列出，但当前音频 CLI 流程并不会加载它。

## 统计依据

下面的数值来自本机实际使用过的 Hugging Face 快照：

- 模型：`onnx-community/gemma-4-E2B-it-ONNX`
- 快照磁盘总大小：`30.82 GiB`
- 本次检查的快照版本：`9f4bef82ea6e296bc69f8a2f5939f73af81b07a6`

文档里刻意区分了三个概念：

1. **原始参数计数**：根据 base 导出中的 ONNX graph initializer 统计。
2. **存储占用**：根据实际 `.onnx` 与 `.onnx_data*` 文件大小统计。
3. **运行时工作集**：结合加载权重、激活、KV cache 进行估算。

这三类数字不能直接混为一谈。

## E2B 命名与原始张量计数的区别

Gemma 4 E2B 的 `E2B` 是 **effective parameter size** 命名，不等于把部署工件里所有标量逐个加起来后的字面参数个数。

直接检查本项目实际使用的 ONNX 音频路径后，可以看到 base 导出在三张核心图中一共暴露出大约 **54.27 亿个 initializer 标量值**。

这并不与 `E2B` 命名冲突，原因在于：

- Gemma 家族命名采用的是 effective size 口径
- ONNX 部署包被拆成了多个子图和辅助张量
- 部署工件里的原始张量数量，不等于模型家族命名标签

## 各模块原始参数规模

下面的统计来自 **base ONNX** 图中的 initializer。对这个仓库来说，这是讨论各大模块资源占比时最有代表性的口径。

| 模块 | 原始 initializer 标量值 | 约合规模 | 占音频路径比例 |
| --- | ---: | ---: | ---: |
| `audio_encoder` | 294,762,669 | 0.295B | 5.43% |
| `embed_tokens` | 2,751,463,434 | 2.751B | 50.70% |
| `decoder_model_merged` | 2,380,427,125 | 2.380B | 43.87% |
| **总计** | **5,426,653,228** | **5.427B** | **100%** |

### 如何理解这张表

- **嵌入路径** 是原始存储里最大的单块。
- **解码器** 略小一些，但依然是运行时的主要成本来源。
- **音频编码器** 相对更小，但无论磁盘还是运行时都不是可以忽略的体量。

## 各变体的存储占用

对 `Gemma4.Voice` 来说，最有意义的不是整个仓库有多大，而是做语音转写时真正需要的 **音频推理路径** 有多大。

### 仅音频路径

下表包含：

- `audio_encoder*`
- `embed_tokens*`
- `decoder_model_merged*`

不包含 tokenizer 文件，也不包含 vision encoder。

| 变体 | Audio encoder | Embed tokens | Decoder | 音频路径总计 |
| --- | ---: | ---: | ---: | ---: |
| `base` | 1.10 GiB | 10.25 GiB | 8.87 GiB | **20.22 GiB** |
| `q4` | 0.18 GiB | 1.64 GiB | 1.74 GiB | **3.56 GiB** |
| `quantized` | 0.32 GiB | 2.96 GiB | 2.83 GiB | **6.11 GiB** |

### Vision encoder 参考值

当前 CLI 在 ASR 场景下不会加载视觉路径，但如果未来扩展到图像+音频多模态，这部分占用也要考虑进去：

| 变体 | Vision encoder |
| --- | ---: |
| `base` | 0.63 GiB |
| `q4` | 0.10 GiB |
| `quantized` | 0.18 GiB |

如果未来把视觉路径也一并加载，粗略总占用会变成：

- `base`：约 `20.84 GiB`
- `q4`：约 `3.66 GiB`
- `quantized`：约 `6.30 GiB`

## 存储缩减结论

以 base 音频路径为基准：

- `q4` 约为 base 存储的 **17.59%**
- `quantized` 约为 base 存储的 **30.24%**
- 单看音频路径，`q4` 相比 `base` 大约节省 **16.66 GiB**

这也是为什么在 CPU 本地部署上，`q4` 非常有吸引力。

## 运行时内存模型

运行时内存并不只是把模型文件映射进内存这么简单。对 CPU 推理来说，主要由下面几部分组成：

1. 已加载的 ONNX 权重
2. ONNX Runtime 内部 buffer 与 arena
3. 输入特征张量
4. embedding 和 decoder 激活
5. 自回归解码中的 **KV cache**

### 1. 权重常驻内存

作为规划口径，可以先把音频路径的磁盘占用，视为权重常驻内存的大致数量级：

- `base`：属于 **20 GiB 级别** 的超大常驻权重集，还不含额外 buffer
- `q4`：属于 **数 GiB 级别**
- `quantized`：介于两者之间

实际 RSS 会受操作系统分页、ONNX Runtime arena 策略、external data 是否立即触达等因素影响，因此不会和磁盘大小完全相等。

### 2. 音频输入张量

processor 配置显示当前音频前处理大致采用：

- 16 kHz 采样率
- 128 维输入特征
- 每秒约 100 帧的特征张量

对一段 30 秒音频来说，原始 `input_features` 的内存只有几 MiB 量级，远小于权重本体。因此音频输入张量 **不是主要内存瓶颈**。

### 3. Decoder 的 KV cache

当前项目使用的 decoder 导出，在 ONNX schema 里为每层暴露了一对 KV cache。结合图结构可得到：

- 28 个 sliding attention 层，cache 宽度为 `256`
- 7 个 full attention 层，cache 宽度为 `512`
- 当前检查到的导出里，KV cache 使用的是 `float32`

因此可以得到一个近似值：

- **每生成 1 个 token，KV cache 大约增加 86,016 字节**
- 也就是约 **84.0 KiB/token**

按 batch size = 1 估算，KV cache 增长大致如下：

| 生成 token 数 | 约合 KV cache |
| ---: | ---: |
| 128 | 10.5 MiB |
| 256 | 21.0 MiB |
| 512 | 42.0 MiB |
| 1024 | 84.0 MiB |
| 2048 | 168.0 MiB |
| 4096 | 336.0 MiB |

这点很重要：即使模型权重做了量化，当前 decoder cache 路径依然会以更传统的浮点形式增长，所以运行时内存不会按权重压缩比例同步缩小。

### 4. 首轮解码的激活峰值

第一轮解码通常是最重的，因为它同时包含：

- 音频特征编码
- token embedding lookup
- per-layer inputs 准备
- 尚未建立完整 cache 的首轮 decoder 前向

后续 token 的单步成本通常更低，但 KV cache 会持续增长。因此不同 prompt、不同输出长度下，峰值内存不一定在完全相同的时刻出现。

## 面向 CPU 部署的实际建议

### `base`

- 适合作为高内存配置使用
- 单是音频路径存储就约 `20.22 GiB`
- 只有在机器有足够 RAM 余量时才推荐尝试

### `q4`

- 是当前项目最推荐的 CPU 起点
- 音频路径占用降到约 `3.56 GiB`
- 更容易在本地 CPU 工作站上稳定常驻

### `quantized`

- 比 `base` 小，但不如 `q4` 小
- 不能因为它叫 quantized 就默认更快

## 为什么“更小”不一定“更快”

在本项目的本地测试里，`q4` 是最有吸引力的 CPU 变体，而仓库里的 `quantized` 虽然磁盘占用小于 `base`，实际语音转写速度却明显更慢。

这是因为运行速度不仅由文件大小决定，还取决于：

- ONNX Runtime 中采用的算子实现
- 内核可用性
- cache dtype 与内存带宽压力
- 图结构本身
- CPU 向量化行为

因此在部署时，正确的问题不是“它有多小”，而是“它在这个 runtime 和这台硬件上到底执行得怎么样”。

## 对 `Gemma4.Voice` 的意义

结合当前项目目标，可以得到比较清晰的结论：

- `embed_tokens` 和 `decoder_model_merged` 是主要体量来源
- `q4` 对 CPU 语音转写提供了最明显的存储优势
- 运行时内存不仅受权重文件影响，也受 KV cache 增长影响
- 时间戳类 prompt 往往会生成更多 token，间接抬高运行时工作集

如果后续项目要支持更长上下文、更大 batch，或者图像+音频联合输入，就应该重新审视 decoder cache 和总常驻图集合的资源上限。

## 注意事项

- 上面的参数量来自 ONNX graph initializer 统计，不是模型家族营销命名口径
- 存储数据基于本次检查的快照版本，若上游重新导出，数值可能变化
- 运行时内存部分结合了图 schema 和部署行为做估算，并不是对所有变体都做了完整 profiler 采样
- 这份文档描述的是 `Gemma4.Voice` 当前以 CPU 为主的实现路径，不代表 Gemma 4 的全部运行配置
