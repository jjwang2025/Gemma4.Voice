# google/gemma-4-12B-it

## 概览

`google/gemma-4-12B-it` 是 Gemma 4 系列里的指令微调版 `12B Unified` 模型。它由 Google 以公开 Hugging Face 仓库形式发布，主路径基于 `transformers`，许可证为 `Apache-2.0`。这次调研时看到的公开元数据更新时间为 `2026-06-03`，可以视为目前较新的 Gemma 4 公共版本之一。

对 `Gemma4.Voice` 来说，这个模型的价值和 `onnx-community/gemma-4-E2B-it-ONNX` 不一样。E2B ONNX 的优势是已经非常接近 CPU 可部署形态；12B 的优势则是能力更强、模态更统一，未来更有机会支撑更高质量的语音理解、转写后分析，以及更完整的多模态工作流。

## 官方身份信息

公开 Hugging Face 元数据显示：

- 模型 ID：`google/gemma-4-12B-it`
- 基座模型：`google/gemma-4-12B`
- 架构：`Gemma4UnifiedForConditionalGeneration`
- 模型类型：`gemma4_unified`
- pipeline tag：`any-to-any`
- 主要库：`transformers`
- 许可证：`Apache-2.0`

仓库结构是标准 Hugging Face 模型仓库，而不是 ONNX 部署包。公开可见的核心文件包括：

- `model.safetensors`
- `config.json`
- `generation_config.json`
- `processor_config.json`
- `tokenizer.json`
- `tokenizer_config.json`
- `chat_template.jinja`
- `README.md`

这说明官方当前主推的使用方式还是 `transformers`，而不是像 E2B 那样的 CPU 优先 ONNX 部署包。

## “Unified” 的含义

12B 模型最值得注意的一点，就是它被官方定位为 **Unified** 模型。这和较小的 E 系列模型相比，是非常关键的架构差异。

在 E2B、E4B 这一路线中，多模态通常是先经过独立的模态编码器，再把结果送入语言模型。而 12B Unified 官方描述为一种不再依赖独立大编码器的统一设计：

- 原始图像 patch 直接投影到语言模型 embedding 空间
- 原始音频 waveform 直接投影到语言模型 embedding 空间
- 所有模态最终都流入同一个 decoder-only transformer

对下游应用来说，这意味着它不只是“更大的语音识别模型”，而是一个真正偏通用型的多模态推理模型，只不过它也原生支持音频任务。

## 核心规格

根据官方 README 和仓库配置，可以整理出以下关键参数：

| 属性 | 数值 |
| --- | --- |
| 总参数量 | `11.95B` |
| 公开 safetensors 元数据中的权重精度 | `BF16` |
| 层数 | `48` |
| Hidden size | `3840` |
| MLP 中间层大小 | `15360` |
| Attention heads | `16` |
| KV heads | `8` |
| Head dimension | `256` |
| Sliding window | `1024` tokens |
| 词表大小 | `262144` |
| 支持模态 | `text`、`image`、`audio` |
| 系列文档宣传的上下文长度 | `256K` tokens |
| 当前仓库 `text_config.max_position_embeddings` | `131072` |

最后两项需要分开理解。Gemma 4 系列文档里把 12B 宣传为 `256K` 上下文模型，但当前仓库配置里暴露出来的是 `max_position_embeddings=131072`。因此在真正做产品或工程规划时，不要只看宣传值，最好在你实际部署的推理栈里确认有效上下文上限。

## 音频能力

对 `Gemma4.Voice` 而言，12B 最重要的一个关注点就是它仍然保留了原生音频能力。

官方文档明确说明音频支持覆盖：

- `E2B`
- `E4B`
- `12B`

上游 README 里也直接给了音频示例，使用的是：

- `AutoProcessor`
- `AutoModelForMultimodalLM`
- 带 `audio` 内容块的 chat prompt

这意味着在官方 `transformers` 路径中，音频已经是第一类输入，而不是临时拼接出来的旁路能力。

实际含义包括：

- 可以做语音转写 prompt
- 可以做语音到文本翻译类 prompt
- 可以把音频理解和更强的推理能力结合起来

但它仍然更适合被理解为“会处理音频的多模态助手模型”，而不是“专门做精确时间对齐的 ASR 引擎”。

## 能力定位

官方 README 里的 benchmark 表明，12B 位于 Gemma 4 系列的中间高位。

可以参考的几个公开分数：

- `MMLU Pro`：`77.2%`
- `AIME 2026 no tools`：`77.5%`
- `LiveCodeBench v6`：`72.0%`
- `GPQA Diamond`：`78.8%`
- `MMMLU`：`83.4%`
- `CoVoST`：`38.5`
- `FLEURS`：`0.069`（越低越好，且官方注释说明未包含中文）

这些数字说明它并不是仅仅在 E2B 基础上“加了一点音频能力”，而是一个整体能力更强的多模态通用模型。

## 存储空间

Hugging Face 元数据显示，这个模型有 `11,959,730,224` 个 BF16 参数。

由此可以得到一个比较实用的存储量级判断：

| 项目 | 近似大小 |
| --- | --- |
| 原始 BF16 权重体量 | `22.28 GiB` |
| 如果成功转成纯权重 INT8 的理论体量 | `11.14 GiB` |
| 如果成功转成纯权重 INT4 的理论体量 | `5.57 GiB` |
| Hugging Face 仓库公开 `usedStorage` | `66.85 GiB` |

需要注意：

- `22.28 GiB` 更接近这个模型主权重文件的实际量级。
- 上面的 INT8、INT4 只是理论上的纯权重量级估算，不代表官方已经提供了对应发布物。
- 仓库级别的 `usedStorage` 不能简单当成本地实际下载量，但它仍然说明这个仓库整体并不轻。

## 运行空间与 KV Cache

部署时不能只看权重大小，还要考虑：

- 框架自身额外开销
- 激活张量
- 模态预处理张量
- 解码时的临时 buffer
- 随生成增长的 KV cache

基于公开配置中的 `48` 层、`8` 个 KV heads、`head_dim=256`，如果用一个简化的 BF16 下界估算，KV cache 的增长量大约是：

- 每个缓存 token 约 `393,216` 字节
- 约等于 `0.375 MiB / token`
- `1024` 个缓存 token 约 `384 MiB`
- `8192` 个缓存 token 约 `3.0 GiB`

这只是简化估算，真实运行时还会受到 Gemma 4 的混合注意力、滑动窗口实现方式以及具体 runtime buffer 策略影响。但它已经足够说明一个事实：就算主权重能装下，长上下文生成带来的 KV cache 也会继续显著抬高内存压力。

## 在这台机器上的可行性

这次调研时，这台机器的观测状态是：

- 总物理内存约 `31.51 GiB`
- 当前可用内存约 `10.44 GiB`
- `D:` 盘可用空间约 `95.81 GiB`
- `torch` 版本：`2.11.0+cpu`
- CUDA 可用性：`False`

由此可以得出非常明确的判断。

### 可以做的事

- 阅读模型卡、做方案设计：可以
- 下载官方仓库到本地缓存：可以，磁盘空间够
- 在文档里把 12B 作为未来后端目标：可以

### 当前这台机器上不适合做的事

- 把原始 Hugging Face BF16 版本当成一个舒适的 CPU 本地运行目标
- 期待 CPU-only 下有实用的推理延迟
- 在长上下文或大多模态输入下稳定运行

原因很直接：`22.28 GiB` 的主权重，再加上框架开销和生成阶段 buffer，已经非常接近这台机器的总内存上限。而在这次测量时，真正可用的空闲内存只有 `10.44 GiB` 左右。即使勉强依赖分页文件让它部分跑起来，速度和稳定性也很难符合这个项目“CPU-first、可实际使用”的目标。

## 值不值得在这里下载

这个问题要看目标。

### 值得下载的情况

- 你想先把原始 checkpoint 缓存到本地，留给未来更强的机器使用
- 你想离线查看 processor、tokenizer、config 等资源
- 你计划后续测试基于同一模型导出的量化版本或替代 runtime 格式

### 不值得优先做的情况

- 你的目标是当前这台机器上立刻做 CPU 本地转写
- 你的目标是给 `Gemma4.Voice` 增加一个实用的 CPU 默认后端
- 你的目标是低延迟、可部署的 ASR 体验

换句话说，在这台机器上下载原始 12B，更像是一个“存储准备动作”，而不是一个“马上能高质量本地推理”的动作。

## 生态现状与项目影响

这次通过 Hugging Face API 做快速搜索时，没有看到一个很明确的公开 `onnx-community/gemma-4-12B-it-ONNX` 仓库结果。

这一点对 `Gemma4.Voice` 很重要，因为当前项目是围绕以下几个前提设计的：

- `onnxruntime`
- CPU 推理
- 更接近部署形态的多模态推理循环

官方 12B 目前还不能自然落到这条技术路径上。要在这个仓库里支持它，比较像是新增一条完全不同的 backend，而不是在现有 E2B ONNX 实现上做一点小扩展。

## 对 Gemma4.Voice 的建议

当前更合理的定位是：

- 继续把 `onnx-community/gemma-4-E2B-it-ONNX` 作为实用的 CPU-first 默认后端
- 把 `google/gemma-4-12B-it` 视为调研目标和未来增强方向
- 等下面任一条件满足后，再考虑正式接入：
  - 出现更适合部署的 ONNX 导出版本
  - 出现成熟稳定的低比特 runtime 路线
  - 本地硬件升级，尤其是更大的 GPU 或更多系统内存

也就是说，`Gemma 4 12B` 在战略上值得持续关注，但它现在还不是这个仓库最合适的默认引擎。
