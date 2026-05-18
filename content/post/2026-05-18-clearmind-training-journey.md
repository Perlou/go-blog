+++
date = '2026-05-18T22:30:00+08:00'
draft = false
title = '炼丹: 从0开始训练出属于自己的大模型'
image = '/images/bg/madagascar-baobab-avenue.webp'
categories = ['AI', '技术']
tags = ['AI', 'LLM', '大模型', '深度学习']
+++


这段时间一直在做一件事——从零训一个属于自己的中文大模型。

不是套个 LoRA、不是基于现成 checkpoint 继续训，而是从随机初始化的权重开始，自己写架构、自己跑 Pretrain / SFT / DPO 全流程，最后发布到 HuggingFace 和 ModelScope，配一个能在浏览器里直接对话的 Demo。

项目叫 **ClearMind**。Base 版本（68.8M 参数）已经训完上线了，Plus 版本（486M）算过账之后暂时没训——后面会讲为什么。

先把成果摆出来：

- 🤗 **HuggingFace**：[Perlous/ClearMind-Base](https://huggingface.co/Perlous/ClearMind-Base)
- 🧠 **ModelScope**：[Perlou/ClearMind-Base](https://www.modelscope.cn/models/Perlou/ClearMind-Base)
- 💬 **在线 Demo**：[ClearMind 在线演示](https://www.modelscope.cn/studios/Perlou/ClearMind-Demo)
- 💻 **GitHub 源码**：[github.com/Perlou/clear-mind](https://github.com/Perlou/clear-mind)

「炼丹」是国内深度学习圈对训练神经网络的戏称，意思是过程像古代道士炼丹——配方齐全也不一定出货，loss 不收敛的时候真的会一头雾水盯着 tensorboard 发呆。

这篇文章想聊的不是教程，是踩坑日记。

## 为什么要做这件事

说实话，开始之前我犹豫了很久。

现在的开源模型一抓一大把，Qwen、DeepSeek、Llama 3 随便一个都比我自己能训出来的强好几个数量级。花一个月时间训一个 68M 的小模型，从结果导向看，意义不大。

但有件事是模型再大也替代不了的：**你不亲手训一次，永远只是"调 API 的人"**。

什么叫 warmup steps，什么叫 gradient accumulation，什么叫 EarlyStopping 在 val loss 平台期触发，什么叫 attention mask 在 valid 位置算出 NaN——这些东西你看十篇 paper 都不如自己被坑一次记得牢。

我的起点是 [minimind](https://github.com/jingyaogong/minimind)。这是一个非常优秀的中文小模型教学项目，作者 gongjy 把数据集、tokenizer、训练脚本都开源了，社区生态成熟。但我想做的不是"复刻 minimind"，而是**在同等参数规模下做得更扎实，最好能反超一些指标**。

具体做法是这样：
- **复用** minimind 的数据集和 tokenizer（chat_template / tool_call / `<think>` 标记），不重复造轮子
- **重写**训练框架，抽象出 `BaseTrainer` 父类，PreTrainer / SFTTrainer / DPOTrainer 共享逻辑，不像 minimind 那样 9 个独立训练脚本各自维护
- **升级**架构细节：残差初始化 1/√(2L) 缩放（参考 GPT-2 论文）、QK-Norm（Llama-3、Gemma2 同款）、RoPE θ 调到 1e6、加上 YaRN 长上下文扩展
- **修复**已知 bug，下面会有专门一节讲

站在巨人肩上不是抄袭，是把别人没做好的地方做得更好。这个项目就是这么个定位。

## 这个项目到底有多大

```
ClearMind-Base   68.8M 参数   dense   对标 minimind-3 64M dense
ClearMind-Plus   486M  参数   dense   对标 minimind-3-moe 198M-A64M
```

技术栈说一下，黑话我尽量不堆：

- 模型架构：标准 GPT decoder，配 RoPE 位置编码、RMSNorm、SwiGLU 前馈层、GQA 多查询注意力（让推理时 KV Cache 更省显存）
- 训练阶段：Pretrain（学语言）→ SFT（学对话）→ DPO（学偏好），都是标配
- 发布格式：训练态用我自己的属性命名（`w_q` / `w_k`），发布前跑 `convert_to_qwen3.py` 转换成 `Qwen3ForCausalLM` 格式，这样 `transformers`、`vLLM`、`Ollama` 都能直接吃

这些黑话感兴趣可以去 [GitHub 仓库](https://github.com/Perlou/clear-mind) 翻代码，我就不在这里展开了。

## 关于 Plus，我为什么暂时没训

这是这篇文章里最诚实的一段。

ClearMind-Plus 是 486M dense 模型，对标 minimind-3-moe（198M-A64M）。dense 路线单 token 算力 7.1×，理论上同尺寸效果应该明显压制 MoE 版本。

我特别想训。

而且实话讲，**Base 整个流程是在一张 24G 显存的 4090 上跑下来的**——一杯星巴克的钱。最初我也想顺着这条路把 Plus 也试试，但很快就发现不太行。

24G 显存装 486M 模型的训练态（参数 + 梯度 + 优化器状态 + activation）很紧。即使开 activation checkpointing 把 batch size 压到 prtetrain 1 SFT 2，单卡跑完 38000 steps 大概要 4-5 天连续训。在 AutoDL 按量付费模式下，这意味着 4-5 天不能断电、不能断网、余额不能耗尽——任何一个出问题，`/root/autodl-tmp/` 跟着没，所有 checkpoint 灰飞烟灭。

要真想稳，就得换 **A800 80G**。但 A800 这个量级，**租一台连续训一周，差不多要花近 ¥1000**。

钱不是不能花，是这个杠杆比让我犹豫， 这钱还不如拿去订阅一个月的ChatGPT Pro， Codex里用GPT5.5 + xHigh + Fast + Goal 原地起飞。

一个 486M 模型相对于已经训好的 68M Base，对个人项目的边际收益没那么大。Base 已经能跑通整个发布链路、能在 Demo 上对话、能验证我的工程基础和架构——Plus 上去之后效果可能从「能聊」变成「聊得更顺」，但本质上没有突破。

近一千块钱花在一个"聊得更顺"上，对我来说目前性价比不够。所以我把这条路先冻起来，等之后能凑齐"完整一周不间断 + 一千预算 + 把 Plus 训完之后能干一件 Base 干不了的事"这三件，再回来训它。

敢说 Plus 没训，也是项目自我管理的一部分。

不过这条路我没堵死。`space/app.py` 里已经写好了运行时切换：Plus 占位先指向 Small，等真训完，去 ModelScope Studio 设置面板加一行环境变量 `CLEARMIND_REPO_PLUS=Perlou/ClearMind-Plus`，Studio 自动重启，新模型就上线了——零代码改动。

代码已经写好了。等的是钱和决心。

## 一些感受

自己训过一次大模型之后，再看任何 paper 里的 warmup steps、cosine schedule、KV Cache 命中率，心里有底了。这种「心里有底」的感觉，是看再多博客和论文都给不了你的。

但更让我有底的，其实不是模型本身，是那 147 个 pytest 测试。

CPU 上 3 秒跑完，覆盖 attention mask、loss mask、RMSNorm 的 bf16/fp16 contract、DPO max_steps、AdamW 参数分组……每修一个 bug 都立刻补一个回归 case。AutoDL 一小时几块钱，没人愿意烧空跑发现训练阶段才暴露的低级问题。

**基础设施才是真护城河**。这话不是我先说的，但我现在彻底信了。


## 最后

把项目链接再放一遍：

- 🤗 **HuggingFace**：[Perlous/ClearMind-Base](https://huggingface.co/Perlous/ClearMind-Base)
- 🧠 **ModelScope 模型**：[Perlou/ClearMind-Base](https://www.modelscope.cn/models/Perlou/ClearMind-Base)
- 💬 **在线 Demo**（可以直接对话）：[ClearMind 在线演示](https://www.modelscope.cn/studios/Perlou/ClearMind-Demo)
- 💻 **GitHub 源码**：[github.com/Perlou/clear-mind](https://github.com/Perlou/clear-mind)

Demo 用的是免费 CPU 配额，第一次访问会冷启动需要 1-2 分钟，请耐心。Plus 训完会回来更新这篇文章。

---

*炼丹这件事，最大的收获不是模型，是把"原来这一行字底下藏着这么多坑"看清楚的那一刻。*
