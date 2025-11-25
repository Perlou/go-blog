+++
date = '2025-11-25T03:00:00+08:00'
draft = false
title = '大模型入门：原理、架构与实战思考'
image = '/images/covers/llm-learning.jpg'
categories = ['AI开发', '大模型']
tags = ['LLM', 'RAG', 'Agents', '大模型']
+++

在过去的一段时间里，我通过 [ai-learning-ts](https://github.com/Perlou/ai-learning-ts) 项目，从零开始构建了一套大模型学习路径。写代码的过程很有趣，但更让我着迷的是代码背后的原理。

当我们谈论"开发 AI 应用"时，我们到底在开发什么？是调用 API 的胶水代码？还是在构建某种新的计算范式？

这篇文章不谈具体的 API 调用，而是试图剥开大模型的表象，探讨其底层的运行机制、架构设计以及工程上的权衡。

项目源码:

- [ai-learning-ts](https://github.com/Perlou/ai-learning-ts)
- [web-chat](https://github.com/Perlou/web-chat)

WebChat: [演示地址](https://webchat-rho-nine.vercel.app/)

## 第一部分：概率的博弈

要理解 LLM，首先要打破一个幻想：**AI 并不像人类那样"思考"**。

### 统计学的胜利

本质上，LLM 是一个巨大的**概率预测机**。当你输入"今天天气真"时，模型并不是在理解天气的概念，而是在高维空间中计算下一个字出现的概率分布：

- "好"：70%
- "坏"：20%
- "不错"：9%

它所做的一切，都是基于海量数据训练出来的统计规律。这种机制解释了为什么 LLM 会产生**幻觉 (Hallucination)**。

### 幻觉：Bug 还是 Feature？

在传统软件工程中，我们追求确定性：输入 A，必须输出 B。但在 LLM 的世界里，确定性被打破了。

幻觉的本质是模型在概率空间中的一次"冒险"。当它面对未知的领域时，它倾向于选择概率上最合理的词，而不是事实。

这在创意写作中是 Feature（创造力），但在事实问答中就是 Bug（胡说八道）。

作为开发者，我们无法消除幻觉，只能通过**Temperature（温度）**参数来控制这种随机性，或者通过 RAG 等架构手段来规避它。

```typescript
// 控制随机性的关键参数
const model = genAI.getGenerativeModel({
  model: "gemini-pro",
  generationConfig: {
    temperature: 0.9, // 高创造性
    // temperature: 0.1, // 高确定性
  },
});
```

## 第二部分：语义的几何

如果说概率是 LLM 的灵魂，那么 **Embedding (嵌入)** 就是它的骨架。

### 词语的坐标

计算机无法理解"苹果"这个词的含义，但它可以理解数字。Embedding 技术将文本映射到一个高维向量空间（通常是 768 维或更高）。

在这个空间里，语义相似的词，距离会很近。

- "猫" 和 "狗" 的距离，远小于 "猫" 和 "汽车" 的距离。
- "国王" - "男人" + "女人" ≈ "女王"

这种数学上的几何关系，让计算机第一次真正拥有了处理"语义"的能力。

```typescript
// 将文本转换为向量
const embedding = await model.embedContent("苹果");
// 输出: [0.012, -0.045, 0.891, ...] (768维向量)
```

### 为什么关键词搜索会死？

传统的搜索引擎基于关键词匹配（Keyword Matching）。你搜"好吃的红球"，它可能搜不到"苹果"，因为字面上没有重叠。

而基于 Embedding 的**语义搜索 (Semantic Search)**，计算的是向量之间的距离。它知道"红球"和"苹果"在语义空间中靠得很近。

这就是为什么 RAG（检索增强生成）能起作用的核心原因：**它不再是匹配文字，而是在匹配意图**。

## 第三部分：RAG 的本质

**RAG (Retrieval-Augmented Generation)** 是目前解决 LLM 知识时效性和私有数据问题的最佳实践。

### 突破上下文的枷锁

LLM 的上下文窗口（Context Window）虽然在不断变大（从 4k 到 1M+），但依然是有限且昂贵的。你不可能把公司所有的文档都塞进 Prompt 里。

RAG 的本质是一种**分治策略**：

1.  **切片 (Chunking)**：将海量知识切分成小块。
2.  **索引 (Indexing)**：将这些小块向量化，存入向量数据库。
3.  **检索 (Retrieval)**：根据用户问题，只捞出最相关的几块。
4.  **生成 (Generation)**：让 LLM 基于这几块信息回答。

### 架构的挑战

RAG 听起来简单，但工程细节极多：

- **切片策略**：切多大？按段落切还是按语义切？切太小丢失上下文，切太大干扰检索。
- **检索质量**：单纯的向量检索（Dense Retrieval）在处理精确匹配（如人名、ID）时往往不如关键词检索。现在的趋势是 **Hybrid Search**（混合检索），结合向量和关键词的优势。
- **重排序 (Reranking)**：检索回来的结果可能包含噪声，需要一个精细的模型进行二次排序。

## 第四部分：从对话到行动

如果 LLM 只能聊天，那它只是一个高级玩具。**Function Calling (函数调用)** 是让 AI 落地到生产环境的关键一步。

### 确定性与概率的桥梁

Function Calling 解决了最大的痛点：**如何让概率性的 AI 与确定性的程序交互？**

AI 输出的是自然语言，而 API 需要的是结构化的 JSON。Function Calling 让模型经过微调，能够稳定地输出符合 Schema 定义的 JSON 对象。

这不仅仅是格式转换，更是**意图识别**。AI 需要理解：

1.  用户想要做什么？
2.  我有哪些工具？
3.  这个任务需要哪个工具？
4.  参数从哪里提取？

```typescript
// 定义工具 Schema，告诉 AI 如何调用
const tools = [
  {
    functionDeclarations: [
      {
        name: "get_weather",
        description: "获取指定城市的天气",
        parameters: {
          type: SchemaType.OBJECT,
          properties: {
            city: { type: SchemaType.STRING, description: "城市名称" },
          },
          required: ["city"],
        },
      },
    ],
  },
];
```

### Agent：感知的闭环

基于 Function Calling，我们构建出了 **Agent (智能体)**。

Agent 不再是被动的问答机，它拥有了**感知 -> 决策 -> 行动**的闭环能力：

1.  **感知**：接收用户指令和环境状态。
2.  **规划**：拆解任务，决定调用哪些工具。
3.  **行动**：执行 API 调用。
4.  **反馈**：根据执行结果，决定下一步操作。

这就是从 Copilot（副驾驶）到 Autopilot（自动驾驶）的跨越。

## 第五部分：本地化的权衡

随着 Llama 3, Mistral, Qwen 等开源模型的崛起，**本地部署 (Local LLM)** 成为了一种严肃的选项。

```typescript
// 使用 Ollama 在本地运行模型，代码体验与云端几乎一致
import ollama from "ollama";

const response = await ollama.chat({
  model: "qwen2.5:7b",
  messages: [{ role: "user", content: "你好" }],
});
```

### 量化 (Quantization) 的魔法

一个 70B 参数的模型，FP16 精度下需要 140GB 显存，普通人根本跑不起来。

**量化**技术通过降低数值精度（从 16-bit 降到 4-bit 甚至更低），将模型体积压缩到原来的 1/4，而性能损失却微乎其微。

这使得我们可以在 MacBook 甚至手机上运行强大的大模型。

### 隐私与性能的博弈

本地部署最大的优势是**隐私**和**成本**。数据不出域，Token 不要钱。

但代价是**推理能力**。目前的 7B/14B 模型在复杂逻辑推理和指令遵循上，依然无法与 GPT-4 或 Claude 3.5 这种万亿参数的巨兽抗衡。

因此，未来的架构一定是**混合 (Hybrid)** 的：

- **边缘端**：本地小模型处理隐私数据、实时响应、简单任务。
- **云端**：大模型处理复杂推理、长上下文分析。

## 第六部分：体验的最后一公里

即便模型再强大，如果用户体验很差，应用依然会失败。在 LLM 应用中，**延迟 (Latency)** 是最大的敌人。

### 流式响应 (Streaming) 的心理学

LLM 生成内容是逐字进行的，生成一篇长文可能需要 10 秒甚至更久。如果让用户对着空白屏幕等 10 秒，他会觉得系统卡死了。

**流式响应**（Streaming）解决了这个问题。它利用 HTTP 分块传输（Chunked Transfer Encoding），每生成一个字就推送到前端。

虽然总耗时没变，但**首字延迟 (Time to First Token, TTFT)** 从 10 秒降到了 0.5 秒。这种即时反馈极大地降低了用户的心理等待时间。

### Web Chat 实战

在 [web-chat](https://github.com/Perlou/web-chat) 项目中，我使用 Next.js 和 Vercel AI SDK 构建了一个现代化的聊天界面。

具体项目演示：[Demo](https://webchat-rho-nine.vercel.app/)

```typescript
// 前端处理流式响应的核心逻辑
const { messages, input, handleInputChange, handleSubmit } = useChat({
  api: "/api/chat",
  streamProtocol: "text", // 逐字接收
});

// 这种简单的 Hook 背后，封装了复杂的流处理逻辑
```

从 CLI 到 Web，不仅仅是界面的变化，更是对**异步、状态管理、流式处理**等前端工程能力的考验。

## 结语

AI 工程正在经历寒武纪大爆发。新的模型、新的架构、新的工具层出不穷。

但技术的热点总会冷却，唯有原理长存。

理解了概率预测，你就懂了 Prompt 的边界；理解了 Embedding，你就懂了 RAG 的本质；理解了 Function Calling，你就懂了 Agent 的未来。

希望这篇文章能带给你一些代码之外的思考。

---

_本文提到的所有概念，在 [ai-learning-ts](https://github.com/Perlou/ai-learning-ts) 项目中都有对应的 TypeScript 代码实现。_
