---
title: "Rust 快速入门"
date: 2025-11-19T12:07:13+08:00
description: "一篇关于 Rust 语言的快速入门指南"
categories:
  - "Rust"
tags:
  - "Rust"
  - "编程语言"
image:
---

# Rust 快速入门

## 简介

Rust 是一门赋予每个人构建可靠且高效软件能力的语言。

## 安装

你可以通过 rustup 来安装 Rust：

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

## Hello World

创建一个名为 `main.rs` 的文件，并写入以下内容：

```rust
fn main() {
    println!("Hello, world!");
}
```

运行它：

```bash
rustc main.rs
./main
```
