# S4 & Mamba In-Depth Guide


## 📚 Learning Path Overview

This guide provides a structured learning path to understand S4 (Structured State Space Sequence Model) and Mamba from fundamentals to implementation. Follow these steps sequentially for the best learning experience.

### 1️⃣ Introduction to Sequence Modeling
**Goal**: Understand the basics and motivation behind state space models

- 📖 Start with [A Visual Guide to Mamba and State Space Models](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-mamba-and-state)
  - Focus on [Part 1: The Problem with Transformers](https://newsletter.maartengrootendorst.com/i/141228095/part-the-problem-with-transformers)
  - This explains why we need alternatives to Transformers and introduces S4

### 2️⃣ Deep Dive into State Space Models (SSMs)
**Goal**: Master the fundamental concepts of State Space Models

1. 🎥 Watch [State Space Models (SSMs) and Mamba](https://www.youtube.com/watch?v=g1AqUhP00Do) by Serrano.Academy
   - Provides excellent visualizations and examples of SSMs

2. 📘 Continue with [Part 2: The State Space Model](https://newsletter.maartengrootendorst.com/i/141228095/part-the-state-space-model-ssm)
   - Builds on the theoretical foundation

3. 💻 Study [The Annotated S4](https://srush.github.io/annotated-s4/#part-1b-addressing-long-range-dependencies-with-hippo)
   > Note: Since the original repo is outdated, use the [s4_in_depth.ipynb](./s4_in_depth.ipynb) notebook in this repository
   >
   > 对于中文读者，我已经将原作者的The Annotated S4翻译成了中文，放在了[cn 文件夹](.cn/The-Annotated-S4-CN-Part1.md)中，方便中文读者阅读。

### 3️⃣ Mamba Architecture
**Goal**: Understand how Mamba builds upon and improves SSMs

- 📖 Read [Part 3: Visual Guide to Mamba](https://newsletter.maartengrootendorst.com/i/141228095/part-mamba-a-selective-ssm)
  - Explains how Mamba extends and improves upon S4

### 4️⃣ Hands-On Implementation
**Goal**: Get practical experience with Mamba

- 💻 Study [The Annotated Mamba](https://srush.github.io/annotated-mamba/hard.html)
  - Detailed implementation walkthrough
  - Includes code examples and explanations

### 5️⃣ Applications & Real-World Usage
**Goal**: Explore practical applications and stay updated

1. 📚 Explore [Awesome-Mamba-Collection](https://github.com/XiudingCai/Awesome-Mamba-Collection?tab=readme-ov-file#head18)
   - Comprehensive overview of Mamba applications
   - Covers NLP, time series analysis, and more

2. 🔍 Additional Resources:
   - [Official Mamba Repository](https://github.com/state-spaces/mamba)
   - 📄 [Research Paper: Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)


> 💡 **Tip**: Follow this path sequentially for the best learning experience. Each section builds upon the knowledge from previous sections.