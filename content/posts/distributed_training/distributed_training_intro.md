---
title: "Exploring Distributed Deep Learning with LizardDist"
date: 2025-06-30
draft: false
ShowToc: true
math: true
tags: ["deep learning", "distributed training", "Distributed Data Parallel", "Data Parallel", "Tensor Parallelism"]
---

In this post, I’m laying the groundwork for my experiments with distributed training strategies using PyTorch and mpi4py. I want to explore approaches like distributed data parallelism, tensor parallelism, hybrid strategies, and more, digging into how communication, computation overlap, and scaling trade-offs work under the hood.

[LizarDist](https://github.com/adelbennaceur/lizardist) is the playground I’m building to test and learn these concepts. My goal is not just to build something that works, but to truly internalize the theory and practical challenges of distributed deep learning.

I’ll share what I learn as I build support for these strategies. This is just the start I’ll be publishing a series of blog posts diving deeper into each concept and design choice as I continue developing the framework from scratch.
