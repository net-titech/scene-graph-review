---
layout: post
title: "LinkNet: Relational Embedding for Scene Graph - A Summary"
excerpt: A summary of LinkNet paper.
categories:
 - Deep Neural Networks
 - Graph Neural Networks
 - Machine Learning
comments: true
author: Kaushalya
date: "2019-07-12"
published: true
pinned: true
category: blog
---

# LinkNet: Relational Embedding for Scene Graph

- Conference: [NeurIPS 2018](https://papers.nips.cc/paper/7337-linknet-relational-embedding-for-scene-graph)
- Code: [Unofficial](https://github.com/jiayan97/linknet-pytorch)
- Authors:
    - Sanghyun Woo, Dahun Kim, Donghyeon Cho, and In So Kweon - _KAIST, South Korea_

This paper proposes LinkNet, a new model for scene graph generation. LinkNet model consists of three modules.
1. Object relational embedding
2. Global context encoding (GCE)
3. Geometric layout encoding

__Input__ : Object proposals and features from a region proposal network (RPN).
Each object proposal is represented as a vector \\( o_i = (f_i^{RoI}, K_0l_i, c) \\). \\( K_0 \\) is a parameter matrix which maps the distribution of predicted labels \\( l_i \\) of each of the object proposal \\( i=1,..., N \\).

### 1. Object relational embedding
Object features are learnt using a graph-based approach.

$$ \mathbf{R}_{1}=\operatorname{softmax}\left(\left(\mathbf{O}_{0} \mathbf{W}_{1}\right)\left(\mathbf{O}_{0} \mathbf{U}_{1}\right)^{\mathbf{T}}\right) \in \mathbb{R}^{\mathbf{N} \times \mathbf{N}} - \text{Relational embedding} $$

$$ \mathbf{O}_{1}=\mathbf{O}_{0} \oplus f c_{0}\left(\left(\mathbf{R}_{\mathbf{1}}\left(\mathbf{O}_{0} \mathbf{H}_{1}\right)\right)\right) \in \mathbb{R}^{\mathbf{N} \times 4808} $$

$$ \mathbf{O}_{2}=f c_{1}\left(\mathbf{O}_{1}\right) \in \mathbb{R}^{\mathbf{N} \times 256} - \text{Relation-aware embedding} $$

$\oplus$ denotes elementwise summation. \\( O_1 \\) can be considered as applying a graph convolutional (GCN) layer with a residual connection. The resultant features \\( O_2 \\) is once again fed into a similar set of layers to get \\( O_4 \in \mathbb{R}^{N \times C_{obj}} \\).

### 2. Global context encoding (GCE)
\\( c \in \mathbb{R}^{512} - \text{Average pooling of RPN image featurs} \\)

Feature vector $c$ is concatenated with other RPN featurs to get \\\( o_i \\).

### 3. Geometric layout encoding
This encodes relative location and scale information of an object.

$$
\mathbf{b}_{\mathbf{o} | \mathbf{s}}=\left(\frac{\mathbf{x}_{\mathbf{o}}-\mathbf{x}_{\mathbf{s}}}{\mathbf{w}_{\mathbf{s}}}, \frac{\mathbf{y}_{\mathbf{o}}-\mathbf{y}_{\mathbf{s}}}{\mathbf{h}_{\mathbf{s}}}, \log \left(\frac{\mathbf{w}_{\mathbf{o}}}{\mathbf{w}_{\mathbf{s}}}\right), \log \left(\frac{\mathbf{h}_{\mathbf{o}}}{\mathbf{h}_{\mathbf{s}}}\right)\right) $$

\\( x_o, y_o, h_o, w_o \\): coordinates, height, and width of the object proposal of object \\( o \\)
$o$ and $s$ stand for object and subject respectively. These features are used for learning _edge-relational embeddings_.

### Loss function
$$
\mathcal{L}_{\text {final}}=\mathcal{L}_{\text {obj}_{-} c l s}+\lambda_{1} \mathcal{L}_{\text {rel}_{-} c l s}+\lambda_{2} \mathcal{L}_{\text {gce}} $$
By default $\lambda_1$ and $\lambda_2$ are set to 1.
