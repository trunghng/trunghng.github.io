---
layout: post
title:  "Neural Network from scratch"
date:   2022-08-13 13:00:00 +0700
categories: artificial-intelligent machine-learning deep-learning
tags: artificial-intelligent machine-learning deep-learning neural-network
description: Build a neural network from scratch using Numpy only
comments: true
---
> 
<!-- excerpt-end -->
- [Feedforward neural networks](#fnn)
	- [The XOR function](#xor)
- [References](#references)
- [Footnotes](#footnotes)

## Feedforward neural networks
{: #fnn}

### The XOR function
{: #xor}
The **XOR function** (or **exclusive or function**), denoted as $\oplus:\\{0,1\\}\times\\{0,1\\}\to\\{0,1\\}$, is defined as:
\begin{align}
\oplus(0,0)&=0, \\\\ \oplus(0,1)&=1, \\\\ \oplus(1,0)&=1, \\\\ \oplus(1,1)&=0,
\end{align}
or by words, $f(x_1,x_2)=1$ only if exactly one of the two binary inputs having the value of $1$, otherwise it returns the value of $0$.

Suppose given a set of four points $\mathbb{X}=\left\\{(0,0),(0,1),(1,0),(1,1)\right\\}$ and their projected value of them on $\oplus$ space, $\hat{f}(\mathbf{x}),\mathbf{x}\in\mathbb{X}$, we will learn an approximator of $\oplus$, denoted as $f$, by a feedforward network.

Consider this as a regression problem, we will be using the MSE as our loss function. Or in particular,
\begin{equation}
J(\mathbf{w},b)=\frac{1}{4}\sum_{\mathbf{x}\in\mathbb{X}}\left(\hat{f}(\mathbf{x})-f(\mathbf{x};\mathbf{w},b)\right)^2\tag{1}\label{1}
\end{equation}
Let us assume that we can learn a linear model $f$, that means
\begin{equation}
f(\mathbf{x};\mathbf{w},b)=\mathbf{x}^\intercal\mathbf{w}+b,
\end{equation}
which lets equation \eqref{1} be written as
\begin{equation}
J(\mathbf{w},b)=\frac{1}{4}\sum_{\mathbf{x}\in\mathbb{X}}\left(\hat{f}(\mathbf{x})-\left(\mathbf{x}^\intercal\mathbf{w}+b\right)\right)^2
\end{equation}
Taking the derivatives of $J$ w.r.t $\mathbf{w}$ and $b$, we have
\begin{align}
\nabla_\mathbf{w}J(\mathbf{w},b)&\propto\sum_{\mathbf{x}\in\mathbb{X}}\left(\hat{f}(x)-\left(\mathbf{x}^\intercal\mathbf{w}+b\right)\right)\mathbf{x}, \\\\ \nabla_b J(\mathbf{w},b)&\propto\sum_{\mathbf{x}\in\mathbb{X}}\left(\hat{f}(x)-\left(\mathbf{x}^\intercal\mathbf{w}+b\right)\right)
\end{align}
Letting these gradients be zero gives us $\mathbf{w}=\mathbf{0}$ and $b=\frac{1}{2}$. With this solution, our model simply returns $\frac{1}{2}$ for any given input. This means that we can not find a linear function that describes exactly how the XOR works.

One possible solution to this problem is that instead of taking $\mathbb{X}$ as the domain of our linear model, we will choose a space $\mathbb{X}'$ on which we can successfully apply a linear model. On other words, we select a vector-valued function $f^{(1)}:\mathbb{X}\to\mathbb{X}'$ such that
\begin{equation}
\oplus(\mathbf{x})=f^{(2)}(f^{(1}(\mathbf{x})),
\end{equation}
where $f^{(2)}$ is a linear function.

Clearly we can not pick $f^{(1)}$ as a linear function, or specifically a linear transform because the composition $f^{(2)}\circ f^{(1)}$ of two linear functions $f^{(2)}$ and $f^{(1)}$ is still a linear function. In particular, assume that $f^{(1)}(\mathbf{x})=\mathbf{W}^\intercal\mathbf{x}+\mathbf{c}$, then
\begin{align}
f^{(2)}(f^{(1)}(\mathbf{x}))&=\left(\mathbf{W}^\intercal\mathbf{x}+\mathbf{c}\right)^\intercal\mathbf{w}+b \\\\ &=\left(\mathbf{w}^\intercal\mathbf{W}^\intercal\right)\mathbf{x}+\left(\mathbf{w}^\intercal\mathbf{c}+b\right)
\end{align}



## References
{: #references}
[1] joelgrus [JoelNet](https://github.com/joelgrus/joelnet).

[2] Ian Goodfellow & Yoshua Bengio & Aaron Courville. [Deep Learning](https://www.deeplearningbook.org). MIT Press (2016).

[3] Adrew Ng. [Deep Learning Specialization](https://coursera.com). Coursera.

[4] Pytorch Documentation [Pytorch Docs](https://pytorch.org/docs/stable/index.html).

## Footnotes
{: #footnotes}