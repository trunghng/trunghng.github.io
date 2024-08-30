---
title: "Gaussian Processes"
date: 2024-06-28T16:14:16+07:00
tags: [machine-learning, probability-statistics, gaussian-process, normal-distribution]
draft: true
math: true
eqn-number: true
hideSummary: true
---

## Gaussian Processes
A **Gaussian process (GP)** is a collection of r.v.s, any finite number of which have a joint Gaussian distribution. Each GP $f(\mathbf{x})$ is fully defined by a mean function $m(\mathbf{x})$ and a positive definite covariance function $k(\mathbf{x},\mathbf{x}')$.
\begin{align}
m(\mathbf{x})&=\mathbb{E}\big[f(\mathbf{x})\big] \\\\ k(\mathbf{x},\mathbf{x}')&=\mathbb{E}\big[(f(\mathbf{x})-m(\mathbf{x}))(f(\mathbf{x}')-m(\mathbf{x}'))\big]
\end{align}
And we denote
\begin{equation}
f(\mathbf{x})\sim\mathcal{GP}(m(\mathbf{x}),k(\mathbf{x},\mathbf{x}'))
\end{equation}

## References
[1] Carl Edward Rasmussen & Christopher K. I. Williams. [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml). The MIT Press, 2006.

## Footnotes
