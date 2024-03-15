---
title: "PILCO"
date: 2024-03-08T10:03:08+07:00
tags: [reinforcement-learning, model-based, my-rl]
math: true
eqn-number: true
draft: true
---
> A model-based RL method that reduces model bias to improve sample efficiency.
<!--more-->

Consider dynamic systems
\begin{equation}
\mathbf{x}\_t=f(\mathbf{x}\_{t-1},\mathbf{u}\_{t-1}),
\end{equation}
where continuous-valued states $\mathbf{x}\in\mathbb{R}^D$ and controls $\mathbf{u}\in\mathbb{R}^F$ and unknown transition dynamics $f$. Our goal is to find a deterministic policy (or controller) $\pi:\mathbf{x}\mapsto\pi(\mathbf{x})=\mathbf{u}$, parameterized by $\theta$, that minimizes the expected return of following $\pi$ for $T$ steps
\begin{equation}
J^\pi(\theta)=\sum_{t=0}^{T}\mathbb{E}\_{\mathbf{x}\_t}\big[c(\mathbf{x}\_t)\big],\hspace{1cm}\mathbf{x}\_0\sim\mathcal{N}(\mu_0,\Sigma_0),
\end{equation}
where $c(\mathbf{x}\_t)$ is the cost (negative reward) of being in state $\mathbf{x}$ at time $t$.

### Gaussian Processes
Given a dataset $(\mathbf{X},\mathbf{y})$ where
\begin{equation}
\mathbf{X}=\left[\begin{matrix}\vert&&\vert \\\\ \mathbf{x}\_1 & \ldots & \mathbf{x}\_n \\\\ \vert&&\vert\end{matrix}\right],\hspace{1cm}\mathbf{y}=\left[\begin{matrix}y_1 \\\\ \vdots \\\\ y_n \end{matrix}\right],
\end{equation}
are respectively the matrix of training inputs $\mathbf{x}\_i$ and the vector of training targets $y_i$. We want to infer a model of the function $h$ that generated the data
\begin{equation}
y_i=h(\mathbf{x}\_i)+\epsilon,\hspace{1cm}\epsilon\sim\mathcal{N}(0,\sigma_\epsilon^2)
\end{equation}
In a Bayesian framework, the inference of $h$ is described by the posterior probability
\begin{equation}
p(h\vert\mathbf{X},\mathbf{y})=\frac{p(\mathbf{y}\vert h,\mathbf{X})p(h)}{p(\mathbf{y}\vert\mathbf{X})},
\end{equation}
where $p(\mathbf{y}\vert h,\mathbf{X})$ is referred as the likelihood and $p(h)$ is known as the prior distribution on functions assumed the model

When modeling with Gaussian processes (GPs), we replace a GP prior $p(h)$ directly in the space of functions without the necessity to consider an explicit parameterization of the function $h$

### Mode-based indirect policy search
The PILCO method consists of three key components: the dynamics model, analytic approximate policy evaluation and gradient-based policy improvement.

#### Dynamics Model Learning
The probabilistic dynamics model is implemented as a Gaussian process (GP), where we use the tuple $(\mathbf{x}\_{t-1},\mathbf{u}\_{t-1})\in\mathbb{R}^{D+F}$ as training inputs and differences $\delta_t=\mathbf{x}\_t-\mathbf{x}\_{t-1}+\epsilon\in]\mathbb{R}^D$


### Policy Evaluation

### References
[1] Marc Peter Deisenroth & Carl Edward Rasmussen. [PILCO: a model-based and data-efficient approach to policy search](https://dl.acm.org/doi/10.5555/3104482.3104541). ICML, 2011.

[2] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

### Footnotes
