---
title: "PILCO"
date: 2024-03-08T10:03:08+07:00
tags: [reinforcement-learning, model-based, gaussian-process, my-rl]
math: true
eqn-number: true
---
> A model-based RL method that learns a Bayesian nonparametric model (Gaussian process) and reduces model bias to improve sample efficiency.
<!--more-->

Consider dynamical systems
\begin{equation}
\mathbf{x}\_{t+1}=f(\mathbf{x}\_t,\mathbf{u}\_t)+\mathbf{w},
\end{equation}
with continuous-valued states $\mathbf{x}\in\mathbb{R}^D$, controls $\mathbf{u}\in\mathbb{R}^F$, i.i.d Gaussian system noise $\mathbf{w}\sim\mathcal{N}(\mathbf{0},\mathbf{\Sigma}\_w)$, and unknown transition dynamics $f$.

Our goal is to find a deterministic policy (or controller), $\pi:\mathbf{x}\mapsto\pi(\mathbf{x};\boldsymbol{\theta})=\mathbf{u}$, parameterized by $\boldsymbol{\theta}$, that minimizes the expected return of following $\pi$ for $T$ steps
\begin{equation}
J^\pi(\boldsymbol{\theta})=\sum_{t=0}^{T}\mathbb{E}\_{\mathbf{x}\_t}\big[c(\mathbf{x}\_t)\big],\hspace{1cm}\mathbf{x}\_0\sim\mathcal{N}(\boldsymbol{\mu}\_0,\mathbf{\Sigma}\_0),
\end{equation}
where $c(\mathbf{x}\_t)$ is the cost (negative reward) of being in state $\mathbf{x}$ at time $t$.

### Model-based policy search
The **PILCO** (probabilistic inference for learning control) method consists of three key components: the dynamics model, analytic approximate policy evaluation and gradient-based policy improvement.

#### Model Learning
The probabilistic dynamics model is implemented as a Gaussian process (GP) with
<ul class='number-list'>
	<li>
		Training inputs: $(\mathbf{x}_t,\mathbf{u}_t)\in\mathbb{R}^{D+F}$.
	</li>
	<li>
		Training targets: $\Delta_t=\mathbf{x}_{t+1}-\mathbf{x}_t\in\mathbb{R}^D$.
	</li>
</ul>

A GP is fully defined by a mean function $m(\cdot)$ and a positive semidefinite covariance function $k(\cdot,\cdot)$. In the paper, authors took the mean function to be zero and the squared exponential (SE) covariance function, as
\begin{equation}
k(\tilde{\mathbf{x}}\_p,\tilde{\mathbf{x}}\_q)=\sigma_f^2\exp\left(-\frac{1}{2}(\tilde{\mathbf{x}\_p}-\tilde{\mathbf{x}}\_q)^\text{T}\mathbf{\Lambda}^{-1}(\tilde{\mathbf{x}}\_p-\tilde{\mathbf{x}}\_q)\right)+\delta_{pq}\sigma_w^2,
\end{equation}
where $\tilde{\mathbf{x}}=\left[\begin{matrix}\mathbf{x} \\\\ \mathbf{u}\end{matrix}\right]\in\mathbb{R}^{D+F}$ and $\mathbf{\Lambda}=\text{diag}\left(\left[\ell_1^2,\ldots,\ell_{D+F}^2\right]\right)$ with characteristic length-scales $\ell_i$, and $\sigma_f^2$ is the variance of the latent transition function $f$. Given $n$ training inputs-targets
\begin{equation}
\tilde{\mathbf{X}}=\left[\begin{matrix}\vert&&\vert \\\\ \tilde{\mathbf{x}}\_1&\ldots&\tilde{\mathbf{x}}\_n \\\\ \vert&&\vert\end{matrix}\right]\in\mathbb{R}^{n\times D+F},\hspace{1cm}\mathbf{y}=\left[\begin{matrix}-\hspace{0.1cm}\Delta_1\hspace{0.1cm}- \\\\ \vdots \\\\ -\hspace{0.1cm}\Delta_n\hspace{0.1cm}-\end{matrix}\right]\in\mathbb{R}^{n\times D},
\end{equation}
the posterior GP hyperparameters ($\ell_i,\sigma_f^2$ and $\sigma_w^2$) are learned by evidence maximization.

The posterior GP is a one-step prediction model, and the predicted successor state $\mathbf{x}_{t+1}$ is Gaussian distributed
\begin{equation}
p(\mathbf{x}\_{t+1}\vert\mathbf{x}\_t,\mathbf{u}\_t)=\mathcal{N}(\mathbf{x}\_{t+1}\vert\boldsymbol{\mu}\_{t+1},\mathbf{\Sigma}\_{t+1})
\end{equation}
with
\begin{equation}
\boldsymbol{\mu}\_{t+1}=\mathbf{x}\_t+\mathbb{E}\_f[\mathbf{\Delta}\_t],\hspace{1cm}\mathbf{\Sigma}\_{t+1}=\text{var}\_f[\mathbf{\Delta}\_t]
\end{equation}
where
\begin{align}
\mathbb{E}\_f[\mathbf{\Delta}\_t]&=m_f(\tilde{\mathbf{x}}\_1)=\mathbf{k}\_\*^\text{T}(\mathbf{K}+\sigma_w^2\mathbf{I})^{-1}\mathbf{y}=\mathbf{k}\_\*^\text{T}\boldsymbol{\beta}, \\\\ \text{var}\_f[\mathbf{\Delta}\_t]&=k\_{\*\*}-\mathbf{k}\_\*^\text{T}(\mathbf{K}+\sigma_w^2\mathbf{I})^{-1}\mathbf{k}\_\*,
\end{align}
with $\mathbf{k}\_\*=k(\tilde{\mathbf{X}},\tilde{\mathbf{x}}_t)$ and $\boldsymbol{\beta}=(\mathbf{K}+\sigma_w^2\mathbf{I})^{-1}\mathbf{y}$ where $\mathbf{K}$ is the kernel matrix with entries $K\_{ij}=k(\tilde{\mathbf{x}}_i,\tilde{\mathbf{x}}_j)$.

#### Policy Evaluation

#### Policy Improvement

### References
[1] Marc Peter Deisenroth & Carl Edward Rasmussen. [PILCO: a model-based and data-efficient approach to policy search](https://dl.acm.org/doi/10.5555/3104482.3104541). ICML, 2011.

[2] Marc Peter Deisenroth, Dieter Fox, Carl Edward Rasmussen. [Gaussian Processes for Data-Efficient Learning in Robotics and Control](https://dx.doi.org/10.1109/TPAMI.2013.218). IEEE Transactions on Pattern Analysis and Machine Intelligence, 2015.

[3] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

### Footnotes
