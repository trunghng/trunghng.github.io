---
title: "Variational Inference"
date: 2023-12-12T08:51:06+07:00
draft: true
tags: [machine-learning, variational-inference]
math: true
eqn-number: true
---
> The goal of Variational Inference (VI) is to approximate a conditional densities of latent variables given observed variables. The key idea is to solve this problem with optimization.
<!--more-->

## Variational Inference

### The variational objective
Consider a model with (observed) variables $\mathbf{x}$, latent variables $\mathbf{z}$, and parameterized by $\boldsymbol{\theta}$.

Say, we assume that the prior is $p_\boldsymbol{\theta}(\mathbf{z})$ and the likelihood is $p_\boldsymbol{\theta}(\mathbf{x}\vert\mathbf{z})$. Thus the unnormalized joint distribution is
\begin{equation}
p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})=p_\boldsymbol{\theta}(\mathbf{x}\vert\mathbf{z})p_\boldsymbol{\theta}(\mathbf{z}),
\end{equation}
and the posterior is
\begin{equation}
p_\boldsymbol{\theta}(\mathbf{z}\vert\mathbf{x})=\frac{p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})}{p_\boldsymbol{\theta}(\mathbf{x})}=\frac{p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})}{\int_z p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})dz}
\end{equation}
When the integration is intractable, it is then intractable to compute the normalized posterior, $p_\boldsymbol{\theta}(\mathbf{z}\vert\mathbf{x})$. We instead find an approximation to it, denoted $q(\mathbf{z})$, such that we minimize the following loss
\begin{equation}
q=\underset{q\in\mathcal{Q}}{\text{argmin}}\hspace{0.1cm}D_\text{KL}(q(\mathbf{z})\Vert p_\boldsymbol{\theta}(\mathbf{z}\vert\mathbf{x})),
\end{equation}
where $\mathcal{Q}$ is some tractable family of distribution (e.g. fully factorized distributions). This is known as the **variational method**. Moreover, if $\mathcal{Q}$ is a parametric family, parameterized by $\boldsymbol{\psi}$, which is referred as the **variational parameters**, we have that
\begin{align}
\boldsymbol{\psi}^\*&=\underset{\boldsymbol{\psi}}{\text{argmin}}\hspace{0.1cm}D_\text{KL}(q_\boldsymbol{\psi}(\mathbf{z})\Vert p_\boldsymbol{\theta}(\mathbf{z}\vert\mathbf{x})) \\\\ &=\underset{\boldsymbol{\psi}}{\text{argmin}}\hspace{0.1cm}\mathbb{E}\_{q_\boldsymbol{\psi}(z)}\left[\log q_\boldsymbol{\psi}(\mathbf{z})-\log\frac{p_\boldsymbol{\theta}(\mathbf{x}\vert\mathbf{z})p_\boldsymbol{\theta}(\mathbf{z})}{p_\boldsymbol{\theta}(\mathbf{x})}\right] \\\\ &=\underset{\boldsymbol{\psi}}{\text{argmin}}\hspace{0.1cm}\mathbb{E}\_{q_\boldsymbol{\psi}(\mathbf{z})}\big[\log q_\boldsymbol{\psi}(\mathbf{z})-\log p_\boldsymbol{\theta}(\mathbf{x}\vert\mathbf{z})-\log p_\boldsymbol{\theta}(\mathbf{z})\big]+\log p_\boldsymbol{\theta}(\mathbf{x}) \\\\ &=\underset{\boldsymbol{\psi}}{\text{argmin}}\hspace{0.1cm}\mathbb{E}\_{q_\boldsymbol{\psi}(\mathbf{z})}\big[\log q_\boldsymbol{\psi}(\mathbf{z})-\log p_\boldsymbol{\theta}(\mathbf{x}\vert\mathbf{z})-\log p_\boldsymbol{\theta}(\mathbf{z})\big],
\end{align}
which is equivalent to finding $\boldsymbol{\psi}$ that minimizes
\begin{equation}
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})=\mathbb{E}\_{q_\boldsymbol{\psi}(\mathbf{z})}\big[-\log p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})+\log q_\boldsymbol{\psi(\mathbf{z})}\big]\label{eq:tvo.1}
\end{equation}

#### Physical perspective: minimize the variational free energy
If we define $\mathcal{E}\_\boldsymbol{\theta}(\mathbf{z})=-\log p_\boldsymbol{\theta}(\mathbf{z},\mathbf{x})$ as the energy, we can rewrite the loss in \eqref{eq:tvo.1} as
\begin{equation}
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})=\mathbb{E}\_{q_\boldsymbol{\psi}(\mathbf{z})}[\mathcal{E}\_\boldsymbol{\theta}(\mathbf{z})]-H(q_\boldsymbol{\psi}),
\end{equation}
where $H(q_\boldsymbol{\psi})$ denotes the entropy. This is known as the **variational free energy** (VFE) in physics. This is an upper bound on the **free energy** (FE), $-\log p_\boldsymbol{\theta}(\mathbf{x})$, since
\begin{equation}
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})+\log p_\boldsymbol{\theta}(\mathbf{x})=D_\text{KL}(q_\boldsymbol{\psi}(\mathbf{z})\Vert p_\boldsymbol{\theta}(\mathbf{z}\vert\mathbf{x}))\geq 0,
\end{equation}
which implies that
\begin{equation}
\mathcal{L}(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})\geq-\log p_\boldsymbol{\theta}(\mathbf{x})
\end{equation}
In other words, variational inference is equivalent to minimizing the VFE.

#### Statistical perspective: maximize the evidence lower bound
The negative of the VFE is referred as the **evidence lower bound** or **ELBO** function
\begin{equation}
\text{ELBO}=L(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})\doteq\mathbb{E}\_{q_\boldsymbol{\psi}(\mathbf{z})}\big[\log p_\boldsymbol{\theta}(\mathbf{x},\mathbf{z})-\log q_\boldsymbol{\psi}(\mathbf{z})\big]
\end{equation}
The name comes up from the fact that $L(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})$ is a lower bound of $\log p_\boldsymbol{\theta}(\mathbf{x})$
\begin{equation}
L(\boldsymbol{\theta},\boldsymbol{\psi}\vert\mathbf{x})\leq\log p_\boldsymbol{\theta}(\mathbf{x}),
\end{equation}
where $\log p_\boldsymbol{\theta}(\mathbf{x})$ is called the evidence. Therefore, we have that variational inference is equivalent to maximize the ELBO w.r.t $\boldsymbol{\psi}$.

### Parameter estimation using variational EM
So far, we have assumed that the model parameters, $\boldsymbol{\theta}$, are known. However, we can try to estimate them by maximizing the log marginal likelihood of the dataset, $\mathcal{D}=\\{\mathbf{x}\_1,\ldots,\mathbf{x}\_N\\}$
\begin{equation}
\log p(\mathcal{D}\vert\boldsymbol{\theta})=\sum_{n=1}^{N}\log p(\mathbf{x}\_n\vert\boldsymbol{\theta})
\end{equation}

#### MLE for latent variable models
Consider a latent variable model of the form
\begin{equation}
p(\mathcal{D},\mathbf{z}\_{1:N}\vert\boldsymbol{\theta})=\prod_{n=1}^{N}p(\mathbf{z}\_n\vert\boldsymbol{\theta})p(\mathbf{x}\_n\vert\mathbf{z}\_n,\boldsymbol{\theta})
\end{equation}
Suppose we want to compute the MLE for $\boldsymbol{\theta}$ given $\mathcal{D}$. Since the local latent variables $\mathbf{z}_n$ are hidden, we must marginalize them out to get the local (per example) log marginal likelihood
\begin{equation}
\log p(\mathbf{x}\_n\vert\boldsymbol{\theta})=\log\int\_{z_n}p(\mathbf{x}\_n\vert\mathbf{z}\_n,\boldsymbol{\theta})p(\mathbf{z}\_n\vert\boldsymbol{\theta})\hspace{0.1cm}d z_n
\end{equation}
Computing this integral is intractable, since it corresponds to the normalization constant of the exact posterior. However, since
\begin{equation}
L(\boldsymbol{\theta},\boldsymbol{\psi}\_n\vert\mathbf{x}\_n)\leq\log p(\mathbf{x}\_n\vert\boldsymbol{\theta}),
\end{equation}
we then can optimize the model parameters by maximizing
\begin{equation}
L(\boldsymbol{\theta},\boldsymbol{\psi}\_{1:N}\vert\mathcal{D})\doteq\sum\_{n=1}^{N}L(\boldsymbol{\theta},\boldsymbol{\psi}\_n\vert\mathbf{x}\_n)\leq\log p(\mathcal{D}\vert\boldsymbol{\theta})
\end{equation}
This gives rise to **variational EM**, in which we alternate between maximizing the ELBO w.r.t $\\{\boldsymbol{\psi}\_n\\}$ in the E-step, to give us $q\_{\boldsymbol{\psi}\_n}(\mathbf{z}\_n)$, and then maximizing the ELBO using the new $\boldsymbol{\psi}\_n$ w.r.t $\boldsymbol{\theta}$ in the M-step.

#### Empirical Bayes for fully observed models
Consider a fully observed model of the form
\begin{equation}
p(\mathcal{D},\boldsymbol{\theta}\vert\boldsymbol{\xi})=p(\boldsymbol{\theta}\vert\boldsymbol{\xi})\prod_{n=1}^{N}p(\mathbf{x}\_n\vert\boldsymbol{\theta})
\end{equation}
In the context of Bayesian parameter inference, our goal is to compute the parameter posterior
\begin{equation}
p(\boldsymbol{\theta}\vert\mathcal{D},\boldsymbol{\xi})=\frac{p(\mathcal{D}\vert\boldsymbol{\theta})p(\boldsymbol{\theta}\vert\boldsymbol{\xi})}{p(\mathcal{D}\vert\boldsymbol{\xi})},
\end{equation}
where $\boldsymbol{\theta}$ are the global unknown model parameters (latent variables), and $\boldsymbol{\xi}$ are the hyper-parameters for the prior. If the hyper-parameters are unknown, we can estimate them using empirical Bayes by computing
\begin{equation}
\hat{\boldsymbol{\xi}}=\underset{\boldsymbol{\xi}}{\text{argmax}}\hspace{0.1cm}\log p(\mathcal{D}\vert\boldsymbol{\xi})
\end{equation}
We can use variational EM to compute this. Specifically, we have the lower bound
\begin{equation}
\log p(\mathcal{D}\vert\boldsymbol{\xi})\geq L(\boldsymbol{\xi},\boldsymbol{\psi}\vert\mathcal{D})=\mathbb{E}\_{q_\boldsymbol{\psi}(\boldsymbol{\theta})}\left[\sum_{n=1}^{N}\log p(\mathbf{x}\_n\vert\boldsymbol{\theta})\right]-D_\text{KL}(q_\boldsymbol{\psi}(\boldsymbol{\theta})\Vert p(\boldsymbol{\theta}))
\end{equation}

#### Stochastic VI

#### Amortized VI

## References
[1] Kevin P. Murphy. [Probabilistic Machine Learning: Advanced Topics](http://probml.github.io/book2). The MIT Press, 2023.

## Footnotes
