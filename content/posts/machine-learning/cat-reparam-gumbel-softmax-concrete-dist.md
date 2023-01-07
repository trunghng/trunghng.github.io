---
title: "Categorical Reparameterization with Gumbel-Softmax & Concrete Distribution"
date: 2023-01-02T13:49:15+07:00
tags: [machine-learning gumbel]
math: true
eqn-number: true
---
Notes on using Gumbel-Softmax & Concrete Distribution in Categorical sampling.
<!--more-->

## Gumbel distribution{#gubel-dist}
Gumbel distribution, denoted $\text{Gumbel}(\mu,\beta)$, is a continous probability distribution whose cummulative density function (CDF) is given by
\begin{equation}
F(x)=\exp\left(-\exp\left(-\frac{x-\mu}{\beta}\right)\right),
\end{equation}
which implies that the probability density function (PDF) is given as
\begin{equation}
f(x)=F'(x)=\frac{1}{\beta}e^{-(e^{-z}+z)},
\end{equation}
where
\begin{equation}
z=\frac{x-\mu}{\beta}
\end{equation}
The **standard Gumbel** distribution, denoted $\text{Gumbel}(0,1)$, is specified at location $\mu=0$ and unit scale $\beta=1$, whose densitiy functions, i.e. CDF and PDF, are then explicitly given as
\begin{align}
F(x)&=e^{-e^{-x}} \\\\ f(x)&=e^{-(e^{-x}+x)}\label{eq:gd.1}
\end{align}
Below are some illustrations of Gumbel distribution.
<figure>
	<img src="/images/cat-reparam-gumbel-softmax-concrete-dist/gumbel-dist.png" alt="normal distribution" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: Gumbel distribution $\text{Gumbel}(\mu,\beta)$. The code can be found <a href='https://github.com/trunghng/visualization-collection/blob/main/distributions/gumbel.py' target='_blank'>here</a></figcaption>
</figure>

Since the quantile function, i.e. inverse of the CDF, of Gumbel r.v $\text{Gumbel}(\mu,\beta)$ is referred as 
\begin{equation}
Q(p)=\mu-\beta\log(-\log p),
\end{equation}
which implies that the standard Gumbel random variable <span id='std-unif-gumbel'></span>$X\sim\text{Gumbel}(0,1)$ can be sampled using inverse transform sampling by first drawing $U\sim\text{Unif}(0, 1)$ and then computing
\begin{equation}
X=−\log(−\log U)
\end{equation}

## Optimizing Stochastic Computation Graph{#opt-stochastic-computation-graph}
Consider the following Stochastic Computation Graph
<figure>
	<img src="/images/cat-reparam-gumbel-softmax-concrete-dist/sgc.png" alt="SGC" style="display: block; margin-left: auto; margin-right: auto; width: 50%; height: 50%"/>
	<figcaption></figcaption>
</figure>

where
- $w,\phi,\theta$ denote input nodes.
- $X$ is a stochastic node, which is given by sampling according to $p_\phi(x\vert w)$.
- $f$ is a deterministic node, i.e. $f_\theta(x)$ is a deterministic function at $X$.

The graph corresponds to the objective function
\begin{equation}
L(\theta,\phi)=\mathbb{E}\_{X\sim p_\phi(x)}\big[f_\theta(X)\big],\label{eq:oscg.1}
\end{equation}
where without loss of generality, we have considered $w$ as a constant.

Consider the backpropagation through the computation graph, we have that the gradient w.r.t $\theta$ of the cost function is given by
\begin{equation}
\nabla_\theta L(\theta,\phi)=\nabla_\theta\mathbb{E}\_{X\sim p_\phi(x)}\big[f_\theta(X)\big]=\mathbb{E}\_{X\sim p_\phi(x)}\big[\nabla_\theta f_\theta(X)\big],\label{eq:oscg.2}
\end{equation}
which, as an expectation, can be estimated using Monte Carlo method. In particular, let $X_1,\ldots,X_s$ be $s$ i.i.d samples drawn from $p_\phi(x)$, the gradient given in \eqref{eq:oscg.2} can be estimated with the unbiased
\begin{equation}
\nabla_\theta L(\theta,\phi)\approx\frac{1}{s}\sum_{i=1}^{s}\nabla_\theta f_\theta(X_i)
\end{equation}
On the other hands, taking the gradient w.r.t paramters $\phi$ gives us
\begin{equation}
\nabla_\phi L(\theta,\phi)=\nabla_\phi\int p_\phi(x)f_\theta(x)dx=\int f_\theta(x)\nabla_\phi p_\phi(x)dx,\label{eq:oscg.3}
\end{equation}
which can not be estimated directly using Monte Carlo sampling.

### Score Function Estimators{#sfe}
**Score function estimatior** ultilizes the **log-likelihood trick** to rewrite the gradient in \eqref{eq:oscg.3} in an expectation form
\begin{align}
\nabla_\phi L(\theta,\phi)&=\int f_\theta(x)\nabla_\phi p_\phi(x)dx \\\\ &=\int f_\theta(x)p_\phi(x)\nabla_\phi\log p_\phi(x)dx \\\\ &=\mathbb{E}\_{X\sim p_\phi(x)}\big[f_\theta(X)\nabla_\phi\log p_\phi(X)\big],
\end{align}
which analogously can be estimated by $s$ samples $X_1,\ldots,X_s\overset{\text{i.i.d}}{\sim} p_\phi(x)$
\begin{equation}
\nabla_\phi L(\theta,\phi)\approx\frac{1}{s}\sum_{i=1}^{s}f_\theta(X_i)\nabla_\phi\log p_\phi(X_i)
\end{equation}

### Reparameterization Trick{#reparam-trick}
In some circumstances, it could be helpful that instead of sampling from $p_\phi(x)$, we first sample $Z$ according to some fixed distribution $q(z)$ and then transform the sample to $x$ using some function $x=g_\phi(z)$.

For instance, by properties of the Normal distribution, a Gaussian sample $X\sim\mathcal{N}(\mu,\sigma^2)$ can always be obtained through a standard Normal $Z\sim\mathcal{N}(0,1)$ by computing $X=g_{\mu,\sigma}(Z)=\mu+\sigma Z$.

This **reparameterization trick**, $x=g_\phi(z)$, let us transfer the dependence on $\phi$ from $p$ into $f$ by writing
\begin{equation}
f_\theta(x)=f_\theta(g_\phi(z)),
\end{equation}
which enables the possibility of reducing the problem of estimating the gradient of w.r.t parameters of a distribution into a more trivial task of estimating the gradient w.r.t parameters of a deterministic function.

Applying this **reparameterization trick** allows us to rewrite the objective function given in \eqref{eq:oscg.1} as
\begin{equation}
L(\theta,\phi)=\mathbb{E}\_{X\sim p_\phi(x)}\big[f_\theta(X)\big]=\mathbb{E}\_{Z\sim q(z)}\big[f_\theta(g_\phi(Z))\big],
\end{equation}
which has the gradient w.r.t $\phi$ given by
\begin{align}
\nabla_\phi L(\theta,\phi)&=\nabla_\phi\mathbb{E}\_{Z\sim q(z)}\big[f_\theta(g_\phi(Z))\big] \\\\ &=\mathbb{E}\_{Z\sim q(z)}\big[\nabla_\phi f_\theta(g_\phi(Z))\big] \\\\ &=\mathbb{E}\_{Z\sim q(z)}\big[f_\theta'(g_\phi(Z))\nabla_\phi g_\phi(Z)\big]
\end{align}

## Gumbel-Max Trick{#gummbel-max-trick}
Using the idea of reparameterization trick, **Gumbel-Max trick** refers to an approach that allows us to sample from a **categorical distribution**[^1] through sampling according to Gumbel distribution.

First let $D$ be a categorical variable with class probabilities $\pi_1,\pi_2,\ldots,\pi_k$ for $\sum_{i=1}^{k}\pi_i=1$ and without loss of generality we can assume that zero category probability excluded, i.e. $\pi_i>0$. Thus, we can express each sample drawn from the distribution as a $k$-dimensional one-hot vector lying in the corner (or vertex) of a $(k-1)$-dimensional probability simplex $\Delta^{k-1}$[^2]. In particular, each categorical sample is in form of
\begin{equation}
D=\left[\begin{matrix}D_1 \\\\ \vdots \\\\ D_k\end{matrix}\right],
\end{equation}
where $\sum_{i=1}^{k}D_i=1$; for $i=1,\ldots,k$ we have $D_i\in\\{0,1\\}$ and $P(D_i=1)=\pi_i$.

Usually, we rewrite each class probability as a softmax function
\begin{equation}
\pi_i=\frac{\exp(\alpha_i)}{\sum_{j=1}^{k}\exp(\alpha_j)}
\end{equation}
where $\alpha_i\in(-\infty,0)$.

Gumbel-max trick provides us another way to get samples following this discrete distribution through drawing samples from Gumbel distribution. The trick is described as follow.

Consider $k$ unit-scaled Gumble random variables $G_1,\ldots,G_k$ where $G_i\sim\text{Gumble}(\alpha_i,1)$. Also, let us denote the CDF corresponds to $\text{Gumble}(\alpha_i,1)$ as $F_i(x)$, which implies that
\begin{equation}
P(G_i\leq x)=F_i(x)=\exp(-\exp(-x+\alpha_i)),
\end{equation}
and also its PDF
\begin{equation}
f_i(x)=\exp(-\exp(-x+\alpha_i)-x+\alpha_i)
\end{equation}
We have that the probabilty that $G_m$ taking the maximal value across $k$ samples can be computed as
\begin{align} 
P\left(G_m=\max_{i=1,\ldots,k}G_i\Bigg\vert G_m\right)&=P\big(G_1\leq G_m,\ldots, G_k\leq G_m\big\vert G_m\big) \\\\ &=\prod_{i=1,i\neq m}^{k}P(G_i\leq G_m\vert G_m) \\\\ &=\prod_{i=1,i\neq m}^{k}F_i(G_m) \\\\ &=\prod_{i=1,i\neq m}^{k}\exp(-\exp(-G_m+\alpha_i))
\end{align}
Therefore, integrating over sample space of $G_m$, the probability that an arbitrary index $m$ corresponds to the largest sample $G_m$, i.e. $m=\underset{i=1,\ldots,k}{\text{argmax}}G_i$ is computed by
\begin{align}
&P\left(m=\underset{i}{\text{argmax}}G_i\right)\nonumber \\\\ &=\int f_m(x)\left(G_m=\max_{i=1,\ldots,k}G_i\Bigg\vert G_m=x\right)dx \\\\ &=\int\exp(-\exp(-x+\alpha_m)-x+\alpha_m)\prod_{i=1,i\neq m}^{k}\exp(-\exp(-x+\alpha_i))dx \\\\ &=\int\exp(-x+\alpha_m)\prod_{i=1}^{k}\exp(-\exp(-x+\alpha_i))dx \\\\ &=\exp(\alpha_m)\int\exp(-x)+\exp\Bigg(-\exp(-x)\sum_{i=1}^{k}\exp(\alpha_i)\Bigg)dx \\\\ &=\frac{\exp(\alpha_m)}{\sum_{i=1}^{k}\exp(\alpha_i)}\int\exp(-\exp(-x)-x)dx\label{eq:gmt.1} \\\\ &=\frac{\exp(\alpha_m)}{\sum_{i=1}^{k}\exp(\alpha_i)},
\end{align}
where the last step is due to that the integrand in \eqref{eq:gmt.1} is the PDF of a standard Gumbel distribution, as defined in \eqref{eq:gd.1}, which therefore integrates to $1$.

Hence, we have that
\begin{equation}
P\left(m=\underset{i}{\text{argmax}}G_i\right)=\pi_m
\end{equation}
Since a $\text{Gumble}(\mu,\beta)$ can always be obtained from a standard $\text{Gumble}(0,1)$ by scaling it with $\beta$ then translationing with $\mu$, then $G_i\sim\text{Gumble}(\alpha_i,1)$ can be computed as
\begin{equation}
G_i=g+\alpha_i,
\end{equation}
where $g\sim\text{Gumbel}(0,1)$, which as mentioned [above](#std-unif-gumbel), can be obtained with
\begin{equation}
g=-\log(-\log u),
\end{equation}
where $u$ is drawn from an Uniform distribution, $u\sim\text{Unif}(0,1)$.

To summarize this, the Gumbel-max trick proceeds as: let $U_1,\ldots,U_k\overset{\text{i.i.d}}{\sim}\text{Unif}(0,1)$ and let
\begin{equation}
m=\max_{i=1,\ldots,k}\log\pi_i-\log(-\log U_i),
\end{equation}
where $\pi=(\pi_1,\ldots,\pi_k)$ with $\pi_i\in(0,\infty)$ is an unnormalized parameterization of a discrete distribution $D\sim\text{Discrete}(\pi)$. Then each sample $D$ can be expressed as a one-hot vector
\begin{equation}
D=\left[\begin{matrix}D_1 \\\\ \vdots \\\\ D_k\end{matrix}\right],
\end{equation}
where $\sum_{i=1}^{k}D_i=1$ and $D_i\in\\{0,1\\}$ for $i=1,\ldots,k$ with
\begin{equation}
P(D_i=1)=\frac{\pi_i}{\sum_{j=1}^{k}\pi_j}
\end{equation}

## Gumbel-Softmax & Concrete Distribution{#gumbel-softmax-concrete}


## References
[1] Eric Jang, Shixiang Gu, Ben Poole. [Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144). ICLR 2017.

[2] Chris J. Maddison, Andriy Mnih, Yee Whye Teh. [The Concrete Distribution: A Continuous Relaxation of Discrete Random Variables](https://arxiv.org/abs/1611.00712). ICLR 2017.

[3] John Schulman, Theophane Weber, Nicolas Heess, Pieter Abbeel. [Gradient Estimation Using Stochastic Computation Graphs](https://dl.acm.org/doi/10.5555/2969442.2969633). NIPS, 2015.

[4] Wikipedia. [Gumbel distribution](https://en.wikipedia.org/wiki/Gumbel_distribution).

[5] Chris J. Maddison, Daniel Tarlow, Tom Minka. [A$^*$ Sampling](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=1e5aeb91ef853facf79502af62b5e28f6e5fd031). NIPS, 2014.

## Footnotes
[^1]: The generalization of Bernoulli distribution into $k$ dimensions.
[^2]: This is due to the constraint that the probabilities asscociated with each category sum to $1$ reduces the dimensionality by $1$.
