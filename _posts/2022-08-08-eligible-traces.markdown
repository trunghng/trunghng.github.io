---
layout: post
title:  "Eligible Traces"
date:   2022-08-8 14:11:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning td-learning n-step-td eligible-traces my-rl
description: Eligible Traces
comments: true
---
> Beside [$n$-step TD]({% post_url 2022-07-10-func-approx %}#n-step-td) methods, there is another mechanism called **Eligible traces** that unify TD and Monte Carlo. Setting $\lambda$ in TD($\lambda$) from $0$ to $1$, we end up with a spectrum ranging from TD methods, when $\lambda=0$ to Monte Carlo methods with $\lambda=1$.
<!-- excerpt-end -->

- [The λ-return](#lambda-return)
	- [Offline \\(\lambda\\)-return](#off-lambda-return)
- [TD(\\(\lambda\\))](#td-lambda)
- [Truncated TD Methods](#truncated-td)
- [Online \\(\lambda\\)-return](#onl-lambda-return)
- [True Online TD(λ)](#true-onl-td-lambda)
	- [Dutch Traces in Monte Carlo](#dutch-traces-mc)
- [Sarsa(\\(\lambda\\))](#sarsa-lambda)
- [Variable \\(\lambda\\) and \\(\gamma\\)](#lambda-gamma)
- [Off-policy Traces with Control Variates](#off-policy-traces-control-variates)
- [Tree-Backup(\\(\lambda\\))](#tree-backup-lambda)
- [Other Off-policy Methods with Traces](#other-off-policy-methods-traces)
- [References](#references)
- [Footnotes](#footnotes)

## The $\lambda$-return
{: #lambda-return}
Recall that in [TD-Learning]({% post_url 2022-04-08-td-learning %}#n-step-td-prediction) post, we have defined the $n$-step return as
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}V_{t+n-1}(S_{t+n})
\end{equation}
for all $n,t$ such that $n\geq 1$ and $0\leq t\lt T-n$. After the post of [Function Approximation]({% post_url 2022-07-10-func-approx %}), for any parameterized function approximator, we can generalize that equation into:
\begin{equation}
G_{t:t+n}\doteq R_{t+1}+\gamma R_{t+2}+
\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1}),\hspace{1cm}0\leq t\leq T-n
\end{equation}
where $\hat{v}(s,\mathbf{w})$ is the approximate value of state $s$ given weight vector $\mathbf{w}$. 

We already know that by selecting $n$-step return as the target for a tabular learning update, just as it is for an approximate [SGD update]({% post_url 2022-07-10-func-approx %}#stochastic-grad), we can reach to an optimal point. In fact, a valid update can be also be done toward any average of $n$-step returns for different $n$. For example, we can choose
\begin{equation}
\frac{1}{2}G_{t:t+2}+\frac{1}{2}G_{t:t+4}
\end{equation}
as the target for our update.

The **TD($\lambda$)** is a particular way of averaging $n$-step updates. This average contains all the $n$-step updates, each weighted proportionally to $\lambda^{n-1}$, for $\lambda\in\left[0,1\right]$, and is normalized by a factor of $1-\lambda$ to guarantee that the weights sum to $1$, as:
\begin{equation}
G_t^\lambda\doteq(1-\lambda)\sum_{n=1}^{\infty}\lambda^{n-1}G_{t:t+n}
\end{equation}
The $G_t^\lambda$ is called **$\lambda$-return** of the update. 

This figure below illustrates the backup diagram of TD($\lambda$) algorithm.
<figure>
	<img src="/assets/images/2022-08-08/td-lambda-backup.png" alt="Backup diagram of TD(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 450px; height: 370px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: The backup diagram of TD($\lambda$)</figcaption>
</figure>

### Offline $\lambda$-return
{: #off-lambda-return}
With the definition of $\lambda$-return, we can define the **offline $\lambda$-return** algorithm, which use semi-gradient update and using $\lambda$-return as the target:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\left[G_t^\lambda-\hat{v}(S_t,\mathbf{w}\_t)\right]\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t),\hspace{1cm}t=0,\dots,T-1
\end{equation}

[TODO] Add example

## TD($\lambda$)
{: #td-lambda}
**TD($\lambda$)** improves over the offline $\lambda$-return algorithm since:
- It updates the weight vector $\mathbf{w}$ on every step of an episode rather than only at the end, which leads to a time improvement.
- Its computations are equally distributed in time rather than all at the end of the episode.
- It can be applied to continuing problems rather than just to episodic ones. 

With function approximation, the eligible trace is a vector $\mathbf{z}\_t\in\mathbb{R}^d$ with the same number of components as the weight vector $\mathbf{w}\_t$. Whereas $\mathbf{w}\_t$ is long-term memory, $\mathbf{z}\_t$ on the other hand is a short-term memory, typically lasting less time than the length of an episode.  

In TD($\lambda$), starting at the initial value of zero at the beginning of the episode, on each time step, the eligible trace vector $\mathbf{z}\_t$ is incremented by the value gradient, and then fades away by $\gamma\lambda$:
\begin{align}
\mathbf{z}\_{-1}&\doteq\mathbf{0} \\\\ \mathbf{z}\_t&\doteq\gamma\lambda\mathbf{z}\_t+\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t),\hspace{1cm}0\leq t\lt T\tag{1}\label{1}
\end{align}
where $\gamma$ is the discount factor; $\lambda$ is also called **trace-decay parameter**. On the other hand, the weight vector $\mathbf{w}\_t$ is updated on each step proportional to the scalar [TD errors]({% post_url 2022-04-08-td-learning %}#td_error) and the eligible trace vector $\mathbf{z}\_t$:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\delta_t\mathbf{z}\_t,\tag{2}\label{2}
\end{equation}
where the TD error is defined as
\begin{equation}
\delta_t\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_t)
\end{equation}

Pseudocode of **semi-gradient TD($\lambda$)** is given below.
<figure>
	<img src="/assets/images/2022-08-08/semi-grad-td-lambda.png" alt="Semi-gradient TD(lambda)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

Linear TD($\lambda$) has been proved to converge in the on-policy case if the step size parameter, $\alpha$, is reduced over time according to the [usual conditions]({% post_url 2022-04-08-td-learning %}#stochastic-approx-condition). And also in the continuing discounted case, for any $\lambda$, $\overline{\text{VE}}$ is proven to be within a bounded expansion of the lowest possible error:
\begin{equation}
\overline{\text{VE}}(\mathbf{w}\_\infty)\leq\dfrac{1-\gamma\lambda}{1-\gamma}\min_\mathbf{w}\overline{\text{VE}}(\mathbf{w})
\end{equation}

## Truncated TD Methods
{: #truncated-td}
Since in the offline $\lambda$-return, the target $\lambda$-return is not known until the end of episode. And moreover, in the continuing case, since the $n$-step returns depend on arbitrary large $n$, it maybe never known.
However, the dependence becomes weaker for longer-delayed rewards, falling by $\gamma\lambda$ for each step of delay.  

A natural approximation is to truncate the sequence after some number of steps. In general, we define the **truncated $\lambda$-return** for time $t$, given data only up to some later horizon, $h$, as:
\begin{equation}
G_{t:h}^\lambda\doteq(1-\lambda)\sum_{n=1}^{h-t-1}\lambda^{n-1}G_{t:t+n}+\lambda^{h-t-1}G_{t:h},\hspace{1cm}0\leq t\lt h\leq T
\end{equation}
With this definition of the return, and based on the function approximation version of the $n$-step TD we have defined [before]({% post_url 2022-07-10-func-approx %}#semi-grad-n-step-td-update), we have the **TTD($\lambda$)** is defined as:
\begin{equation}
\mathbf{w}\_{t+n}\doteq\mathbf{w}\_{t+n-1}+\alpha\left[G_{t:t+n}^\lambda-\hat{v}(S_t,\mathbf{w}\_{t+n-1})\right]\nabla_\mathbf{w}\hat{w}(S_t,\mathbf{w}\_{t+n-1}),\hspace{1cm}0\leq t\lt T
\end{equation}
We have the $k$-step $\lambda$-return can be written as:
\begin{align}
G_{t:t+k}^\lambda&=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}G_{t:t+n}+\lambda^{k-1}G_{t:t+k} \\\\ &=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right] \\\\ &\hspace{1cm}+\lambda^{k-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{k-1}R_{t+k}+\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})\right] \\\\ &=R_{t+1}+\gamma\lambda R_{t+2}+\dots+\gamma^{k-1}\lambda^{k-1}R_{t+k} \\\\ &\hspace{1cm}+(1-\lambda)\left[\sum_{n=1}^{k-1}\lambda^{n-1}\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right]+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1}) \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\left[R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1})\right] \\\\ &\hspace{1cm}+\left[\lambda\gamma R_{t+2}+\lambda\gamma^2\hat{v}(S_{t+2},\mathbf{w}\_{t+1})-\lambda\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)\right]+\dots \\\\ &\hspace{1cm}+\left[\lambda^{k-1}\gamma^{k-1}R_{t+k}+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})-\lambda^{k-1}\gamma^{k-1}\hat{v}(S_{t+k-1},\mathbf{w}\_{t+k-2})\right] \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\sum_{i=t}^{t+k-1}(\gamma\lambda)^{i-t}\delta_i',\tag{3}\label{3}
\end{align}
with
\begin{equation}
\delta_t'\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1}),
\end{equation}
where in the third step of the derivation, we use the identity
\begin{equation}
(1-\lambda)(1+\lambda+\dots+\lambda^{k-2})=1-\lambda^{k-1}
\end{equation}
From \eqref{3}, we can see that the $k$-step $\lambda$-return can be written as sums of TD errors if the value function is held constant, which allows us to implement the TTD($\lambda$) algorithm efficiently.

<figure>
	<img src="/assets/images/2022-08-08/ttd-lambda-backup.png" alt="Backup diagram of truncated TD(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 500px; height: 370px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The backup diagram of truncated TD($\lambda$)</figcaption>
</figure>

## Online $\lambda$-return
{: #onl-lambda-return}
The idea of **online $\lambda$-return** involves multiple passes over the episode, one at each horizon, each generating a different sequence of weight vectors.

Let $\mathbf{w}\_t^h$ denote the weights used to generate the value at time $t$ in the sequence up to horizon $h$. The first weight vector $\mathbf{w}\_0^h$ in each sequence is the one that inherited from the previous episode (thus they are the same for all $h$), and the last weight vector $\mathbf{w}\_h^h$ in each sequence defines the weight-vector sequence of the algorithm. At the final horizon $h=T$, we obtain the final weight $\mathbf{w}\_T^T$  which will be passed on to form the initial weights of the next episode.

In particular, we can define the first three sequences as:
\begin{align}
h=1:\hspace{1cm}&\mathbf{w}\_1^1\doteq\mathbf{w}\_0^1+\alpha\left[G_{0:1}^\lambda-\hat{v}(S_0,\mathbf{w}\_0^1)\right]\nabla_\mathbf{w}\hat{v}(S_0,\mathbf{w}\_0^1), \\\\ \\\\ h=2:\hspace{1cm}&\mathbf{w}\_1^2\doteq\mathbf{w}\_0^2+\alpha\left[G_{0:2}^\lambda-\hat{v}(S_0,\mathbf{w}\_0^2)\right]\nabla_\mathbf{w}\hat{v}(S_0,\mathbf{w}\_0^2), \\\\ &\mathbf{w}\_2^2\doteq\mathbf{w}\_1^2+\alpha\left[G_{1:2}^\lambda-\hat{v}(S_t,\mathbf{w}\_1^2)\right]\nabla_\mathbf{w}\hat{v}(S_1,\mathbf{w}\_1^2), \\\\ \\\\ h=3:\hspace{1cm}&\mathbf{w}\_1^3\doteq\mathbf{w}\_0^3+\alpha\left[G_{0:3}^\lambda-\hat{v}(S_0,\mathbf{w}\_0^3)\right]\nabla_\mathbf{w}\hat{v}(S_0,\mathbf{w}\_0^3), \\\\ &\mathbf{w}\_2^3\doteq\mathbf{w}\_1^3+\alpha\left[G_{1:3}^\lambda-\hat{v}(S_1,\mathbf{w}\_1^3)\right]\nabla_\mathbf{w}\hat{v}(S_1,\mathbf{w}\_1^3), \\\\ &\mathbf{w}\_3^3\doteq\mathbf{w}\_2^3+\alpha\left[G_{2:3}^\lambda-\hat{v}(S_2,\mathbf{w}\_2^3)\right]\nabla_\mathbf{w}\hat{v}(S_2,\mathbf{w}\_2^3)
\end{align}
The general form for the update of the **online $\lambda$-return** is
\begin{equation}
\mathbf{w}\_{t+1}^h\doteq\mathbf{w}\_t^h+\alpha\left[G_{t:h}^\lambda-\hat{v}(S_t,\mathbf{w}\_t^h)\right]\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t^h),\hspace{1cm}0\leq t\lt h\leq T,\tag{4}\label{4}
\end{equation}
with $\mathbf{w}\_t\doteq\mathbf{w}\_t^t$, and $\mathbf{w}\_0^h$ is the same for all $h$, we denote this vector as $\mathbf{w}\_{init}$.

The online $\lambda$-return algorithm is fully online, determining a new weight vector $\mathbf{w}\_t$ at each time step $t$ during an episode, using only information available at time $t$. Whereas the offline version passes through all the steps at the time of termination but does not make any updates during the episode.

## True Online TD($\lambda$)
{: #true-onl-td-lambda}
In the online $\lambda$-return, at each time step a sequence of updates is performed. The length of this sequence, and hence the computation per time step, increase over time. 

However, it is possible to compute the weight vector resulting from time step $t+1$, $\mathbf{w}\_{t+1}$, directly from the weight vector resulting from the sequence at time step $t$, $\mathbf{w}\_t$. 

Consider using linear approximation for our task, which gives us 
\begin{align}
\hat{v}(S_t,\mathbf{w}\_t)&=\mathbf{w}\_t^\intercal\mathbf{x}\_t; \\\\ \nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)&=\mathbf{x}\_t,
\end{align}
where $\mathbf{x}\_t=\mathbf{x}(S_t)$ as usual.

We begin by rewriting \eqref{4}, as
\begin{align}
\mathbf{w}\_{t+1}^h&\doteq\mathbf{w}\_t^h+\alpha\left[G_{t:h}^\lambda-\hat{v}(S_t,\mathbf{w}\_t^h)\right]\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t^h) \\\\ &=\mathbf{w}\_t^h+\alpha\left[G_{t:h}^\lambda-\left(\mathbf{w}\_t^h\right)^\intercal\mathbf{x}\_t\right]\mathbf{x}\_t \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\\intercal\right)\mathbf{w}\_t^h+\alpha\mathbf{x}\_t G_{t:h}^\lambda,
\end{align}
where $\mathbf{I}$ is the identity matrix. With this equation, consider $\mathbf{w}\_t^h$ in the cases of $t=1$ and $t=2$, we have:
\begin{align}
\mathbf{w}\_1^h&=\left(\mathbf{I}-\alpha\mathbf{x}\_0\mathbf{x}\_0^\intercal\right)\mathbf{w}\_0^h+\alpha\mathbf{x}\_0 G_{0:h}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_0\mathbf{x}\_0^\intercal\right)\mathbf{w}\_{init}+\alpha\mathbf{x}\_0 G_{0:h}^\lambda, \\\\ \mathbf{w}\_2^h&=\left(\mathbf{I}-\alpha\mathbf{x}\_1\mathbf{x}\_1^\intercal\right)\mathbf{w}\_1^h+\alpha\mathbf{x}\_1 G_{1:h}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_1\mathbf{x}\_1^\intercal\right)\left(\mathbf{I}-\alpha\mathbf{x}\_0\mathbf{x}\_0^\intercal\right)\mathbf{w}\_{init}+\alpha\left(\mathbf{I}-\alpha\mathbf{x}\_1\mathbf{x}\_1^\intercal\right)\mathbf{x}\_0 G_{0:h}^\lambda+\alpha\mathbf{x}\_1 G_{1:h}^\lambda
\end{align}
In general, for $t\leq h$, we can write:
\begin{equation}
\mathbf{w}\_t^h=\mathbf{A}\_0^{t-1}\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i G_{i:h}^\lambda,
\end{equation}
where $\mathbf{A}\_i^j$ is defined as:
\begin{equation}
\mathbf{A}\_i^j\doteq\left(\mathbf{I}-\alpha\mathbf{x}\_j\mathbf{x}\_j^\intercal\right)\left(\mathbf{I}-\alpha\mathbf{x}\_{j-1}\mathbf{x}\_{j-1}^\intercal\right)\dots\left(\mathbf{I}-\alpha\mathbf{x}\_i\mathbf{x}\_i^\intercal\right),\hspace{1cm}j\geq i,
\end{equation}
with $\mathbf{A}\_{j+1}^j\doteq\mathbf{I}$. Hence, we can express $\mathbf{w}\_t$ as:
\begin{equation}
\mathbf{w}\_t=\mathbf{w}\_t^t=\mathbf{A}\_0^{t-1}\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i G_{i:t}^\lambda\tag{5}\label{5}
\end{equation}
Using \eqref{3}, we have:
\begin{align}
G_{i:t+1}^\lambda-G_{i:t}^\lambda&=\mathbf{w}\_i^\intercal\mathbf{x}\_i+\sum_{j=1}^{t}(\gamma\lambda)^{j-i}\delta_j'-\left(\mathbf{w}\_i^\intercal\mathbf{x}\_i+\sum_{j=1}^{t-1}(\gamma\lambda)^{j-i}\delta_j'\right) \\\\ &=(\gamma\lambda)^{t-i}\delta_t'\tag{6}\label{6}
\end{align}
with the TD error, $\delta_t'$ is defined as earlier:
\begin{equation}
\delta_t'\doteq R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\tag{7}\label{7}
\end{equation}
Using \eqref{5}, \eqref{6} and \eqref{7}, we have:
\begin{align}
\mathbf{w}\_{t+1}&=\mathbf{A}\_0^t\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t+1}^\lambda \\\\ &=\mathbf{A}\_0^t\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t+1}^\lambda+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\mathbf{A}\_0^t\mathbf{w}\_0+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t}^\lambda+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\left(\mathbf{A}\_0^{t-1}\mathbf{w}\_0+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i G_{t:t+1}^\lambda\right) \\\\ &\hspace{1cm}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_t\mathbf{x}\_t\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t' \\\\ &\hspace{1cm}+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t+\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t-\mathbf{w}\_t^\intercal\mathbf{x}\_t\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\left(\delta_t+\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\delta_t+\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\left(\mathbf{z}\_t-\mathbf{x}\_t\right),\tag{8}\label{8}
\end{align}
where in the eleventh step, we define $\mathbf{z}\_t$ as:
\begin{equation}
\mathbf{z}\_t\doteq\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i},
\end{equation}
and in the twelfth step, we also define $\delta_t$ as:
\begin{align}
\delta_t&\doteq\delta_t'-\mathbf{w}\_t^\intercal\mathbf{x}\_t+\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t \\\\ &=R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_t^\intercal\mathbf{x}\_t,
\end{align}
which is the same as the TD error of TD($\lambda$) we have defined earlier. 

We then need to derive an update rule to compute $\mathbf{z}\_t$ from $\mathbf{z}\_{t-1}$, as:
\begin{align}
\mathbf{z}\_t&=\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i} \\\\ &=\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i}+\mathbf{x}\_t \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\gamma\lambda\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i(\gamma\lambda)^{t-i-1}+\mathbf{x}\_t \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\gamma\lambda\mathbf{z}\_{t-1}+\mathbf{x}\_t \\\\ &=\gamma\lambda\mathbf{z}\_{t-1}+\left(1-\alpha\gamma\lambda\left(\mathbf{z}\_t^\intercal\mathbf{x}\_t\right)\right)\mathbf{x}\_t\tag{9}\label{9}
\end{align}
Equation \eqref{8} and \eqref{9} form the update of the **true online TD($\lambda$)** algorithm:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\delta_t\mathbf{z}\_t+\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\left(\mathbf{z}t\_t-\mathbf{x}\_t\right),
\end{equation}
where
\begin{align}
\mathbf{z}\_t&\doteq\gamma\lambda\mathbf{z}\_{t-1}+\left(1-\alpha\gamma\lambda\left(\mathbf{z}\_t^\intercal\mathbf{x}\_t\right)\right)\mathbf{x}\_t,\tag{10}\label{10} \\\\ \delta_t&\doteq R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_t^\intercal\mathbf{x}\_t
\end{align}
Pseudocode of the algorithm is given below.
<figure>
	<img src="/assets/images/2022-08-08/true-onl-td-lambda.png" alt="True Online TD(lambda)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

The eligible trace \eqref{10} is called **dutch trace** to distinguish it from the trace \eqref{1} of TD($\lambda$), which is called **accumulating trace**. 

There is another kind of trace called **replacing trace**, defined for the tabular case or for binary feature vectors
\begin{equation}
z_{i,t}\doteq\begin{cases}1 &\text{if }x_{i,t}=1 \\\\ \gamma\lambda z_{i,t-1} &\text{if }x_{i,t}=0\end{cases}
\end{equation}

### Dutch Traces In Monte Carlo
{: #dutch-traces-mc}

## Sarsa($\lambda$)
{: #sarsa-lambda}
To apply the use off eligible traces on control problems, we begin by defining the $n$-step return, which is the same as what we have defined [before]({% post_url 2022-07-10-func-approx %}#n-step-return):
\begin{equation}
G_{t:t+n}\doteq\ R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{q}(S_{t+n},A_{t+n},\mathbf{w}\_{t+n-1}),\hspace{1cm}t+n\lt T\tag{11}\label{11}
\end{equation}
with $G_{t:t+n}\doteq G_t$ if $t+n\geq T$. With this definition of the return, the action-value form of offline $\lambda$-return can be defined as:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\left[G_t^\lambda-\hat{q}(S_t,A_t,\mathbf{w}\_t)\right]\nabla_\mathbf{w}\hat{q}(S_t,A_t,\mathbf{w}\_t),\hspace{1cm}t=0,\dots,T-1
\end{equation}
where $G_t^\lambda\doteq G_{t:\infty}^\lambda$. 

The TD method for action values, known as **Sarsa($\lambda$)**, approximates this forward view and has the same update rule as TD($\lambda$):
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\delta_t\mathbf{z}\_t,
\end{equation}
except that the TD error, $\delta_t$, is defined in terms of action-value function:
\begin{equation}
\delta_t\doteq R_{t+1}+\gamma\hat{q}(S_{t+1},A_{t+1},\mathbf{w}\_t)-\hat{q}(S_t,A_t,\mathbf{w}\_t),
\end{equation}
and so it is with eligible trace vector:
\begin{align}
\mathbf{z}\_{-1}&\doteq\mathbf{0}, \\\\ \mathbf{z}&\_t\doteq\gamma\lambda\mathbf{z}\_{t-1}+\nabla_\mathbf{w}\hat{q}(S_t,A_t,\mathbf{w}\_t),\hspace{1cm}0\leq t\lt T
\end{align}
<figure>
	<img src="/assets/images/2022-08-08/sarsa-lambda-backup.png" alt="Backup diagram of Sarsa(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 450px; height: 390px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The backup diagram of Sarsa($\lambda$)</figcaption>
</figure>
Pseudocode of the Sarsa($\lambda$) is given below.
<figure>
	<img src="/assets/images/2022-08-08/sarsa-lambda.png" alt="Sarsa(lambda)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

There is also an action-value version of the online $\lambda$-return algorithm, and its efficient implementation as true online TD($\lambda$), called **True online TD($\lambda$)**, which can be achived by using $n$-step return \eqref{11} instead (which also leads to the change of $\mathbf{x}\_t=\mathbf{x}(S_t)$ to $\mathbf{x}\_t=\mathbf{x}(S_t,A_t)$). 

Pseudocode of the true online Sarsa($\lambda$) is given below.
<figure>
	<img src="/assets/images/2022-08-08/true-online-sarsa-lambda.png" alt="True online Sarsa(lambda)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

## Variable $\lambda$ and $\gamma$
{: #lambda-gamma}
We can generalize the degree of bootstrapping and discounting beyond constant parameters to functions potentially dependent on the state and action. In other words, each time step $t$, we will have a different $\lambda$ and $\gamma$, denoted as $\lambda_t$ and $\gamma_t$. 

In particular, say $\lambda:\mathcal{S}\times\mathcal{A}\to[0,1]$ such that $\lambda_t\doteq\lambda(S_t,A_t)$ and similarly, $\gamma:\mathcal{S}\to[0,1]$ such that $\gamma_t\doteq\gamma(S_t)$.

With this definition of $\gamma$, the return can be rewritten generally as:
\begin{align}
G_t&\doteq R_{t+1}+\gamma_{t+1}G_{t+1} \\\\ &=R_{t+1}+\gamma_{t+1}R_{t+2}+\gamma_{t+1}\gamma_{t+2}R_{t+3}+\dots \\\\ &=\sum_{k=t}^{\infty}\left(\prod_{i=t+1}^{k}\gamma_i\right)R_{k+1},
\end{align}
where we require that $\prod_{k=t}^{\infty}\gamma_k=0$ with probability $1$ for all $t$ to assure the sums are finite. 

The generalization of $\lambda$ also lets us rewrite the state-based $\lambda$-return as:
\begin{equation}
G_t^{\lambda s}\doteq R_{t+1}+\gamma_{t+1}\Big((1-\lambda_{t+1})\hat{v}(S_{t+1},\mathbf{w}\_t)+\lambda_{t+1}G_{t+1}^{\lambda s}\Big),\tag{12}\label{12}
\end{equation}
where $G_t^{\lambda s}$ denotes that this $\lambda$
-return is bootstrapped from state values, and hence the $G_t^{\lambda a}$ denotes the $\lambda$-return that bootstraps from action values. The Sarsa form of action-based $\lambda$-return is defined as:
\begin{equation}
G_t^{\lambda a}\doteq R_{t+1}+\gamma_{t+1}\Big((1-\lambda_{t+1})\hat{q}(S_{t+1},A_{t+1},\mathbf{w}\_t)+\lambda_{t+1}G_{t+1}^{\lambda a}\Big),
\end{equation}
and the Expected Sarsa form of its can be defined as:
\begin{equation}
G_t^{\lambda a}\doteq R_{t+1}+\gamma_{t+1}\Big((1-\lambda_{t+1})\bar{V}\_t(S_{t+1})+\lambda_{t+1}G_{t+1}^{\lambda a}\Big),\tag{13}\label{13}
\end{equation}
where the [expected approximate value]({% post_url 2022-04-08-td-learning %}#expected-approximate-value) is generalized to function approximation as:
\begin{equation}
\bar{V}\_t\doteq\sum_a\pi(a|s)\hat{q}(s,a,\mathbf{w}\_t)\tag{14}\label{14}
\end{equation}

## Off-policy Traces with Control Variates
{: #off-policy-traces-control-variates}
We can also apply the use of importance sampling with eligible traces. 

We begin with the new definition of $\lambda$-return, which is achieved by generalizing the $\lambda$-return \eqref{12} with the idea of [control variates on $n$-step off-policy return]({% post_url 2022-04-08-td-learning %}#n-step-return-control-variate-state-value):
\begin{equation}
G_t^{\lambda s}\doteq\rho_t\Big(R_{t+1}+\gamma_{t+1}\big((1-\lambda_{t+1})\hat{v}(S_{t+1},\mathbf{w}\_t)+\lambda_{t+1}G_{t+1}^{\lambda s}\big)\Big)+(1-\rho_t)\hat{v}(S_t,\mathbf{w}\_t),
\end{equation}
where the single-step importance sampling ratio $\rho_t$ is defined as usual:
\begin{equation}
\rho_t\doteq\frac{\pi(A_t|S_t)}{b(A_t|S_t)}
\end{equation}
Much like the other returns, the truncated version of this return can be approximated simply in terms of sums of state-based TD errors:
\begin{equation}
G_t^{\lambda s}\approx\hat{v}(S_t,\mathbf{w}\_t)+\rho_t\sum_{k=t}^{\infty}\delta_k^s\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i,
\end{equation}
where the state-based TD error, $\delta_t^s$, is defined as:
\begin{equation}
\delta_t^s\doteq R_{t+1}+\gamma_{t+1}\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_t),
\end{equation}
with the approximation becoming exact if the approximate value function does not change. 

With this appximation, we have that:
\begin{align}
\mathbf{w}\_{t+1}&=\mathbf{w}\_t+\alpha\left(G_t^{\lambda s}-\hat{v}(S_t,\mathbf{w}\_t)\right)\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t) \\\\ &\approx\mathbf{w}\_t+\alpha\rho_t\left(\sum_{k=t}^{\infty}\delta_k^s\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i\right)\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)
\end{align}
This is one time step of a forward view. And in fact, the forward-view update, summed over time, is approximately equal to a backward-view update, summed over time. Since the sum of the forward-view update over time is:
\begin{align}
\sum_{t=1}^{\infty}(\mathbf{w}\_{t+1}-\mathbf{w}\_t)&\approx\sum_{t=1}^{\infty}\sum_{k=t}^{\infty}\alpha\rho_t\delta_k^s\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i \\\\ &=\sum_{k=1}^{\infty}\sum_{t=1}^{k}\alpha\rho_t\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)\delta_k^s\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i \\\\ &=\sum_{k=1}^{\infty}\alpha\delta_k^s\sum_{t=1}^{k}\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i,\tag{15}\label{15}
\end{align}
where in the second step, we use the summation rule: $\sum_{t=x}^{y}\sum_{k=t}^{y}=\sum_{k=x}^{y}\sum_{t=x}^{k}$. 

Let $\mathbf{z}\_k$ is defined as:
\begin{align}
\mathbf{z}\_k &=\sum_{t=1}^{k}\rho_t\nabla_\mathbf{w}\hat{v}\left(S_t, \mathbf{w}\_t\right)\prod_{i=t+1}^{k} \gamma_i\lambda_i\rho_i \\\\ &=\sum_{t=1}^{k-1}\rho_t\nabla_\mathbf{w}\hat{v}\left(S_t,\mathbf{w}\_t\right)\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i+\rho_k\nabla_\mathbf{w}\hat{v}\left(S_k,\mathbf{w}\_k\right) \\\\ &=\gamma_k\lambda_k\rho_k\underbrace{\sum_{t=1}^{k-1}\rho_t\nabla_\mathbf{w}\hat{v}\left(S_t,\mathbf{w}\_t\right)\prod_{i=t+1}^{k-1}\gamma_i\lambda_i\rho_i}\_{\mathbf{z}\_{k-1}}+\rho_k\nabla_\mathbf{w}\hat{v}\left(S_k,\mathbf{w}\_k\right) \\\\ &=\rho_k\big(\gamma_k\lambda_k\mathbf{z}\_{k-1}+\nabla_\mathbf{w}\hat{v}\left(S_k,\mathbf{w}\_k\right)\big)
\end{align}
Then we can rewrite \eqref{15} as:
\begin{equation}
\sum_{t=1}^{\infty}\left(\mathbf{w}\_{t+1}-\mathbf{w}\_t\right)\approx\sum_{k=1}^{\infty}\alpha\delta_k^s\mathbf{z}\_k,
\end{equation}
which is sum of the backward-view update over time, with the eligible trace vector is defined as:
\begin{equation}
\mathbf{z}\_t\doteq\rho_t\left(\gamma_t\lambda_t\mathbf{z}\_{t-1}+\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t)\right)\tag{16}\label{16}
\end{equation}
Using this eligibile trace with the parameter update rule \eqref{2} of TD($\lambda$), we obtain a general TD($\lambda$) algorithm that can be applied to either on-policy or off-policy data.
- In the on-policy case, the algorithm is exactly TD($\lambda$) because $\rho_t=1$ for all $t$ and \eqref{16} becomes the accumulating trace \eqref{1} with extending to variable $\lambda$ and $\gamma$.
- In the off-policy case, the algorithm often works well but, as a semi-gradient method, is not guaranteed to be stable. 

For action-value function, we generalize the definition of the $\lambda$-return \eqref{13} of Expected Sarsa with the idea of [control variate]({% post_url 2022-04-08-td-learning %}#n-step-return-control-variate-action-value):
\begin{align}
G_t^{\lambda a}&\doteq R_{t+1}+\gamma_{t+1}\Big((1-\lambda_{t+1})\bar{V}\_t(S_{t+1})+\lambda_{t+1}\big[\rho_{t+1}G_{t+1}^{\lambda a}+\bar{V}\_t(S_{t+1}) \\\\ &\hspace{2cm}-\rho_{t+1}\hat{q}(S_{t+1},A_{t+1},\mathbf{w}\_t)\big]\Big) \\\\ &=R_{t+1}+\gamma_{t+1}\Big(\bar{V}\_t(S_{t+1})+\lambda_{t+1}\rho_{t+1}\left[G_{t+1}^{\lambda a}-\hat{q}(S_{t+1},A_{t+1},\mathbf{w}\_t)\right]\Big),
\end{align}
where the expected approximate value $\bar{V}\_t(S_{t+1})$ is as given by \eqref{14}.

Similar to the others, this $\lambda$-return can also be written approximately as the sum of TD errors
\begin{equation}
G_t^{\lambda a}\approx\hat{q}(S_t,A_t,\mathbf{w}\_t)+\sum_{k=t}^{\infty}\delta_k^a\prod_{i=t+1}^{k}\gamma_i\lambda_i\rho_i,
\end{equation}
with the action-based TD error is defined in terms of the expected approximate value:
\begin{equation}
\delta_t^a=R_{t+1}+\gamma_{t+1}\bar{V}\_t(S_{t+1})-\hat{q}(S_t,A_t,\mathbf{w}\_t)\tag{17}\label{17}
\end{equation}
Like the state value function case, this approximation also becomes exact if the appriximate value function does not change.

Similar to the state case \eqref{16}, we can also define the eligible trace for action values:
\begin{equation}
\mathbf{z}\_t\doteq\gamma_t\lambda_t\rho_t\mathbf{z}\_{t-1}+\nabla_\mathbf{w}\hat{q}(S_t,A_t,\mathbf{w}\_t)
\end{equation}
Using this eligibile trace with the parameter update rule \eqref{2} of TD($\lambda$) and the expectation-based TD error \eqref{17}, we end up with an Expected Sarsa($\lambda$) algorithm that can applied to either on-policy or off-policy data.
- In the on-policy case with constant $\lambda$ and $\gamma$, this becomes the Sarsa($\lambda$) algorithm.

## Tree-Backup($\lambda$)
{: #tree-backup-lambda}

<figure>
	<img src="/assets/images/2022-08-08/tree-backup-lambda-backup.png" alt="Backup diagram of Tree Backup(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 450px; height: 390px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: The backup diagram of Tree Backup($\lambda$)</figcaption>
</figure>

## Other Off-policy Methods with Traces
{: #other-off-policy-methods-traces}

## References
{: #references}
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] Doina Precup & Richard S. Sutton & Satinder Singh. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cs_faculty_pubs/80) (2000). ICML '00 Proceedings of the Seventeenth International Conference on Machine Learning. 80. 

[3] Deepmind x UCL. [Reinforcement Learning Lecture Series 2021](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021). 

[4] Harm van Seijen & A. Rupam Mahmood & Patrick M. Pilarski & Marlos C. Machado & Richard S. Sutton. [True Online Temporal-Difference Learning](http://jmlr.org/papers/v17/15-599.html). Journal of Machine Learning Research. 17(145):1−40, 2016. 

[5] Hado Van Hasselt & A. Rupam Mahmood & Richard S. Sutton. [Off-policy TD(λ) with a true online equivalence](https://www.researchgate.net/publication/263653431_Off-policy_TDl_with_a_true_online_equivalence). Uncertainty in Artificial Intelligence - Proceedings of the 30th Conference, UAI 2014. 

[6] Shangtong Zhang. [Reinforcement Learning: An Introduction implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction). 

## Footnotes
{: #footnotes}