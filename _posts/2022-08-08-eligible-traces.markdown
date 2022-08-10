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
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\delta_t\mathbf{z}\_t,
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
G_{t:t+k}^\lambda&=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}G_{t:t+n}+\lambda^{k-1}G_{t:t+k} \\\\ &=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right] \\\\ &\hspace{1cm}+\lambda^{k-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{k-1}R_{t+k}+\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})\right] \\\\ &=R_{t+1}+\gamma\lambda R_{t+2}+\dots+\gamma^{k-1}\lambda^{k-1}R_{t+k} \\\\ &\hspace{1cm}+(1-\lambda)\left[\sum_{n=1}^{k-1}\lambda^{n-1}\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right]+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1}) \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\left[R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1})\right] \\\\ &\hspace{1cm}+\left[\lambda\gamma R_{t+2}+\lambda\gamma^2\hat{v}(S_{t+2},\mathbf{w}\_{t+1})-\lambda\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)\right]+\dots \\\\ &\hspace{1cm}+\left[\lambda^{k-1}\gamma^{k-1}R_{t+k}+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})-\lambda^{k-1}\gamma^{k-1}\hat{v}(S_{t+k-1},\mathbf{w}\_{t+k-2})\right] \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\sum_{i=t}^{t+k-1}(\gamma\lambda)^{i-t}\delta_i',\tag{2}\label{2}
\end{align}
with
\begin{equation}
\delta_t'\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1}),
\end{equation}
where in the third step of the derivation, we use the identity
\begin{equation}
(1-\lambda)(1+\lambda+\dots+\lambda^{k-2})=1-\lambda^{k-1}
\end{equation}
From \eqref{2}, we can see that the $k$-step $\lambda$-return can be written as sums of TD errors if the value function is held constant, which allows us to implement the TTD($\lambda$) algorithm efficiently.

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
\mathbf{w}\_{t+1}^h\doteq\mathbf{w}\_t^h+\alpha\left[G_{t:h}^\lambda-\hat{v}(S_t,\mathbf{w}\_t^h)\right]\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t^h),\hspace{1cm}0\leq t\lt h\leq T,\tag{3}\label{3}
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

We begin by rewriting \eqref{3}, as
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
\mathbf{w}\_t=\mathbf{w}\_t^t=\mathbf{A}\_0^{t-1}\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i G_{i:t}^\lambda\tag{4}\label{4}
\end{equation}
Using \eqref{2}, we have:
\begin{align}
G_{i:t+1}^\lambda-G_{i:t}^\lambda&=\mathbf{w}\_i^\intercal\mathbf{x}\_i+\sum_{j=1}^{t}(\gamma\lambda)^{j-i}\delta_j'-\left(\mathbf{w}\_i^\intercal\mathbf{x}\_i+\sum_{j=1}^{t-1}(\gamma\lambda)^{j-i}\delta_j'\right) \\\\ &=(\gamma\lambda)^{t-i}\delta_t'\tag{5}\label{5}
\end{align}
with the TD error, $\delta_t'$ is defined as earlier:
\begin{equation}
\delta_t'\doteq R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\tag{6}\label{6}
\end{equation}
Using \eqref{4}, \eqref{5} and \eqref{6}, we have:
\begin{align}
\mathbf{w}\_{t+1}&=\mathbf{A}\_0^t\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t+1}^\lambda \\\\ &=\mathbf{A}\_0^t\mathbf{w}\_{init}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t+1}^\lambda+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\mathbf{A}\_0^t\mathbf{w}\_0+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i G_{i:t}^\lambda+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\left(\mathbf{A}\_0^{t-1}\mathbf{w}\_0+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i G_{t:t+1}^\lambda\right) \\\\ &\hspace{1cm}+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i\left(G_{i:t+1}^\lambda-G_{i:t}^\lambda\right)+\alpha\mathbf{x}\_t G_{t:t+1}^\lambda \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_t\mathbf{x}\_t\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t' \\\\ &\hspace{1cm}+\alpha\mathbf{x}\_t\left(R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t+\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t-\mathbf{w}\_t^\intercal\mathbf{x}\_t\right) \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'+\alpha\mathbf{x}\_t\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_t(\gamma\lambda)^{t-i}\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\delta_t'-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\left(\delta_t+\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)-\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\mathbf{x}\_t \\\\ &=\mathbf{w}\_t+\alpha\mathbf{z}\_t\delta_t+\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\left(\mathbf{z}\_t-\mathbf{x}\_t\right),\tag{7}\label{7}
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
\mathbf{z}\_t&=\sum_{i=0}^{t}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i} \\\\ &=\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^t\mathbf{x}\_i(\gamma\lambda)^{t-i}+\mathbf{x}\_t \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\gamma\lambda\sum_{i=0}^{t-1}\mathbf{A}\_{i+1}^{t-1}\mathbf{x}\_i(\gamma\lambda)^{t-i-1}+\mathbf{x}\_t \\\\ &=\left(\mathbf{I}-\alpha\mathbf{x}\_t\mathbf{x}\_t^\intercal\right)\gamma\lambda\mathbf{z}\_{t-1}+\mathbf{x}\_t \\\\ &=\gamma\lambda\mathbf{z}\_{t-1}+\left(1-\alpha\gamma\lambda\left(\mathbf{z}\_t^\intercal\mathbf{x}\_t\right)\right)\mathbf{x}\_t\tag{8}\label{8}
\end{align}
Equation \eqref{7} and \eqref{8} form the update of the **true online TD($\lambda$)** algorithm:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\delta_t\mathbf{z}\_t+\alpha\left(\mathbf{w}\_t^\intercal\mathbf{x}\_t-\mathbf{w}\_{t-1}^\intercal\mathbf{x}\_t\right)\left(\mathbf{z}t\_t-\mathbf{x}\_t\right),
\end{equation}
where
\begin{align}
\mathbf{z}\_t&\doteq\gamma\lambda\mathbf{z}\_{t-1}+\left(1-\alpha\gamma\lambda\left(\mathbf{z}\_t^\intercal\mathbf{x}\_t\right)\right)\mathbf{x}\_t,\tag{9}\label{9} \\\\ \delta_t&\doteq R_{t+1}+\gamma\mathbf{w}\_t^\intercal\mathbf{x}\_{t+1}-\mathbf{w}\_t^\intercal\mathbf{x}\_t
\end{align}
Pseudocode of the algorithm is given below.
<figure>
	<img src="/assets/images/2022-08-08/true-onl-td-lambda.png" alt="True Online TD(lambda)" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

The eligible trace \eqref{9} is called **dutch trace** to distinguish it from the trace \eqref{1} of TD($\lambda$), which is called **accumulating trace**. 

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
G_{t:t+n}\doteq\ R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{q}(S_{t+n},A_{t+n},\mathbf{w}\_{t+n-1}),\hspace{1cm}t+n\lt T
\end{equation}
with $G_{t:t+n}\doteq G_t$ if $t+n\geq T$. With this definition of the return, the action-value form of offline $\lambda$-return can de defined as:
\begin{equation}
\mathbf{w}\_{t+1}\doteq\mathbf{w}\_t+\alpha\left[G_t^\lambda-\hat{q}(S_t,A_t,\mathbf{w}\_t)\right]\nabla_\mathbf{w}\hat{q}(S_t,A_t,\mathbf{w}\_t),\hspace{1cm}t=0,\dots,T-1
\end{equation}
where $G_t^\lambda\doteq G_{t:\infty}^\lambda$.

<figure>
	<img src="/assets/images/2022-08-08/sarsa-lambda-backup.png" alt="Backup diagram of Sarsa(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 450px; height: 390px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The backup diagram of Sarsa($\lambda$)</figcaption>
</figure>

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