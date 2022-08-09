---
layout: post
title:  "Eligible Traces"
date:   2022-08-8 14:11:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning td-learning n-step-td my-rl
description: Eligible Traces
comments: true
---
> Beside [$n$-step TD]({% post_url 2022-07-10-func-approx %}#n-step-td) methods, there is another mechanism called **Eligible traces** that unify TD and Monte Carlo. Setting $\lambda$ in TD($\lambda$) from $0$ to $1$, we end up with a spectrum ranging from TD methods, when $\lambda=0$ to Monte Carlo methods with $\lambda=1$.
<!-- excerpt-end -->

- [The \\(\lambda\\)-return](#lambda-return)
	- [Offline \\(\lambda\\)-return](#off-lambda-return)
- [TD(\\(\lambda\\))](#td-lambda)
- [Truncated TD Methods](#truncated-td)
- [Online \\(\lambda\\)-return](#onl-lambda-return)
- [True Online TD(\\(\lambda\\))](#true-onl-td-lambda)
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
\mathbf{z}\_{-1}&\doteq\mathbf{0} \\\\ \mathbf{z}\_t&\doteq\gamma\lambda\mathbf{z}\_t+\nabla_\mathbf{w}\hat{v}(S_t,\mathbf{w}\_t),\hspace{1cm}0\leq t\lt T
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
G_{t:t+k}^\lambda&=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}G_{t:t+n}+\lambda^{k-1}G_{t:t+k} \\\\ &=(1-\lambda)\sum_{n=1}^{k-1}\lambda^{n-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{n-1}R_{t+n}+\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right] \\\\ &\hspace{1cm}+\lambda^{k-1}\left[R_{t+1}+\gamma R_{t+2}+\dots+\gamma^{k-1}R_{t+k}+\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})\right] \\\\ &=R_{t+1}+\gamma\lambda R_{t+2}+\dots+\gamma^{k-1}\lambda^{k-1}R_{t+k} \\\\ &\hspace{1cm}+(1-\lambda)\left[\sum_{n=1}^{k-1}\lambda^{n-1}\gamma^n\hat{v}(S_{t+n},\mathbf{w}\_{t+n-1})\right]+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1}) \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\left[R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1})\right] \\\\ &\hspace{1cm}+\left[\lambda\gamma R_{t+2}+\lambda\gamma^2\hat{v}(S_{t+2},\mathbf{w}\_{t+1})-\lambda\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)\right]+\dots \\\\ &\hspace{1cm}+\left[\lambda^{k-1}\gamma^{k-1}R_{t+k}+\lambda^{k-1}\gamma^k\hat{v}(S_{t+k},\mathbf{w}\_{t+k-1})-\lambda^{k-1}\gamma^{k-1}\hat{v}(S_{t+k-1},\mathbf{w}\_{t+k-2})\right] \\\\ &=\hat{v}(S_t,\mathbf{w}\_{t-1})+\sum_{i=t}^{t+k-1}(\gamma\lambda)^{i-t}\delta_i',
\end{align}
with
\begin{equation}
\delta_t'\doteq R_{t+1}+\gamma\hat{v}(S_{t+1},\mathbf{w}\_t)-\hat{v}(S_t,\mathbf{w}\_{t-1}),
\end{equation}
where in the third step of the derivation, we use the identity
\begin{equation}
(1-\lambda)(1+\lambda+\dots+\lambda^{k-2})=1-\lambda^{k-1}
\end{equation}
From the derivation, we can see that the $k$-step $\lambda$-return can be written as sums of TD errors if the value function is held constant, which allows us to implement the TTD($\lambda$) algorithm efficiently.

<figure>
	<img src="/assets/images/2022-08-08/ttd-lambda-backup.png" alt="Backup diagram of truncated TD(lambda)" style="display: block; margin-left: auto; margin-right: auto; width: 500px; height: 370px"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: The backup diagram of truncated TD($\lambda$)</figcaption>
</figure>

## Online $\lambda$-return
{: $onl-lambda-return}


\begin{equation}

\end{equation}

## Sarsa($\lambda$)

## References
{: #references}
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] Precup, Doina; Sutton, Richard S.; and Singh, Satinder. [Eligibility Traces for Off-Policy Policy Evaluation](https://scholarworks.umass.edu/cs_faculty_pubs/80) (2000). ICML '00 Proceedings of the Seventeenth International Conference on Machine Learning. 80. 

[3] Deepmind x UCL. [Reinforcement Learning Lecture Series 2021](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021). 

[4] Shangtong Zhang. [Reinforcement Learning: An Introduction implementation](https://github.com/ShangtongZhang/reinforcement-learning-an-introduction). 

## Footnotes
{: #footnotes}