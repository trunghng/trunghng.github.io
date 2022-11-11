---
layout: post
title:  "Deep Q-learning"
date:   2022-05-25 15:26:00 +0700
tags: reinforcement-learning deep-learning function-approximation q-learning my-rl
description: Deep Q-learning and variants
comments: true
eqn-number: true
---
> 

<!-- excerpt-end -->
- [Q-value iteration](#q-value-iter)
- [Q-learning](#q-learning)
- [Neural networks with Q-learning](#nn-q-learning)
	- [Experience replay](#exp-replay)
	- [Target network](#target-net)
	- [RMSProp](#rmsprop)
- [References](#references)
- [Footnotes](#footnotes)

## Q-value iteration
{: #q-value-iter}
Recall that in the post [**Markov Decision Processes, Bellman equations**]({% post_url 2021-06-27-mdp-bellman-eqn %}), we have defined the **state-value function** for a policy $\pi$ to measure how good the state $s$ is, given as
\begin{equation}
V_\pi(s)=\sum_{a}\pi(a\vert s)\sum_{s'}P(s'\vert s,a)\big[R(s,a,s')+\gamma V_\pi(s')\big]
\end{equation}
From the definition of $V_\pi(s)$, we have continued to define the Bellman equation for the optimal value at state $s$, denoted $V^\*(s)$:
\begin{equation}
V^\*(s)=\max_{a}\sum_{s'}P(s'\vert s,a)\big[R(s,a,s')+\gamma V^\*(s')\big],\label{eq:qvi.1}
\end{equation}
which characterizes the optimal value of state $s$ in terms of the optimal values of successor state $s'$.

Then, with [**Dynamic programming**]({% post_url 2021-07-25-dp-in-mdp %}), we can solve \eqref{eq:qvi.1} by an iterative method, called [**value iteration**]({% post_url 2021-07-25-dp-in-mdp %}#value-iteration), given as
\begin{equation}
V_{k+1}(s)=\max_{a}\sum_{s'}P(s'\vert s,a)\big[R(s,a,s')+\gamma V_k(s')\big]\hspace{1cm}\forall s\in\mathcal{S}
\end{equation}
For an arbitrary initial $V_0(s)$, the iteration, or the sequence $\\{V_k\\}$, will eventually converge to the optimal value function $V^\*(s)$. This can be shown by applying the [**Banach's fixed point theorem**]({% post_url 2021-07-10-optimal-policy-existence %}), the one we have also used to prove the existence of the optimal policy, to prove that the iteration from $V_k(s)$ to $V_{k+1}(s)$ is a contraction mapping.

Details for value iteration method can be seen in the following pseudocode.
<figure>
	<img src="/assets/images/2022-05-25/value-iteration.png" alt="value iteration pseudocode" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption></figcaption>
</figure>

Remember that along with the state-value function $V_\pi(s)$, we have also defined the **action-value function**, or **Q-values** for a policy $\pi$, denoted $Q$, given by
\begin{align}
Q_\pi(s,a)&=\sum_{s'}P(s'\vert s,a)\left[R(s,a,s')+\gamma\sum_{a'}\pi(a'\vert s')Q_\pi(s',a')\right] \\\\ &=\sum_{s'}P(s'\vert s,a)\big[R(s,a,s')+\gamma V_\pi(s')\big]
\end{align}
which measures how good it is to be in state $s$ and take action $a$.

Analogously, we also have the Bellman equation for the optimal action-value function, given as
\begin{align}
Q^\*(s,a)&=\sum_{s'}P(s'\vert s,a)\left[R(s,a,s')+\gamma\max_{a'}Q^\*(s',a')\right]\label{eq:qvi.2} \\\\ &=\sum_{s'}P(s'\vert s,a)\big[R(s,a,s')+\gamma V^\*(s')\big]\label{eq:qvi.3}
\end{align}
The optimal value $Q^\*(s,a)$ gives us the expected discounted cumulative reward for executing action $a$ at state $s$ and following the optimal policy, $\pi^\*$, thereafter.  

Equation \eqref{eq:qvi.3} allows us to write
\begin{equation}
V^\*(s)=\max_a Q^\*(s,a)
\end{equation}
Hence, analogy to the state-value function, we can also apply Dynamic programming to develop an iterative method in order to solve \eqref{eq:qvi.2}, called **Q-value iteration**. The method is given by the update rule
\begin{equation}
Q_{k+1}(s,a)=\sum_{s'}P(s'\vert s,a)\left[R(s,a,s')+\gamma\max_{a'}Q_k(s',a')\right]\label{eq:qvi.4}
\end{equation}
This iteration, given an initial value $Q_0(s,a)$, eventually will also converge to the optimal Q-values $Q^\*(s,a)$ due to the relationship between $V$ and $Q$ as defined above. Pseudocode for Q-value iteration is given below.
<figure>
	<img src="/assets/images/2022-05-25/q-value-iteration.png" alt="value iteration pseudocode" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption></figcaption>
</figure>

## Q-learning
{: #q-learning}
The update formula \eqref{eq:qvi.4} can be rewritten as an expected update
\begin{equation}
Q_{k+1}(s,a)=\mathbb{E}\_{s'\sim P(s'\vert s,a)}\left[R(s,a,s')+\gamma\max_{a'}Q_k(s',a')\right]\label{eq:ql.1}
\end{equation}
It is noticeable that the above update rule requires the transition model $P(s'\vert s,a)$. And since sample mean is an unbiased estimator of the population mean, or in other words, the expectation in \eqref{eq:ql.1} can be approximated by sampling, as
<ul id='number-list'>
	<li>
		At a state, taking (sampling) action $a$ (e.g. due to an $\varepsilon$-greedy policy), we get the next state:
		\begin{equation}
		s'\sim P(s'\vert s,a)
		\end{equation}
	</li>
	<li>Consider the old estimate $Q_k(s,a)$.</li>
	<li>
		Consider the new sample estimate (target):
		\begin{equation}
		Q_\text{target}=R(s,a,s')+\gamma\max_{a'}Q_k(s',a')
		\end{equation}
	</li>
	<li>
		Append the new estimate into a running average to iteratively update Q-values:
		\begin{align}
		Q_{k+1}(s,a)&=(1-\alpha)Q_k(s,a)+\alpha Q_\text{target} \\ &=(1-\alpha)Q_k(s,a)+\alpha\left[R(s,a,s')+\gamma\max_{a'}Q_k(s',a')\right]
		\end{align}
	</li>
</ul>

This update rule is in form of a **stochastic process**, and thus, can be [proved](#q-learning-td-convergence) to be converged to the optimal $Q^\*$, under the [stochastic approximation conditions]({% post_url 2022-01-31-td-learning %}#stochastic-approx-condition) for the learning rate $\alpha$.
\begin{equation}
\sum_{t=1}^{\infty}\alpha_t(s,a)=\infty\hspace{1cm}\text{and}\hspace{1cm}\sum_{t=1}^{\infty}\alpha_t^2(s,a)<\infty,\label{eq:ql.2}
\end{equation}
for all $(s,a)\in\mathcal{S}\times\mathcal{A}$.

The method is so called **Q-learning**, with pseudocode given below.

## Neural networks with Q-learning
{: #nn-q-learning}
As a tabular method, Q-learning might work with a discrete space. However, for continuous environments, the exact solution might never be found in a given short time. To overcome this, we have been instead trying to find an [approximated solution]({% post_url 2022-02-11-func-approx %}).  

In particular, we have tried to find an approximated action-value function $Q_\boldsymbol{\theta}(s,a)$, parameterized by a vector $\boldsymbol{\theta}$, of the action-value function $Q(s,a)$, as
\begin{equation}
Q_\boldsymbol{\theta}(s,a)
\end{equation}
Then, we could have applied stochastic gradient descent (SGD) to repeatedly update $\boldsymbol{\theta}$ so as to minimize the loss function
\begin{equation}
L(\boldsymbol{\theta})=\mathbb{E}\_{s,a\sim\mu(\cdot)}\Big[\big(Q(s,a)-Q_\boldsymbol{\theta}(s,a)\big)^2\Big]
\end{equation}
The resulting SGD update had the form
\begin{align}
\boldsymbol{\theta}\_{k+1}&=\boldsymbol{\theta}\_k-\frac{1}{2}\alpha\nabla_\boldsymbol{\theta}\big[Q(s_k,a_k)-Q_\boldsymbol{\theta}(s_k,a_k)\big]^2 \\\\ &=\boldsymbol{\theta}\_k+\alpha\big[Q(s_k,a_k)-Q_\boldsymbol{\theta}(s_k,a_k)\big]\nabla_\boldsymbol{\theta}Q_\boldsymbol{\theta}(s_k,a_k)\label{eq:nql.1}
\end{align}
However, we could not perform the exact update \eqref{eq:nql.1} since the true value $Q(s_k,a_k)$ was unknown. Fortunately, we could instead approximate it by, says $U_k$, which let us rewrite the SGD update as
\begin{equation}
\boldsymbol{\theta}\_{k+1}=\boldsymbol{\theta}\_k+\alpha\big[U_k-Q_\boldsymbol{\theta}(s_k,a_k)\big]\nabla_\boldsymbol{\theta}Q_\boldsymbol{\theta}(s_k,a_k)\label{eq:nql.2}
\end{equation}
If $U_k$ is an unbiased estimate for $Q(s_k,a_k)$, i.e. $\mathbb{E}\big[U_k\vert s_k,a_k\big]=Q(s_k,a_k)$, for each $k$, then $\boldsymbol{\theta}$ was guaranteed to converge to the local minimum under the stochastic approximation condition for decreasing the learning rate $\alpha$ as given in \eqref{eq:ql.2}.

Recall that, we have applied [linear methods]({% post_url 2022-02-11-func-approx %}#lin-func-approx) as our function approximators:
\begin{equation}
Q_\boldsymbol{\theta}(s,a)=\boldsymbol{\theta}^\text{T}\mathbf{f}(s,a),
\end{equation}
where $\mathbf{f}(s,a)$ represents the **feature vector**, (or **basis functions**) of the state-action pair $(s,a)$.

Linear function approximation allowed us to rewrite \eqref{eq:nql.2} in a simplified form
\begin{equation}
\boldsymbol{\theta}\_{k+1}=\boldsymbol{\theta}\_k+\alpha\big[U_k-Q_\boldsymbol{\theta}(s_k,a_k)\big]\mathbf{f}(s_k,a_k)\label{eq:nql.3}
\end{equation}
On the other hands, we know that a neural network with a particular settings for hidden layers and activation functions can approximate [any]({% post_url 2022-09-02-neural-nets %}#unv-approx) continuous functions on a compact subsets of $\mathbb{R}^n$, so how about using it with the Q-learning algorithm?

Specifically, we will be using neural network with weight $\boldsymbol{\theta}$ as a function approximator for Q-learning update. The network is referred as **Q-network**. The Q-network can be trained by minimizing a sequence of loss function $L_i(\boldsymbol{\theta}\_i)$ that changes at each iteration $i$:
\begin{equation}
L_i(\boldsymbol{\theta}\_i)=\mathbb{E}\_{s,a\sim\rho(\cdot)}\Big[\big(y_i-Q_{\boldsymbol{\theta}\_i}(s,a)\big)^2\Big],
\end{equation}
where
\begin{equation}
y_i=\mathbb{E}\_{s'\sim\mathcal{E}}\left[R(s,a,s')+\gamma\max_{a'}Q_{\boldsymbol{\theta}\_{i-1}}(s',a')\vert s,a\right]
\end{equation}
is the target in iteration $i$, as the target $U_k$ for iteration $k$ in \eqref{eq:nql.3}; and where $\rho(s,a)$ is referred as the behavior policy.

### Experience replay
{: #exp-replay}
Along with Q-network, the authors of deep-Q learning also introduce a mechanism called **experience replay**, which utilizes data efficiency and at the same time reduces the variance of the updates.

In particular, at each time step $t$, the **experience**, $e_t$, defined as
\begin{equation}
e_t=(s_t,a_t,r_t,s_{t+1})
\end{equation}
is added into a set $\mathcal{D}$ of size $N$, which is 


### Target network
{: #target-net}


## References
<span id='q-learning-td-convergence'>[1] Tommi Jaakkola, Michael I. Jordan, Satinder P. Singh. [On the Convergence of Stochastic Iterative Dynamic Programming Algorithms](https://people.eecs.berkeley.edu/~jordan/papers/AIM-1441.ps). A.I. Memo No. 1441, 1993.</span>

[2] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition).

[3] Vlad Mnih, et al. [Playing Atari with Deep Reinforcement Learning](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), 2013.

[4] John N Tsitsiklis and Benjamin Van Roy. [An analysis of temporal-difference learning with function approximation](). Automatic Control, IEEE Transactions on, 42(5):674–690, 1997.

[5] Vlad Mnih, et al. [Human Level Control Through Deep Reinforcement Learning](https://www.deepmind.com/publications/human-level-control-through-deep-reinforcement-learning). Nature, 2015.

[6] Hado van Hasselt. [Double Q-learning](https://proceedings.neurips.cc/paper/2010/file/091d584fced301b442654dd8c23b3fc9-Paper.pdf). NIPS 2010.

[7] Hado van Hasselt, Arthur Guez, David Silver. [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461). AAAI16, 2016.

[8] Pieter Abbeel. [Foundations of Deep RL Series](https://youtube.com/playlist?list=PLwRJQ4m4UJjNymuBM9RdmB3Z9N5-0IlY0), 2021.

## Footnotes
