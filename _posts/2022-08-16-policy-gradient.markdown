---
layout: post
title:  "Policy Gradient Methods"
date:   2022-08-16 14:00:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning policy-gradient actor-critic function-approximation my-rl
description: Policy Gradient Methods
comments: true
---
> So far in the series, we have been choosing the actions based on the estimated action value function. On the other hand, we can instead learn a **parameterized policy**, $\boldsymbol{\theta}$, that can select actions without consulting a value function by considering the gradient of some performance measure w.r.t $\boldsymbol{\theta}$. Such methods are called **policy gradient methods**.
<!-- excerpt-end -->

- [Policy Gradient for Episodic Problems](#policy-grad-ep)
	- [The Policy Gradient Theorem](#policy-grad-theorem-ep)
	- [REINFORCE](#reinforce)
	- [REINFORCE with Baseline](#reinforce-baseline)
	- [Actor-Critic Methods](#actor-critic-methods)
- [Policy Gradient for Continuing Problems](#policy-grad-cont)
	- [The Policy Gradient Theorem](#policy-grad-theorem-cont)
- [References](#references)
- [Footnotes](#footnotes)

## Policy Gradient for Episodic Problems
{: #policy-grad-ep}
We begin by considering episodic case, for which we define the performance measure $J(\boldsymbol{\theta})$ as the value of the start state of the episode. By assuming without loss of generality that every episode starts in some particular state $s_0$, we have:
\begin{equation}
J(\boldsymbol{\theta})\doteq v_{\pi_\boldsymbol{\theta}}(s_0),
\end{equation}
where $v_{\pi_\boldsymbol{\theta}}$ is the true value function for $\pi_\boldsymbol{\theta}$, the policy determined by $\boldsymbol{\theta}$.

### The Policy Gradient Theorem
{: #policy-grad-theorem-ep}
**Theorem 1**  
The policy gradient theorem for the episodic case establishes that
\begin{equation}
\nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\nabla_\boldsymbol{\theta}\pi(a|s,\boldsymbol{\theta}),\tag{1}\label{1}
\end{equation}
where $\pi$ represents the policy corresponding to parameter vector $\boldsymbol{\theta}$.

**Proof**  
We have that the gradient of the state-value function w.r.t $\boldsymbol{\theta}$ can be written in terms of the action-value function as:
\begin{align}
\nabla_\boldsymbol{\theta}v_\pi(s)&=\nabla_\boldsymbol{\theta}\Big[\sum_a\pi(a|s)q_\pi(s,a)\Big],\hspace{1cm}\forall s\in\mathcal{S} \\\\ &=\sum_a\Big[\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\nabla_\boldsymbol{\theta}q_\pi(s,a)\Big] \\\\ &=\sum_a\Big[\nabla_\boldsymbol{\theta}\pi(s|a)q_\pi(a,s)+\pi(a|s)\nabla_\boldsymbol{\theta}\sum_{s',r}p(s',r|s,a)\big(r+v_\pi(s')\big)\Big] \\\\ &=\sum_a\Big[\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\nabla_\boldsymbol{\theta}v_\pi(s')\Big] \\\\ &=\sum_a\Big[\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\sum_{a'}\big(\nabla_\boldsymbol{\theta}\pi(s'|a')q_\pi(s',a') \\\\ &\hspace{2cm}+\pi(a'|s')\sum_{s\'\'}p(s\'\'\vert s',a')\nabla_\boldsymbol{\theta}v_\pi(s\'\')\big)\Big] \\\\ &=\sum_{x\in\mathcal{S}}\sum_{k=0}^{\infty}P(s\to x,k,\pi)\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a),
\end{align}
After repeated unrolling as in the fifth step, where $P(s\to x,k,\pi)$ is the probability of transitioning from state $s$ to state $x$ in $k$ steps under policy $\pi$. It is then immediate that:
\begin{align}
\nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})&=\nabla_\boldsymbol{\theta}v_\pi(s_0) \\\\ &=\sum_s\Big(\sum_{k=0}^{\infty}P(s_0\to s,k,\pi)\Big)\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_s\eta(s)\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_{s'}\eta(s')\sum_s\frac{\eta(s)}{\sum_{s'}\eta(s')}\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_{s'}\eta(s')\sum_s\mu(s)\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a) \\\\ &\propto\sum_s\mu(s)\sum_a\nabla_\boldsymbol{\theta}\pi(a|s)q_\pi(s,a),
\end{align}
where $\eta(s)$ denotes the number of time steps spent, on average, in state $s$ in a single episode:
\begin{equation}
\eta(s)=h(s)+\sum_{\bar{s}}\eta(\bar{s})\sum_a\pi(a|s)p(s|\bar{s},a),\hspace{1cm}\forall s\in\mathcal{S}
\end{equation}
where $h(s)$ denotes the probability that an episode begins in each state $s$; $\bar{s}$ denotes the preceding state of $s$. This leads to the result that we have used in the fifth step:
\begin{equation}
\mu(s)=\frac{\eta(s)}{\sum_{s'}\eta(s')},\hspace{1cm}\forall s\in\mathcal{S}
\end{equation}

### REINFORCE
{: #reinforce}
Notice that in **Theorem 1**, the right-hand side is a sum over states weighted by how often the states occur (distributed by $\mu(s)$) under the target policy $\pi$. Therefore, we can rewrite \eqref{1} as:
\begin{align}
\nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})&\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\nabla_\boldsymbol{\theta}\pi(a|s,\boldsymbol{\theta}) \\\\ &=\mathbb{E}\_\pi\left[\sum_a q_\pi(S_t,a)\nabla_\boldsymbol{\theta}\pi(a|S_t,\boldsymbol{\theta})\right]\tag{2}\label{2}
\end{align}
Using SGD on maximizing $J(\boldsymbol{\theta})$ gives us the update rule:
\begin{equation}
\boldsymbol{\theta}\_{t+1}\doteq\boldsymbol{\theta}\_t+\alpha\sum_a\hat{q}(S_t,a,\mathbf{w})\nabla_\boldsymbol{\theta}\pi(a|S_t,\boldsymbol{\theta}),
\end{equation}
where $\hat{q}$ is some learned approximation to $q_\pi$ with $\mathbf{w}$ denoting the weight vector of its as usual. This algorithm is called **all-actions** method because its update involves all of the actions. 

Continue our derivation in \eqref{2}, we have:
\begin{align}
\nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})&=\mathbb{E}\_\pi\left[\sum_a q_\pi(S_t,a)\nabla_\boldsymbol{\theta}\pi(a|S_t,\boldsymbol{\theta})\right] \\\\ &=\mathbb{E}\_\pi\left[\sum_a\pi(a|S_t,\boldsymbol{\theta})q_\pi(S_t,a)\frac{\nabla_\boldsymbol{\theta}\pi(a|S_t,\boldsymbol{\theta})}{\pi(a|S_t,\boldsymbol{\theta})}\right] \\\\ &=\mathbb{E}\_\pi\left[q_\pi(S_t,A_t)\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta}}\right] \\\\ &=\mathbb{E}\_\pi\left[G_t\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta}}\right],
\end{align}
where $G_t$ is the return as usual; in the third step, we have replaced $a$ by the sample $A_t\sim\pi$; and in the fourth step, we have used the identity
\begin{equation}
\mathbb{E}\_\pi\left[G_t|S_t,A_t\right]=q_\pi(S_t,A_t)
\end{equation}
With this gradient, we have the SGD update for time step $t$, called the **REINFORCE** update, is then:
\begin{equation}
\boldsymbol{\theta}\_{t+1}\doteq\boldsymbol{\theta}\_t+\alpha G_t\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta})}\tag{3}\label{3}
\end{equation}
Pseudocode of the algorithm is given below.
<figure>
	<img src="/assets/images/2022-08-16/reinforce.png" alt="REINFORCE" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

The vector
\begin{equation}
\frac{\nabla_\boldsymbol{\theta}\pi(a|s,\boldsymbol{\theta})}{\pi(a|s,\boldsymbol{\theta})}=\nabla_\boldsymbol{\theta}\ln\pi(a|s,\boldsymbol{\theta})
\end{equation}
in \eqref{3} is called the **eligibility vector**.

### REINFORCE with Baseline
{: #reinforce-baseline}
The policy gradient theorem \eqref{1} can be generalized to include a comparison of the action value to an arbitrary *baseline* $b(s)$:
\begin{equation}
\nabla_\boldsymbol{\theta}J(\boldsymbol{\theta})\propto\sum_s\mu(s)\sum_a\Big(q_\pi(s,a)-b(s)\Big)\nabla_\boldsymbol{\theta}\pi(a|s,\boldsymbol{\theta})\tag{4}\label{4}
\end{equation}
The baseline can be any function, even a r.v, as long as it is independent with $a$. The equation is valid because:
\begin{align}
\sum_a b(s)\nabla_\boldsymbol{\theta}\pi(a|s,\boldsymbol{\theta})&=b(s)\nabla_\boldsymbol{\theta}\sum_a\pi(a|s,\boldsymbol{\theta}) \\\\ &=b(s)\nabla_\boldsymbol{\theta}1=0
\end{align}
Using the derivation steps analogous to REINFORCE, we end up with another version of REINFORCE that includes a general baseline:
\begin{equation}
\boldsymbol{\theta}\_{t+1}\doteq\boldsymbol{\theta}\_t+\alpha\Big(G_t-b(s)\Big)\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta})}\tag{5}\label{5}
\end{equation}
One natural baseline choice is the estimate of the state value, $\hat{v}(S_t,\mathbf{w})$, with $\mathbf{w}\in\mathbb{R}^d$ is the weight vector of its. Using this baseline, we have pseudocode of the generalization with baseline of REINFORCE algorithm \eqref{5} given below.
<figure>
	<img src="/assets/images/2022-08-16/reinforce-baseline.png" alt="REINFORCE with Baseline" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

### Actor-Critic Methods
{: #actor-critic-methods}
In Reinforcement Learning, methods that learn both policy and value function at the same time are called **actor-critic methods**, in which **actor** refers to the learned policy and **critic** is a reference to the learned value function. Although the REINFORCE with Baseline method in the previous section learns both policy and value function, but it is not an actor-critic method. Because its state-value function is used as a baseline, not as a critic, which is used for bootstrapping.

We begin by considering one-step actor-critic methods. One-step actor-critic methods replace the full return, $G_t$, of REINFORCE \eqref{5} with the one-step return, $G_{t:t+1}$:
\begin{align}
\boldsymbol{\theta}\_{t+1}&\doteq\boldsymbol{\theta}\_t+\alpha\Big(G_{t:t+1}-\hat{v}(S_t,\mathbf{w})\Big)\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta})}\tag{6}\label{6} \\\\ &=\boldsymbol{\theta}\_t+\alpha\Big(R_{t+1}+\hat{v}(S_{t+1},\mathbf{w})-\hat{v}(S_t,\mathbf{w})\Big)\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta})} \\\\ &=\boldsymbol{\theta}\_t+\alpha\delta_t\frac{\nabla_\boldsymbol{\theta}\pi(A_t|S_t,\boldsymbol{\theta})}{\pi(A_t|S_t,\boldsymbol{\theta})}
\end{align}
The natural state-value function learning method to pair with this is semi-gradient TD(0), which produces the pseudocode given below.
<figure>
	<img src="/assets/images/2022-08-16/one-step-actor-critic.png" alt="One-step Actor-Critic" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

To generalize the one-step methods to the forward view of $n$-step methods and then to $\lambda$-return, in \eqref{6}, we simply replace the one-step return, $G_{t+1}$, by the $n$-step return, $G_{t:t+n}$, and the $\lambda$-return, $G_t^\lambda$, respectively.

In order to obtain the backward view of the $\lambda$-return algorithm, we use separately eligible traces for the actor and critic, as in the pseudocode given below.
<figure>
	<img src="/assets/images/2022-08-16/actor-critic-eligible-traces.png" alt="Actor-Critic with Eligible Traces" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"></figcaption>
</figure>

## Policy Gradient with Continuing Problems
{: #policy-grad-cont}

### The Policy Gradient Theorem
{: #policy-grad-theorem-cont}
**Theorem 2**  

**Proof**  



## References
{: #references}
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition).  

[2] Deepmind x UCL. [Reinforcement Learning Lecture Series 2021](https://www.deepmind.com/learning-resources/reinforcement-learning-lecture-series-2021). 



## Footnotes
{: #footnotes}