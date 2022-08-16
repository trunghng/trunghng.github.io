---
layout: post
title:  "Policy Gradient Methods"
date:   2022-08-16 14:00:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning policy-gradient actor-critic function-approximation my-rl
description: Policy Gradient Methods
comments: true
---
> So far in the series, we have been choosing the actions based on the estimated action value function. On the other hand, we can instead learn a **parameterized policy**, $\mathbf{\theta}$, that can select actions without consulting a value function by considering the gradient of some performance measure w.r.t $\mathbf{\theta}$. Such methods are called **policy gradient methods**.
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
We begin by considering episodic case, for which we define the performance measure $J(\mathbf{\theta})$ as the value of the start state of the episode. By assuming without loss of generality that every episode starts in some particular state $s_0$, we have:
\begin{equation}
J(\mathbf{\theta})\doteq v_{\pi_\mathbf{\theta}}(s_0),
\end{equation}
where $v_{\pi_\mathbf{\theta}}$ is the true value function for $\pi_\mathbf{\theta}$, the policy determined by $\mathbf{\theta}$.

### The Policy Gradient Theorem
{: #policy-grad-theorem-ep}
**Theorem**  
The policy gradient theorem for the episodic case establishes that
\begin{equation}
\nabla_\mathbf{\theta}J(\mathbf{\theta})\propto\sum_s\mu(s)\sum_a q_\pi(s,a)\nabla_\mathbf{\theta}\pi(a|s,\mathbf{\theta}),
\end{equation}
where $\pi$ represents the policy corresponding to parameter vector $\mathbf{\theta}$.

**Proof**  
We have that the gradient of the state-value function w.r.t $\mathbf{\theta}$ can be written in terms of the action-value function as:
\begin{align}
\nabla_\mathbf{\theta}v_\pi(s)&=\nabla_\mathbf{\theta}\Big[\sum_a\pi(a|s)q_\pi(s,a)\Big],\hspace{1cm}\forall s\in\mathcal{S} \\\\ &=\sum_a\Big[\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\nabla_\mathbf{\theta}q_\pi(s,a)\Big] \\\\ &=\sum_a\Big[\nabla_\mathbf{\theta}\pi(s|a)q_\pi(a,s)+\pi(a|s)\nabla_\mathbf{\theta}\sum_{s',r}p(s',r|s,a)\big(r+v_\pi(s')\big)\Big] \\\\ &=\sum_a\Big[\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\nabla_\mathbf{\theta}v_\pi(s')\Big] \\\\ &=\sum_a\Big[\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\sum_{a'}\big(\nabla_\mathbf{\theta}\pi(s'|a')q_\pi(s',a') \\\\ &\hspace{2cm}+\pi(a'|s')\sum_{s\'\'}p(s\'\'\vert s',a')\nabla_\mathbf{\theta}v_\pi(s\'\')\big)\Big] \\\\ &=\sum_{x\in\mathcal{S}}\sum_{k=0}^{\infty}P(s\to x,k,\pi)\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a),
\end{align}
After repeated unrolling as in the fifth step, where $P(s\to x,k,\pi)$ is the probability of transitioning from state $s$ to state $x$ in $k$ steps under policy $\pi$. It is then immediate that:
\begin{align}
\nabla_\mathbf{\theta}J(\mathbf{\theta})&=\nabla_\mathbf{\theta}v_\pi(s_0) \\\\ &=\sum_s\Big(\sum_{k=0}^{\infty}P(s_0\to s,k,\pi)\Big)\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_s\eta(s)\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_{s'}\eta(s')\sum_s\frac{\eta(s)}{\sum_{s'}\eta(s')}\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a) \\\\ &=\sum_{s'}\eta(s')\sum_s\mu(s)\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a) \\\\ &\propto\sum_s\mu(s)\sum_a\nabla_\mathbf{\theta}\pi(a|s)q_\pi(s,a),
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

### REINFORCE with Baseline
{: #reinforce-baseline}

### Actor-Critic Methods
{: #actor-critic-methods}

## References
{: #references}
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  



## Footnotes
{: #footnotes}