---
title: "Read through: An operator view of policy gradient methods"
date: 2023-12-18T16:18:09+07:00
draft: true
tags: [reinforcement-learning, policy-gradient, my-rl]
math: true
eqn-number: true
hideSummary: true
---

### Preliminaries
Let us consider an infinite-horizon discounted MDP defined by the tuple $(\mathcal{S},\mathcal{A},p,r,d_0,\gamma)$, where
- $\mathcal{S}$ is the set of states.
- $\mathcal{A}$ is the set of actions.
- $p:\mathcal{S}\times\mathcal{A}\mapsto\Delta(\mathcal{S})$, where $\Delta(\cdot)$ denotes the probability simplex, is the transition probability distribution.
- $r:\mathcal{S}\times\mathcal{A}\mapsto[0,R_\text{max}]$ is the reward function.
- $d_0$ is the initial distribution of states.
- $\gamma\in[0,1)$ is the discount factor.

#### Trajectory formulation
The goal of agent is to learn a policy $\pi:\mathcal{S}\mapsto\Delta(\mathcal{A})$ that maximizes the expected discounted cumulative rewards
\begin{equation}
J(\pi)=\mathbb{E}\_{s_0,a_0,s_1,\ldots}\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\right]=\mathbb{E}\_\tau\big[R(\tau)\big],
\end{equation}
where
- $s_0\sim d_0,a_t\sim\pi(a_t\vert s_t)$ and $s_{t+1}\sim p(s_{t+1}\vert s_t,a_t)$.
- $\tau=(s_0,a_0,s_1,\ldots)$ denotes a trajectory and $R(\tau)$ denotes the return of that trajectory.

Thus, the optimal policy, denoted $\pi^\*$, that maximizes $J$ can be defined as
\begin{equation}
\pi^\*=\underset{\pi}{\text{argmax}}\hspace{0.1cm}J(\pi)=\underset{\pi}{\text{argmax}}\int_\tau R(\tau)\pi(\tau)\hspace{0.1cm}d\tau
\end{equation}
Assuming that the policy is parameterized by $\theta$, which makes the problem change into
\begin{equation}
\theta^\*=\underset{\theta}{\text{argmax}}\hspace{0.1cm}J(\pi_\theta)=\underset{\theta}{\text{argmax}}\int_\tau R(\tau)\pi_\theta(\tau)\hspace{0.1cm}d\tau
\end{equation}
By the PG theorem, 

#### State-action formulation
Recall that the state and state-action value functions are defined as
\begin{align}
V^\pi(s_t)&=\mathbb{E}\_{a_t,s_{t+1},\ldots}\left[\sum_{k=0}^{\infty}\gamma^k r(s_{t+k},a_{t+k})\right], \\\\ Q^\pi(s_t,a_t)&=\mathbb{E}\_{s_{t+1},a_{t+1},\ldots}\left[\sum_{k=0}^{\infty}\gamma^k r(s_{t+k},a_{t+k})\right],
\end{align}
where $a_t\sim\pi(a_t\vert s_t)$ and $s_{t+1}\sim p(s_{t+1}\vert s_t,a_t)$, for $t\geq 0$.


### References
[1] Dibya Ghosh, Marlos C. Machado, Nicolas Le Roux. [An operator view of policy gradient methods](https://arxiv.org/abs/2006.11266).

### Footnotes

