---
title: "Policy Gradient"
date:   2022-10-06 15:26:00 +0700
tags: [deep-reinforcement-learning, policy-gradient, my-rl]
math: true
eqn-number: true
---
> Notes on Policy gradient methods.
<!--more-->

## Preliminaries
Consider an finite-horizon undiscounted Markov Decision Process (MDP), which is a tuple of $(\mathcal{S},\mathcal{A},P,r,\rho_0,H)$ where
- $\mathcal{S}$ is the state space.
- $\mathcal{A}$ is the action space.
- $P:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the transition probability, i.e. $P(s'\vert s,a)$ denotes the probability of being at state $s'$ by taking action $a$ from state $s$.
- $r:\mathcal{\mathcal{S}\times\mathcal{A}\times\mathcal{S}}\to\mathbb{R}$ is the reward function.
- $\rho_0$ is the distribution of the start state $s_0$.
- $H$ is the horizon time.

To start an episode, the agent is given an initial state $s_0$, which is sampled from $\rho_0$, i.e. $s_0\sim\rho_0$. At each time step $t$, from state $s_t$, until reaching the terminal state, the agent takes action $a_t$, according to a policy $\pi$, where $\pi:\mathcal{S}\times\mathcal{A}\to[0,1]$, which lets agent end up at state $s_{t+1}$ due to the dynamics $P$, and is given a corresponding reward $r_t=r(s_t,a_t,s_{t+1})$. The process gives rise to a sequence, called a **trajectory**, defined by:
\begin{equation}
\tau=(s_0,a_0,s_1,a_1,s_2,a_2\ldots)
\end{equation}
For a policy $\pi$, let $V_\pi:\mathcal{S}\to\mathbb{R}$ denote the state value function, $Q_\pi:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ represent the state-action value function and let $A_\pi:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ be the advantage function:
\begin{align}
V_\pi(s_t)&\doteq\mathbb{E}\_{s_{t+1:H-1},a_{t:H-1}}\left[\sum_{k=0}^{H-1}r_{t+k}\right] \\\\ Q_\pi(s_t,a_t)&\doteq\mathbb{E}\_{s_{t+1:H-1},a_{t+1:H-1}}\left[\sum_{k=0}^{H-1}r_{t+k}\right] \\\\ A_\pi(s_t,a_t)&\doteq Q_\pi(s_t,a_t)-V_\pi(s_t),
\end{align}
where the expectation notation $\mathbb{E}\_{s_{t+1:H-1},a_{t:H-1}}$ denotes that the expected value is computed by integrated over $s_{t+1}\sim P(s_{t+1}\vert s_t,a_t),a_t\sim\pi(a_t\vert s_t)$.

As in **DQN**, here we will be working with a policy $\pi_\theta$ parameterized by a vector $\theta$.  Let $R(\tau)\doteq\sum_{t=0}^{H-1}r_t$ denote the return, or the total reward along trajectory $\tau$. Our goal is to maximize the expected return:
\begin{equation}
\eta(\pi_\theta)\doteq\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\big[R(\tau)\big]=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}r_t\right]\label{eq:pre.1}
\end{equation}

## (Vanilla) Policy Gradient{#vanilla-pg}
In **(vanilla) policy gradient** method, we are trying to optimize the expected total reward \eqref{eq:pre.1} by repeatedly estimating the gradient
\begin{equation}
\nabla_\theta\eta(\pi_\theta)=\nabla_\theta\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\big[R(\tau)\big]\label{eq:vpg.1}
\end{equation}
To continue our derivation, we will be using the probability of a trajectory $\tau\sim\pi_\theta$, computed by
\begin{equation}
P(\tau;\theta)=\rho_0(s_0)\prod_{t=0}^{H-1}P(s_{t+1}\vert s_t,a_t)\pi_\theta(a_t\vert s_t),
\end{equation}
Given this definition, \eqref{eq:vpg.1} can be written in a form that does not require a dynamics model:
\begin{align}
\hspace{-0.8cm}\nabla_\theta\eta(\pi_\theta)&=\nabla_\theta\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\big[R(\tau)\big] \\\\ &=\nabla_\theta\sum_\tau P(\tau;\theta)R(\tau) \\\\ &=\sum_\tau\frac{P(\tau;\theta)}{P(\tau;\theta)}\nabla_\theta P(\tau;\theta)R(\tau) \\\\ &=\sum_\tau P(\theta;\tau)\nabla_\theta\log P(\tau;\theta)R(\tau) \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\big[\nabla_\theta\log P(\tau;\theta)R(\tau)\big] \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\nabla_\theta\left(\sum_{t=0}^{H-1}\log\rho_0(s_0)+\log P(s_{t+1}\vert s_t,a_t)+\log\pi_\theta(a_t\vert s_t)\right)R(\tau)\right] \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)R(\tau)\right]
\end{align}
Since the gradient is now an expectation, we can approximate it with the empirical estimate from $m$ sample trajectories, as
\begin{equation}
\nabla_\theta\eta(\pi_\theta)=\frac{1}{m}\sum_{i=1}^{m}\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t^{(i)}\vert s_t^{(i)})R(\tau^{(i)})
\end{equation}

### Variance reduction{#var-red}

#### Reward-to-go{#reward-to-go}
To reduce the variance, we first notice that the total reward along a trajectory $\tau$, $R(\tau)$, can be expressed as sum of total reward from step $t$, called **reward-to-go** from $t$, and total preceding rewards w.r.t $t$, which has the expected value of zero but non-zero variance. In particular, we can simplify the policy gradient to be indepedent to the reward-to-go only, as:
\begin{align}
\nabla_\theta\eta(\pi_\theta)&=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)R(\tau)\right] \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\left(\sum_{k=0}^{t-1}r_k+\sum_{k=t}^{H-1}r_k\right)\right] \\\\ &=\sum_{t=0}^{H-1}\mathbb{E}\_{s_{0:t},a_{0:t}}\left[\nabla_\theta\log\pi_\theta(a_t\vert s_t)\sum_{k=0}^{t-1}r_k\right]\nonumber \\\\ &\hspace{2cm}+\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\sum_{k=t}^{H-1}r_k\right] \\\\ &\overset{\text{(i)}}{=}\sum_{t=0}^{H-1}\left(\mathbb{E}\_{s_{0:t},a_{0:t-1}}\left[\mathbb{E}\_{a_t}\big[\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big]\cdot\sum_{k=0}^{t-1}r_k\right]\right)\nonumber \\\\ &\hspace{2cm}+\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\sum_{k=t}^{H-1}r_k\right]\label{eq:vr.1} \\\\ &\overset{\text{(ii)}}{=}\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\sum_{k=t}^{H-1}r_k\right] \\\\ &\overset{\text{(iii)}}{=}\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\hat{r}\_t\right],\label{eq:vr.2}
\end{align}
where
- The (i) step is due to that the total past reward is independent of the current action $t$.
- In the (ii) step, we have used
\begin{align}
\mathbb{E}\_{a_t}\big[\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big]&=\sum_{a_t}\pi_\theta(a_t\vert s_t)\nabla_\theta\log\pi_\theta(a_t\vert s_t) \\\\ &=\sum_{a_t}\nabla_\theta 1=\mathbf{0}\label{eq:vr.3}
\end{align}
- In the (iii) step, the $\hat{r}\_t\doteq\sum_{k=t}^{H-1}r_k$ is referred as the **reward-to-go** from step $t$.

#### Baseline
It is worth remarking that we can furtherly reduce the variance of the estimator by adding an baseline, denoted $b$, as
\begin{equation}
\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\left(\hat{r}\_t-b_t(s_{0:t},a_{0:t-1})\right)\right]\label{eq:vrb.1}
\end{equation}

##### Unbiased estimator{#unbiased}
First we will prove that the estimator with baseline \eqref{eq:vrb.1} is still an unbiased with \eqref{eq:vr.2}. Specifically
\begin{align}
&\hspace{-1cm}\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\left(\hat{r}\_t-b_t(s_{0:t},a_{0:t-1})\right)\right]\nonumber \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\hat{r}\_t\right]\nonumber \\\\ &\hspace{0.5cm}-\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)b_t(s_{0:t},a_{0:t-1})\right],
\end{align}
which makes our claim follow if the latter expectation of the RHS is zero. In fact, using the logic as in \eqref{eq:vr.1} and the result \eqref{eq:vr.3}, we have
\begin{align}
&\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)b_t(s_{0:t},a_{0:t-1})\right]\nonumber \\\\ &=\sum_{t=0}^{H-1}\mathbb{E}\_{s_{0:t},a_{0:t-1}}\Big[\mathbb{E}\_{a_t}\big[\nabla_\theta\log\pi_\theta(a_t\vert s_t)b_t(s_{0:t},a_{0:t-1})\big]\Big] \\\\ &=\sum_{t=0}^{H-1}\mathbb{E}\_{s_{0:t},a_{0:t-1}}\Big[\mathbb{E}\_{a_t}\big[\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big]\cdot b_t(s_{0:t},a_{0:t-1})\Big] \\\\ &=\sum_{t=0}^{H-1}\mathbb{E}\_{s_{0:t},a_{0:t-1}}\Big[\mathbf{0}\cdot b_t(s_{0:t},a_{0:t-1})\Big]=\mathbf{0}
\end{align}

##### How can a baseline reduce variance?{#how-baseline-red}
By definition of the variance of a r.v $X$
\begin{equation}
\text{Var}(X)=\mathbb{E}\big[X^2\big]-\mathbb{E}\big[X\big]^2,
\end{equation}
combined with the claim that \eqref{eq:vrb.1} is an unbiased estimator of the policy gradient $\nabla_\theta\eta(\pi_\theta)$, and letting $b_t\doteq b_t(s_{0:t},a_{0:t-1})$ to simplify the notation, we have 
\begin{align}
\text{Var}&=\text{Var}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big(\hat{r}\_t-b_t\big)\right] \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\left(\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big(\hat{r}\_t-b_t\big)\right)^2\right]\nonumber \\\\ &\hspace{1.5cm}-\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big(\hat{r}\_t-b_t\big)\right]^2 \\\\ &=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\left[\left(\sum_{t=0}^{H-1}\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big(\hat{r}\_t-b_t\big)\right)^2\right]-\big(\nabla_\theta\eta(\pi_\theta)\big)^2,
\end{align}
which suggests us that for each $t$, by finding $b_t$ that minimizes the former expectation, we can also reduce the variance $\text{Var}$.

Differentiating the variance w.r.t $b_t$ gives us
\begin{equation}
\frac{\partial\text{Var}}{\partial b_t}=\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\Big[\big(\nabla_\theta\log\pi_\theta(a_t\vert s_t)\big)^2(2b_t-2\hat{r}\_t)\Big]
\end{equation}
Set the derivative to zero and solve for $b_t$, we obtain the optimal baseline
\begin{equation}
b_t=\frac{\mathbb{E}\_{s_{0:H-1},a_{0:H-1}}\Big[\big(\nabla_\theta\pi_\theta(a_t\vert s_t)\big)^2\hat{r}\_t\Big]}{\mathbb{E}\_{a_t}\Big[\big(\nabla_\theta\pi_\theta(a_t\vert s_t)\big)^2\Big]}
\end{equation}

##### Types of baseline{#baseline-types}

#### Discount factor

## Preferences
[1] John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, Pieter Abbeel. [High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438). ICLR 2016.

## Footnotes