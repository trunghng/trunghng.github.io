---
title: "Maximum Entropy Reinforcement Learning via Soft Q-learning & Soft Actor-Critic"
date: 2022-12-26T13:46:09+07:00
tags: [reinforcement-learning, policy-gradient, actor-critic, my-rl]
math: true
eqn-number: true
---
> Notes on Maximum Entropy Reinforcement Learning via SQL & SAC.
<!--more-->

## Entropy-Regularized Reinforcement Learning{#maxent-rl}
Consider an infinite-horizon Markov Decision Process (MDP), defined as a tuple $(\mathcal{S},\mathcal{A},p,r,\gamma)$, where
- $\mathcal{S}$ is the **state space**.
- $\mathcal{A}$ is the **action space**.
- $p:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the **transition probability distribution**, i.e. $p(s,a,s')=p(s'\vert s,a)$ denotes the probability of transitioning to state $s'$ when taking action $a$ from state $s$.
- $r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ is the **reward function**, and let us denote $r_t\doteq r(s_t,a_t)$ for simplicity.
- $\gamma\in(0,1)$ is the **discount factor**.

To consider entropy regularization setting, we first recall some basics in standard RL, then extend them into the maximum entropy framework.

### Objective function{#objective-func} 
Regularly, with discounted infinite-horizon MDP, our objective is to maximize the expected cumulative rewards
\begin{equation}
J_\text{std}(\pi)=\mathbb{E}\_\pi\left[\sum_{t=0}^{\infty}\gamma^t r_t\right]
\end{equation}
In **Entropy-Regularized RL*d*, or **Maximum Entropy RL** framework, we wish to maximize the expected **entropy-augmented return**
\begin{equation}
J_\text{MaxEnt}(\pi)=\mathbb{E}\_\pi\left[\sum_{t=0}^{\infty}\gamma^t\Big(r_t+\alpha H\big(\pi(\cdot\vert s_t)\big)\Big)\right],\label{eq:mr.1}
\end{equation}
where $\alpha\in[0,1]$ is the hyperparameter determines the relative importance of the entropy with the rewards. The corresponding optimal policy of the maximum entropy objective is thus given by
\begin{align}
\pi_\text{MaxEnt}^\*&=\underset{\pi}{\text{argmax}}J_\text{MaxEnt}(\pi) \\\\ &=\underset{\pi}{\text{argmax}}\mathbb{E}\_\pi\left[\sum_{t=0}^{\infty}\gamma^t\Big(r_t+\alpha H\big(\pi(\cdot\vert s_t)\big)\Big)\right]
\end{align}

### Value functions{#val-funcs}
In standard RL, value functions are referred to be the expected returns. Thus, the state-value function and state-action value function in maximum entropy framework could be defined as the expected entropy-augmented returns. Specifically, by adding an entropy term, the state-value function is given by
\begin{equation}
V_\pi(s)=\mathbb{E}\_{\pi,p}\left[\sum_{t=0}^{\infty}\gamma^t\Big(r_t+\alpha H\big(\pi(\cdot\vert s_t)\big)\Big)\Big\vert s_0=s\right],
\end{equation}
and the state-action value function is given as
\begin{equation}
Q_\pi(s,a)=\mathbb{E}\_{\pi,p}\left[r_0+\sum_{t=1}^{\infty}\Big(r_t+\alpha H\big(\pi(\cdot\vert s_t)\big)\Big)\Big\vert s_0=s,a_0=a\right]
\end{equation}
It is worth remarking that those definitions imply that
\begin{equation}
V_\pi(s)=\mathbb{E}\_{a\sim\pi}\Big[Q_\pi(s,a)\Big]+\alpha H\big(\pi(\cdot\vert s)\Big)\label{eq:vf.1}
\end{equation}

### Greedy policy{#greedy-policy}
Recall that in standard setting, the **greedy policy** for state-action value function $Q$ are defined as a deterministic policy that selects the greedy action in the sense that maximizes the state-action value function, i.e.
\begin{equation}
\pi_\text{greedy}(s)\doteq\underset{a}{\text{argmax}}Q(s,a)
\end{equation}
With entropy-regularized, the greedy policy is thus given in stochastic form
\begin{align}
\pi_\text{greedy}(\cdot\vert s)&\doteq\underset{\pi}{\text{argmax}}\mathbb{E}\_{a\sim\pi}\Big[Q(s,a)\Big]+\alpha H\big(\pi(\cdot\vert s)\big) \\\\ &=\frac{\exp\left(\frac{1}{\alpha}Q(s,a)\right)}{\mathbb{E}\_{a'\sim\tilde{\pi}}\left[\exp\left(\frac{1}{\alpha}Q(s,a')\right)\right]},
\end{align}
where $\tilde{\pi}$ is some "reference" policy, and thus the denominator is acting as a normalizing constant since it is dependent of $\pi$.

To verify this, we begin by considering
\begin{align}
\hspace{-0.7cm}H\big(\pi(\cdot\vert s)\big)&=-\text{KL}(\pi\Vert\pi_\text{greedy})(s)-\mathbb{E}\_{a\sim\pi}\big[\log\pi_\text{greedy}(a\vert s)\big] \\\\ &=-\text{KL}(\pi\vert\pi_\text{greedy})(s)-\mathbb{E}\_{a\sim\pi}\left[\frac{1}{\alpha}Q(s,a)-\log\mathbb{E}\_{a\sim\tilde{\pi}}\left[\exp\left(\frac{1}{\alpha}Q(s,a)\right)\right]\right] \\\\ &=-\text{KL}(\pi\Vert\pi_\text{greedy})(s)-\frac{1}{\alpha}\mathbb{E}\_{a\sim\pi}\big[Q(s,a)\big]+\log\mathbb{E}\_{a\sim\tilde{\pi}}\left[\exp\left(\frac{1}{\alpha}Q(s,a)\right)\right],\label{eq:gp.1}
\end{align}
where $\text{KL}(\pi\Vert\pi_\text{greedy})(s)\doteq D_\text{KL}\big(\pi(\cdot\vert s)\Vert\pi_\text{greedy}(\cdot\vert s)\big)$ denotes the KL divergence between $\pi(\cdot\vert s)$ and $\pi_\text{greedy}(\cdot\vert s)$. The result \eqref{eq:gp.1} implies that
\begin{equation}
\hspace{-1cm}\mathbb{E}\_{a\sim\pi}\big[Q(s,a)\big]+\alpha H\big(\pi(\cdot\vert s)\big)=-\alpha\text{KL}(\pi\Vert\pi_\text{greedy})(s)+\alpha\log\mathbb{E}\_{a\sim\tilde{\pi}}\left[\exp\left(\frac{1}{\alpha}Q(s,a)\right)\right]
\end{equation}
Since $\alpha\log\mathbb{E}\_{a\sim\tilde{\pi}}\left[\exp\left(\frac{1}{\alpha}Q(s,a)\right)\right]$ does not depend on $\pi$ and $\alpha\in[0,1]$, the policy $\pi$ that maximizes LHS is the one that minimizes $D_\text{KL}\big(\pi(\cdot\vert s)\Vert\pi_\text{greedy}(\cdot\vert s)\big)$, which proves our claim due to $D_\text{KL}\big(\pi(\cdot\vert s)\Vert\pi_\text{greedy}(\cdot\vert s)\big)\geq 0$ with equality holds when $\pi=\pi_\text{greedy}$.

### Bellman backup operators{#bellman-op}
In standard RL, let $\mathcal{T}\_\pi$ be the Bellman operator[^1], with which we can compute the expected returns by one-step lookahead, i.e.
\begin{align}
(\mathcal{T}\_\pi V_\pi)(s)&=\sum_a\pi(a\vert s)\Big[r(s,a)+\gamma V_\pi(s')\Big] \\\\ &=\mathbb{E}\_{a\sim\pi}\Big[r(s,a)+\gamma\mathbb{E}\_{s'\sim p}\big[V_\pi(s')\big]\Big] \\\\ &=\mathbb{E}\_{a\sim\pi,s'\sim p}\Big[r(s,a)+\gamma V_\pi(s')\Big]\label{eq:bo.1}
\end{align}
and
\begin{align}
(\mathcal{T}\_\pi Q_\pi)(s,a)&=r(s,a)+\gamma\sum_{s',a'}p(s'\vert s,a)\pi(a'\vert s')Q_\pi(s',a') \\\\ &=r(s,a)+\gamma\mathbb{E}\_{s'\sim p,a'\sim\pi}\Big[Q_\pi(s',a')\Big] \\\\ &=\mathbb{E}\_{s'\sim p}\Big[r(s,a)+\gamma\mathbb{E}\_{a'\sim\pi}\big[Q_\pi(s',a')\big]\Big] \\\\ &=\mathbb{E}\_{s'\sim p,a'\sim\pi}\Big[r(s,a)+\gamma Q_\pi(s',a')\Big]
\end{align}
Repeatedly applying $\mathcal{T}\_\pi$ operator $n$ times yields the $n$-step Bellman operator, denoted $\mathcal{T}\_\pi^{(n)}$, which allows us to compute the expected returns with $n$-step lookahead, i.e.
\begin{align}
(\mathcal{T}\_\pi^{(n)}V_\pi)(s)&=\mathbb{E}\_{\pi,p}\left[\sum_{t=0}^{n-1}\gamma^t r_t+\gamma^n V_\pi(s_n)\Bigg\vert s_0=s\right], \\\\ (\mathcal{T}\_\pi^{(n)}Q_\pi)(s,a)&=\mathbb{E}\_{\pi,p}\left[\sum_{t=0}^{n-1}\gamma^t r_t+\gamma^n Q_\pi(s_n,a_n)\Bigg\vert s_0=s,a_0=a\right]
\end{align}
We can generalize the Bellman operator $\mathcal{T}\_\pi$ to the maximum entropy setting to get a recursive relationship between successive value functions. Specifically, by adding an entropy term, \eqref{eq:bo.1} can be rewritten as
\begin{equation}
(\mathcal{T}\_\pi V_\pi)(s)=\mathbb{E}\_{a\sim\pi,s'\sim p}\Big[r(s,a)+\alpha H\big(\pi(\cdot\vert s)\big)+\gamma V_\pi(s')\Big],
\end{equation}
and analogously we have
\begin{align}
(\mathcal{T}\_\pi Q_\pi)(s,a)&=\mathbb{E}\_{s'\sim p}\Big[r(s,a)+\gamma\Big(\mathbb{E}\_{a'\sim\pi}\big[Q_\pi(s',a')\big]+\alpha H\big(\pi(\cdot\vert s)\big)\Big)\Big] \\\\ &=\mathbb{E}\_{s'\sim p,a'\sim\pi}\Big[r(s,a)+\gamma\Big(Q_\pi(s',a')+\alpha H\big(\pi(\cdot\vert s)\big)\Big)\Big]
\end{align}
And also, the $n$-step Bellman operators generalize for entropy regularization framework are given by
\begin{equation}
(\mathcal{T}\_\pi^{(n)}V_\pi)(s)=\mathbb{E}\_{\pi,p}\left[\sum_{t=0}^{n-1}\gamma^t\Big(r_t+\gamma H\big(\pi(\cdot\vert s_t)\big)\Big)+\gamma^n V_\pi(s_n)\Bigg\vert s_0=s\right],\label{eq:bo.2}
\end{equation}
and
\begin{align}
\hspace{-0.8cm}&(\mathcal{T}\_\pi^{(n)}Q_\pi)(s,a)+\alpha H\big(\pi(\cdot\vert s)\big)\nonumber \\\\ \hspace{-0.8cm}&=\mathbb{E}\_{\pi,p}\left[\sum_{t=0}^{\infty}\gamma^t\Big(r_t+\alpha H\big(\pi(\cdot\vert s_t)\big)\Big)+\gamma^n\Big(Q_\pi(s_n,a_n)+\alpha H\big(\pi(\cdot\vert s_n)\big)\Big)\Bigg\vert s_0=s,a_0=a\right]
\end{align}
The above $n$-step Bellman operator for state-action value can also be deduced by combining \eqref{eq:bo.2} with the result \eqref{eq:vf.1}.

## Policy Iteration{#policy-iter}
The Bellman operator provides useful facts to apply the dynamic programming method - [policy iteration]({{< ref "dp-in-mdp#policy-iter" >}}), which alternates between [policy evaluation]({{< ref "dp-in-mdp#policy-eval" >}}) and [policy improvement]({{< ref "dp-in-mdp#policy-imp" >}}) processes, and eventually we end up with the optimal policy. More importantly, it has been [proved](#sql-apper) that we can also apply the method to maximum entropy RL.

### Policy Evaluation{#policy-eval}
In **policy evaluation** process, we wish to compute the value of a given policy according to the entropy regularization objective \eqref{eq:mr.1}. This can be found by an iterative method.

Specifically, starting from an arbitrary function $Q^{(0)}:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ with $\vert\mathcal{A}\vert<\infty$ and consider the **soft Bellman backup operator**, denoted $\mathcal{T}^\pi$, which is defined as
\begin{equation}
Q^{(k+1)}(s,a)\doteq\mathcal{T}^\pi Q^{(k)}(s,a)\doteq r(s,a)+\gamma\mathbb{E}\_{s'\sim\rho_\pi}\big[V^{(k)}(s')\big],\label{eq:spe.1}
\end{equation}
where
\begin{equation}
V^{(k)}(s)=\mathbb{E}\_{a\sim\rho_\pi}\big[Q^{(k)}(s,a)-\log\pi(a\vert s)\big]
\end{equation}
The resulting sequence $\\{Q^{(k)}\\}\_{k=0,1,\ldots}$ will converges to a fixed point called **soft Q-value**, denoted $Q_\text{soft}$.

**Proof**  
Let $r^\pi(s,a)\doteq r(s,a)+\mathbb{E}\_{s'\sim\rho_\pi}\big[\alpha H\big(\pi(\cdot\vert s')\big)\big]$ denote the entropy augmented reward, the Bellman backup \eqref{eq:spe.1} can rewrite as
\begin{equation}
Q^{(k+1)}(s,a)=r^\pi(s,a)+\gamma\mathbb{E}\_{(s',a')\sim\rho_\pi}\big[Q^{(k)}(s',a')\big]
\end{equation}
Since $\vert\mathcal{A}\vert<\infty$, we have that $r^\pi(s,a)$ is bounded. Analogy to the [standard policy evaluation]({{< ref "optimal-policy-existence" >}}), we then can prove that $\mathcal{T}^\pi$ is a [contraction mapping]({{< ref "optimal-policy-existence#contractions" >}}) and then by using the [**Banach's fixed point theorem**]({{< ref "optimal-policy-existence#banach-fixed-pts-theorem" >}}), we can show that $\\{Q^{(k)}\\}\_{k=0,1,\ldots}$ eventually converges to a fixed point, which we call it **soft Q-value**.






## References
[1] <span id='sql-paper'>Tuomas Haarnoja, Haoran Tang, Pieter Abbeel, Sergey Levine. [Reinforcement Learning with Deep Energy-Based Policies](https://dl.acm.org/doi/10.5555/3305381.3305521). ICML, 2017</span>.

[2] Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine. [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/abs/1801.01290). arXiv preprint arXiv:1812.05905, 2018.

[3] Brian D. Ziebart. [Modeling purposeful adaptive behavior with the principle of maximum causal entropy](https://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf). PhD Thesis, Carnegie Mellon University, 2010.

[4] John Schulman, Xi Chen, Pieter Abbeel. [Equivalence Between Policy Gradients and Soft Q-Learning](https://arxiv.org/abs/1704.06440). arXiv preprint arXiv:1704.06440, 2018.

[5] Csaba Szepesvári. [Algorithms for Reinforcement Learning](https://doi.org/10.1007/978-3-031-01551-9). Synthesis Lectures on Artificial Intelligence and Machine Learning, 2010.

[6] Richard S. Sutton, Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

## Footnotes
[^1]: With an abuse of notation, $\mathcal{T}\_\pi$ implicitly represents two mappings $\mathcal{T}\_\pi:\mathcal{S}\to\mathcal{S}$ and $\mathcal{T}\_\pi':\mathcal{S}\times\mathcal{A}\to\mathcal{S}\times\mathcal{A}$.
