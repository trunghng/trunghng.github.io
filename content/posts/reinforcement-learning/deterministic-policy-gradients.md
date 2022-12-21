---
title: "Deterministic Policy Gradients"
date: 2022-12-02T19:26:44+07:00
tags: [reinforcement-learning, policy-gradient, my-rl]
math: true
eqn-number: true
---
> Notes on Deterministic Policy Gradient Algorithms
<!--more-->

## Preliminaries
Consider a (infinite-horizon) Markov Decision Process (MDP), defined as a tuple of $(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma)$ where
- $\mathcal{S}$ is the **state space**.
- $\mathcal{A}$ is the **action space**.
- $p:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the **transition probability distribution**, i.e. $p(s,a,s')=p(s'\vert s,a)$ denotes the probability of transitioning to state $s'$ when taking action $a$ from state $s$.
- $r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ is the **reward function**, and let us denote $r_{t+1}\doteq r(S_t,A_t)$.
- $\rho_0:\mathcal{S}\to\mathbb{R}$ is the distribution of the initial state $s_0$.
- $\gamma\in(0,1)$ is the **discount factor**.

Within an MPD, a policy parameterized by a vector $\theta\in\mathbb{R}^n$ can be given as
<ul id='number-listt'>
	<li>
		<b>Stochastic policy</b>. $\pi_\theta:\mathcal{S}\times\mathcal{A}\to[0,1]$, or
	</li>
	<li>
		<b>Deterministic policy</b>. $\mu_\theta:\mathcal{S}\to\mathcal{A}$.
	</li>
</ul>
In addition, it is worth noticing that the deterministic policy is a special case of the stochastic policy.

## (Stochastic) Policy Gradient Theorem{#spg-theorem}
We continue with an assumption that the action space $\mathcal{A}=\mathbb{R}^m$ and the state space $\mathcal{S}\subset\mathbb{R}^d$ and $\mathcal{S}$ is compact.

### Start-state formulation{#start-state}
Recall that in the [general case]({{< ref "policy-gradient-theorem#policy-grad-theorem-ep" >}}), i.e. policy is given in stochastic form, $\pi_\theta$, the **Policy Gradient Theorem** states that[^1]
\begin{align}
\nabla_\theta J(\pi_\theta)&=\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)\hspace{0.1cm}da\hspace{0.1cm}ds\label{eq:spgt.1} \\\\ &=\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)\Big]\label{eq:spgt.2}
\end{align}
where
<ul id='number-list'>
	<li>
		$J(\pi_\theta)$ is the <b>performance objective function</b>, which we are trying to maximize. It is defined as the expected cumulative discounted reward from the start state
		\begin{align}
		J(\pi_\theta)&\doteq\mathbb{E}_{\rho_0,\pi_\theta}\big[r_0^\gamma\big]\label{eq:spgt.6} \\ &=\mathbb{E}_{\rho_0,\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_{t+1}\right] \\ &=\mathbb{E}_{\rho_0,\pi_\theta}\big[G_0\big] \\ &=\mathbb{E}_{s_0\sim\rho_0}\Big[\mathbb{E}_{\pi_\theta}\big[G_t\vert S_t=s_0\big]\Big] \\ &=\mathbb{E}_{s_0\sim\rho_0}\big[V_\pi(s_0)\big],\label{eq:spgt.3}
		\end{align}
		where $r_t^\gamma$ is defined as the total discounted reward from time-step $t$ onward, which is thus the return at that step
		\begin{equation}
		G_t=r_t^\gamma\doteq\sum_{t=0}^{\infty}\gamma^t r_{t+1}
		\end{equation}
	</li>
	<li>
		The function $\rho_\pi(s)$ is defined as the discounted weighting of states encountered when starting at $s_0$ and following $\pi_\theta$ thereafter
		\begin{align}
		\rho_\pi(s)&\doteq\sum_{t=0}^{\infty}\gamma^t P(S_t=s\vert s_0,\pi_\theta) \\ &=\int_\mathcal{S}\rho_0(\bar{s})\left(\sum_{t=0}^{\infty}\gamma^t P(S_t=s\vert\pi_\theta)\right)d\bar{s} \\ &=\int_\mathcal{S}\sum_{t=0}^{\infty}\gamma^t\rho_0(\bar{s})p(\bar{s}\to s,t,\pi_\theta)d\bar{s},\label{eq:spgt.4}
		\end{align}
		where $p(\bar{s}\to s,t,\pi_\theta)$ is defined as the the probability of transitioning to $s$ after $t$ steps starting from $\bar{s}$ under $\pi_\theta$, which implies that
		\begin{equation}
		p(\bar{s}\to s,t,\pi_\theta)=P(S_t=s\vert\pi_\theta)
		\end{equation}
		In fact, $\rho_\pi(s)$ can be seen as a density function since integrating $\rho_\pi$ over state space $\mathcal{S}$ gives us
		\begin{align}
		\int_\mathcal{S}\rho_\pi(s)d s&=\int_{s\in\mathcal{S}}\int_{\bar{s}\in\mathcal{S}}\sum_{t=0}^{\infty}\gamma^t\rho_0(\bar{s})p(\bar{s}\to s,t,\pi_\theta)d\bar{s}\hspace{0.1cm}d s \\ &=\int_{\bar{s}\in\mathcal{S}}\rho_0(\bar{s})\int_{s\in\mathcal{S}}\sum_{t=0}^{\infty}\gamma^t p(\bar{s}\to s,t,\pi_\theta)d s\hspace{0.1cm}d\bar{s} \\ &=\int_{\bar{s}\in\mathcal{S}}\rho_0(\bar{s})\sum_{t=0}^{\infty}\gamma^t\underbrace{\int_{s\in\mathcal{S}}p(\bar{s}\to s,t,\pi_\theta)d s}_{=1}d\bar{s} \\ &=\int_\mathcal{S}\rho_0(\bar{s})\underbrace{\sum_{t=0}^{\infty}\gamma^t}_{=1}d\bar{s} \\ &=\int_\mathcal{S}\rho_0(\bar{s})d\bar{s} \\ &=1
		\end{align}
		Thus, this definition of $\rho_\pi$ lets us write \eqref{eq:spgt.1} as an expectation, combined with using the <b>log-likelihood trick</b>, we end up with \eqref{eq:spgt.2}.
	</li>
</ul>

**Proof**  
The definition of $J(\pi_\theta)$ given in \eqref{eq:spgt.3} suggests us begin by considering the gradient of the state value function w.r.t $\theta$. For any $s\in\mathcal{S}$, we have
\begin{align}
\hspace{-1cm}\nabla_\theta V_\pi(s)&=\nabla_\theta\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da \\\\ &=\underbrace{\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da}\_{\nu(s)}+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)da \\\\ &=\nu(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\left(r(s,a)+\int_\mathcal{S}\gamma p(s'\vert s,a)V_\pi(s')d s'\right)da \\\\ &=\nu(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}\gamma p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da \\\\ &\overset{\text{(i)}}{=}\nu(s)+\int_\mathcal{S}\left(\int_\mathcal{A}\gamma\pi_\theta(a\vert s)p(s'\vert s,a)da\right)\nabla_\theta V_\pi(s')d s' \\\\ &=\nu(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\nabla_\theta V_\pi(s')d s' \\\\ &=\nu(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\left(\nu(s')+\int_\mathcal{S}\gamma p(s'\to s'',1,\pi_\theta)\nabla_\theta V_\pi(s'')d s''\right)d s' \\\\ &=\nu(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\nu(s')d s'\nonumber \\\\ &\hspace{2cm}+\int_\mathcal{S}\int_\mathcal{S}\gamma^2 p(s\to s',1,\pi_\theta)p(s'\to s'',1,\pi_\theta)\nabla_\theta V_\pi(s'')d s''\hspace{0.1cm}d s' \\\\ &\overset{\text{(ii)}}{=}\nu(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\nu(s')d s'+\int_\mathcal{S}\gamma^2 p(s\to s'',2,\pi_\theta)\nabla_\theta V_\pi(s'')d s'' \\\\ &=\nu(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\nu(s')d s'+\int_\mathcal{S}\gamma^2 p(s\to s'',2,\pi_\theta)\nu(s'')d s''\nonumber \\\\ &\hspace{2cm}+\int_\mathcal{S}\gamma^3 p(s\to s''',3,\pi_\theta)\nabla_\theta V_\pi(s''')d s''' \\\\ &\hspace{0.3cm}\vdots\nonumber \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\pi_\theta)\nu(\tilde{s})d\tilde{s} \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert\tilde{s})Q_\pi(\tilde{s},a)da\hspace{0.1cm}d\tilde{s}\label{eq:spgt.5}
\end{align}
where
<ul id='roman-list'>
	<li>
		In this step, we have used the <b>Fubini's theorem</b> to exchange the order of integrals.
	</li>
	<li>
		We have once again used the <b>Fubini's theorem</b> to exchange the order of integration combined with the identity
		\begin{equation}
		\int_\mathcal{S}p(s\to s',1,\pi_\theta)p(s'\to s'',1,\pi_\theta)d s'=p(s\to s'',2,\pi_\theta)
		\end{equation}
	</li>
</ul>

Combining \eqref{eq:spgt.3},\eqref{eq:spgt.4} and \eqref{eq:spgt.5} together allows us to obtain
\begin{align}
\hspace{-1cm}\nabla_\theta J(\pi_\theta)&=\nabla_\theta\mathbb{E}\_{s_0\sim\rho_0}\big[V_\pi(s_0)\big] \\\\ &=\int_\mathcal{S}\rho_0(s_0)\nabla_\theta V_\pi(s_0)d s_0 \\\\ &=\int_{s_0\in\mathcal{S}}\rho_0(s_0)\int_{s\in\mathcal{S}}\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s\hspace{0.1cm}d s_0 \\\\ &\overset{\text{(i)}}{=}\int_{s\in\mathcal{S}}\int_{s_0\in\mathcal{S}}\rho_0(s_0)\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s_0\hspace{0.1cm}d s \\\\ &\overset{\text{(ii)}}{=}\int_{s\in\mathcal{S}}\int_\mathcal{A}\left(\int_{s_0\in\mathcal{S}}\rho_0(s_0)\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)\right)\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)d s_0\hspace{0.1cm}da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\int_\mathcal{A}\rho_\pi(s)\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s,
\end{align}
where in two steps (i) and (ii), we have exchanged the order of integration by respectively applying the **Fubini's theorem**.$\tag*{$\Box$}$

The theorem lets us rewrite the policy gradient $\nabla_\theta J(\pi_\theta)$ in terms of which does not depend the gradient of state distribution, $\nabla_\theta\rho_\pi(s)$, despite of the fact that $J(\pi_\theta)$ depends on $\rho_\pi(s)$.

The above policy gradient theorem is explicitly known as the **start-state policy gradient theorem** (since it is given in terms of the start state distribution $\rho_0$) defined by the policy objective function $J(\pi_\theta)$, given in \eqref{eq:spgt.6}, for episodic and discounted tasks. To extend the theorem in case of continuing problems, in which the interaction between agent and the environment lasts forever, we start by giving a new definition for the $J(\pi_\theta)$.

### Average-reward formulation{#avg-reward}
The performance objective function in such continuing tasks is defined as the average rate of reward, or **average reward**, denoted $r(\pi_\theta)$[^2], while following policy $\pi_\theta$
\begin{align}
J(\pi_\theta)\doteq r(\pi_\theta)&\doteq\lim_{h\to\infty}\frac{1}{h}\sum_{t=0}^{h}\mathbb{E}\big[r_{t+1}\vert S_0,A_{0:t}\sim\pi_\theta\big] \\\\ &=\lim_{t\to\infty}\mathbb{E}\big[r_{t+1}\vert S_0,A_{0:t}\sim\pi_\theta\big] \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)r(s,a)da\hspace{0.1cm}d s,
\end{align}
where $\bar{\rho}\_\pi(s)\doteq\lim_{t\to\infty}P(S_t=s\vert A_{0:t}\sim\pi_\theta)$ is the **stationary distribution**[^3] of states under policy $\pi_\theta$ which is assumed to exist and does not depend on the start state $S_0$ despite of the fact that the expectations are conditioned on $S_0$.

An MDP with this assumption is referred as an **ergodic MDP**, in which the MDP starts or any early decision made by the agent can have only a temporary effect; in the long run the expectation of being in a state depends only on the policy and the MDP transition probabilities.

Analogously, in continuing problems, the return at time-step $t$, $G_t$, is instead called **differential return** and is defined in terms of differences between rewards and the average reward $r(\pi_\theta)$
\begin{equation}
G_t\doteq\sum_{k=1}^{\infty}\big[r_{t+k}-r(\pi_\theta)\big]
\end{equation}
The value functions, which are defined as the expected return, therefore  are known as **differential value functions** and are respectively given by
\begin{align}
V_\pi(s)&\doteq\mathbb{E}\_{\pi_\theta}\big[G_t\vert S_t=s\big] \\\\ &=\mathbb{E}\_{\pi_\theta}\left[\sum_{k=1}^{\infty}\big(r_{t+k}-r(\pi_\theta)\big)\Big\vert S_t=s\right] \\\\ &=\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\big[r(s,a)-r(\pi_\theta)+V_\pi(s')\big]d s' da \\\\ &=\int_\mathcal{A}\pi_\theta(a\vert s)r(s,a)da-r(\pi_\theta)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)V_\pi(s')d s' da
\end{align}
and
\begin{align}
Q_\pi(s,a)&\doteq\mathbb{E}\_{\pi_\theta}\big[G_t\vert S_t=s,A_t=a\big] \\\\ &=\mathbb{E}\_{\pi_\theta}\left[\sum_{k=1}^{\infty}\big(r_{t+k}-r(\pi_\theta)\big)\Big\vert S_t=s,A_t=a\right] \\\\ &=\int_\mathcal{S}p(s'\vert s,a)\left[r(s,a)-r(\pi_\theta)+\int_\mathcal{A}\pi_\theta(a'\vert s')Q_\pi(s',a')d a'\right]d s' \\\\ &=r(s,a)-r(\pi_\theta)+\int_\mathcal{S}p(s'\vert s,a)\int_\mathcal{A}\pi_\theta(a'\vert s')Q_\pi(s',a')d a' d s' \\\\ &=r(s,a)-r(\pi_\theta)+\int_\mathcal{S}p(s'\vert s,a)V_\pi(s')d s'
\end{align}
Now we are ready for the **policy gradient theorem** specified for continuing tasks, and hence is called **average-reward policy gradient theorem**. The theorem states that
\begin{align}
\nabla_\theta J(\pi_\theta)&=\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)\hspace{0.1cm}da\hspace{0.1cm}ds\label{eq:spgt.8} \\\\ &=\mathbb{E}\_{\bar{\rho}\_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)\Big]
\end{align}
**Proof**  
Analogy to the episodic case, we start with the gradient of the state value function w.r.t $\theta$. For any $s\in\mathcal{S}$, we have
\begin{align}
\nabla_\theta V_\pi(s)&=\nabla_\theta\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da \\\\ &=\underbrace{\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da}\_{\nu(s)}+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)da \\\\ &=\nu(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\left[r(s,a)-r(\pi_\theta)+\int_\mathcal{S}p(s'\vert s,a)V_\pi(s')d s'\right]da \\\\ &=\nu(s)-\nabla_\theta r(\pi_\theta)\underbrace{\int_\mathcal{A}\pi_\theta(a\vert s)da}\_{=1}+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da \\\\ &=\nu(s)-\nabla_\theta r(\pi_\theta)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da,
\end{align}
which implies that the policy gradient can be obtained as
\begin{equation}
\nabla_\theta J(\pi_\theta)=\nabla_\theta r(\pi_\theta)=\nu(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da-\nabla_\theta V_\pi(s)\label{eq:spgt.7}
\end{equation}
Using the identity
\begin{equation}
\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta J(\pi_\theta)d s=\nabla_\theta J(\pi_\theta)\int_\mathcal{S}\bar{\rho}\_\pi(s)d s=\nabla_\theta J(\pi_\theta),
\end{equation}
we can continue to derive \eqref{eq:spgt.7} as
\begin{align}
\hspace{-0.5cm}\nabla_\theta J(\pi_\theta)&=\int_\mathcal{S}\bar{\rho}\_\pi(s)\nu(s)d s+\int_{s\in\mathcal{S}}\bar{\rho}\_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\int_{s'\in\mathcal{S}}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da\hspace{0.1cm}d s\nonumber \\\\ &\hspace{2cm}-\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta V_\pi(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\nu(s)d s+\int_\mathcal{S}\bar{\rho}\_\pi(s')\nabla_\theta V_\pi(s')d s'-\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta V_\pi(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\nu(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s
\end{align}
where the second step is due to \eqref{eq:fn.1}.$\tag*{$\Box$}$

It can be seen that the (stochastic) policy gradient theorem specified in both discounted episodic and continuing settings have the same formulation. In particular, if we replace the state distribution $\rho_\pi$ in start-state formulation \eqref{eq:spgt.1} by the stationary distribution $\bar{\rho}\_\pi$ (also with new definition of the value functions), we obtain average-reward formulation \eqref{eq:spgt.8}.









## Deterministic Policy Gradient Theorem{#dpg-theorem}


## References
[1] David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller. [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf). JMLR 2014.

[2] Richard S. Sutton, Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

[3] Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). NIPS 1999.

## Footnotes
[^1]: To simplify the notation, we have let $\rho_\pi\doteq\rho\_{\pi_\theta}$ and $Q_\pi\doteq Q_{\pi_\theta}$ implicitly. As a result, we will also denote $V_{\pi_\theta}$ by $V_\pi$.
[^2]: The notation $r(\pi)$ of the **average reward** is just a notation-abused and should not be confused with notation $r$ of the reward function.
[^3]: This means that if we keep selecting action according to $\pi_\theta$, we remains in the same state distribution $\bar{\rho}\_\pi$, i.e.
\begin{equation}
\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)p(s'\vert s,a)da\hspace{0.1cm}d s=\bar{\rho}\_\pi(s')\label{eq:fn.1}
\end{equation}
