---
title: "Deterministic Policy Gradients"
date: 2022-12-02T19:26:44+07:00
tags: [reinforcement-learning, policy-gradient, actor-critic, model-free, my-rl]
math: true
eqn-number: true
---
> Notes on Deterministic Policy Gradient algorithms
<!--more-->

## Preliminaries
Consider a (infinite-horizon) Markov Decision Process (MDP), defined as a tuple of $(\mathcal{S},\mathcal{A},p,r,\rho_0,\gamma)$ where
- $\mathcal{S}$ is the **state space**.
- $\mathcal{A}$ is the **action space**.
- $p:\mathcal{S}\times\mathcal{A}\times\mathcal{S}\to[0,1]$ is the **transition probability distribution**, i.e. $p(s,a,s')=p(s'\vert s,a)$ denotes the probability of transitioning to state $s'$ when taking action $a$ from state $s$.
- $r:\mathcal{S}\times\mathcal{A}\to\mathbb{R}$ is the **reward function**, and let us denote $r_{t+1}\doteq r(s_t,a_t)$.
- $\rho_0:\mathcal{S}\to\mathbb{R}$ is the distribution of the initial state $s_0$.
- $\gamma\in(0,1)$ is the **discount factor**.

Within an MDP, a policy parameterized by a vector $\theta\in\mathbb{R}^n$ can be given as
<ul class='number-listt'>
	<li>
		<b>Stochastic policy</b>. $\pi_\theta:\mathcal{S}\times\mathcal{A}\to[0,1]$, or
	</li>
	<li>
		<b>Deterministic policy</b>. $\mu_\theta:\mathcal{S}\to\mathcal{A}$.
	</li>
</ul>

## (Stochastic) Policy Gradient Theorem{#spg-theorem}
We continue with an assumption that the action space $\mathcal{A}=\mathbb{R}^m$ and the state space $\mathcal{S}\subset\mathbb{R}^d$ and $\mathcal{S}$ is compact.

### Start-state formulation{#start-state}
Recall that in the stochastic case, $\pi_\theta$, the [**Policy Gradient Theorem**]({{<ref"policy-gradient-theorem#policy-grad-theorem-ep">}}) states that[^1]
\begin{align}
\nabla_\theta J(\pi_\theta)&=\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)\hspace{0.1cm}da\hspace{0.1cm}ds\label{eq:spgt.1} \\\\ &=\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)\Big]\label{eq:spgt.2}
\end{align}
where
<ul class='number-list'>
	<li>
		$J(\pi_\theta)$ is the <b>performance objective function</b>, which we are trying to maximize. It is defined as the expected cumulative discounted reward from the start state
		\begin{align}
		J(\pi_\theta)&\doteq\mathbb{E}_{\rho_0,\pi_\theta}\big[r_0^\gamma\big]\label{eq:spgt.6} \\ &=\mathbb{E}_{\rho_0,\pi_\theta}\left[\sum_{t=0}^{\infty}\gamma^t r_{t+1}\right] \\ &=\mathbb{E}_{\rho_0,\pi_\theta}\big[G_0\big] \\ &=\mathbb{E}_{s_0\sim\rho_0}\Big[\mathbb{E}_{\pi_\theta}\big[G_t\vert S_t=s_0\big]\Big] \\ &=\mathbb{E}_{s_0\sim\rho_0}\big[V_\pi(s_0)\big],\label{eq:spgt.3}
		\end{align}
		where $r_t^\gamma$ is defined as the total discounted reward from time-step $t$ onward, which is thus the return at that step
		\begin{equation}
		G_t=r_t^\gamma\doteq\sum_{k=t}^{\infty}\gamma^{k-t}r(s_t,a_t)=\sum_{k=t}^{\infty}\gamma^{k-t}r_{k+1}
		\end{equation}
	</li>
	<li>
		The function $\rho_\pi(s)$ is defined as the discounted weighting of states encountered when starting at $s_0$ and following $\pi_\theta$ thereafter
		\begin{align}
		\rho_\pi(s)&\doteq\sum_{t=0}^{\infty}\gamma^t P(S_t=s\vert s_0,\pi_\theta) \\ &=\int_\mathcal{S}\rho_0(\bar{s})\left(\sum_{t=0}^{\infty}\gamma^t P(S_t=s\vert\pi_\theta)\right)d\bar{s} \\ &=\int_\mathcal{S}\sum_{t=0}^{\infty}\gamma^t\rho_0(\bar{s})p(\bar{s}\to s,t,\pi_\theta)d\bar{s},\label{eq:spgt.4}
		\end{align}
		where $p(\bar{s}\to s,t,\pi_\theta)$ is defined as the probability of transitioning to $s$ after $t$ steps starting from $\bar{s}$ under $\pi_\theta$, which implies that
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
\hspace{-1cm}\nabla_\theta V_\pi(s)&=\nabla_\theta\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da \\\\ &=\underbrace{\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da}\_{\psi(s)}+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)da \\\\ &=\psi(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\left(r(s,a)+\int_\mathcal{S}\gamma p(s'\vert s,a)V_\pi(s')d s'\right)da \\\\ &=\psi(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}\gamma p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da \\\\ &\overset{\text{(i)}}{=}\psi(s)+\int_\mathcal{S}\left(\int_\mathcal{A}\gamma\pi_\theta(a\vert s)p(s'\vert s,a)da\right)\nabla_\theta V_\pi(s')d s' \\\\ &=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\nabla_\theta V_\pi(s')d s'\label{eq:spgt.9} \\\\ &=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\left(\psi(s')+\int_\mathcal{S}\gamma p(s'\to s'',1,\pi_\theta)\nabla_\theta V_\pi(s'')d s''\right)d s' \\\\ &=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\psi(s')d s'\nonumber \\\\ &\hspace{2cm}+\int_\mathcal{S}\int_\mathcal{S}\gamma^2 p(s\to s',1,\pi_\theta)p(s'\to s'',1,\pi_\theta)\nabla_\theta V_\pi(s'')d s''\hspace{0.1cm}d s' \\\\ &\overset{\text{(ii)}}{=}\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\psi(s')d s'+\int_\mathcal{S}\gamma^2 p(s\to s'',2,\pi_\theta)\nabla_\theta V_\pi(s'')d s'' \\\\ &=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\pi_\theta)\psi(s')d s'+\int_\mathcal{S}\gamma^2 p(s\to s'',2,\pi_\theta)\psi(s'')d s''\nonumber \\\\ &\hspace{2cm}+\int_\mathcal{S}\gamma^3 p(s\to s''',3,\pi_\theta)\nabla_\theta V_\pi(s''')d s''' \\\\ &\hspace{0.3cm}\vdots\nonumber \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\pi_\theta)\psi(\tilde{s})d\tilde{s} \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert\tilde{s})Q_\pi(\tilde{s},a)da\hspace{0.1cm}d\tilde{s}\label{eq:spgt.5}
\end{align}
where
<ul class='roman-list'>
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
\hspace{-1cm}\nabla_\theta J(\pi_\theta)&=\nabla_\theta\mathbb{E}\_{s_0\sim\rho_0}\big[V_\pi(s_0)\big] \\\\ &=\int_\mathcal{S}\rho_0(s_0)\nabla_\theta V_\pi(s_0)d s_0 \\\\ &=\int_{s_0\in\mathcal{S}}\rho_0(s_0)\int_{s\in\mathcal{S}}\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s\hspace{0.1cm}d s_0 \\\\ &\overset{\text{(i)}}{=}\int_{s\in\mathcal{S}}\int_{s_0\in\mathcal{S}}\rho_0(s_0)\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s_0\hspace{0.1cm}d s \\\\ &\overset{\text{(ii)}}{=}\int_{s\in\mathcal{S}}\int_\mathcal{A}\left(\int_{s_0\in\mathcal{S}}\rho_0(s_0)\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\pi_\theta)d s_0\right)\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\int_\mathcal{A}\rho_\pi(s)\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s,
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
\nabla_\theta V_\pi(s)&=\nabla_\theta\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da \\\\ &=\underbrace{\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da}\_{\psi(s)}+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)da \\\\ &=\psi(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\left[r(s,a)-r(\pi_\theta)+\int_\mathcal{S}p(s'\vert s,a)V_\pi(s')d s'\right]da \\\\ &=\psi(s)-\nabla_\theta r(\pi_\theta)\underbrace{\int_\mathcal{A}\pi_\theta(a\vert s)da}\_{=1}+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da \\\\ &=\psi(s)-\nabla_\theta r(\pi_\theta)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da,
\end{align}
which implies that the policy gradient can be obtained as
\begin{equation}
\nabla_\theta J(\pi_\theta)=\nabla_\theta r(\pi_\theta)=\psi(s)+\int_\mathcal{A}\pi_\theta(a\vert s)\int_\mathcal{S}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da-\nabla_\theta V_\pi(s)\label{eq:spgt.7}
\end{equation}
Using the identity
\begin{equation}
\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta J(\pi_\theta)d s=\nabla_\theta J(\pi_\theta)\int_\mathcal{S}\bar{\rho}\_\pi(s)d s=\nabla_\theta J(\pi_\theta),
\end{equation}
we can continue to derive \eqref{eq:spgt.7} as
\begin{align}
\hspace{-0.5cm}\nabla_\theta J(\pi_\theta)&=\int_\mathcal{S}\bar{\rho}\_\pi(s)\psi(s)d s+\int_{s\in\mathcal{S}}\bar{\rho}\_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\int_{s'\in\mathcal{S}}p(s'\vert s,a)\nabla_\theta V_\pi(s')d s' da\hspace{0.1cm}d s\nonumber \\\\ &\hspace{2cm}-\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta V_\pi(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\psi(s)d s+\int_\mathcal{S}\bar{\rho}\_\pi(s')\nabla_\theta V_\pi(s')d s'-\int_\mathcal{S}\bar{\rho}\_\pi(s)\nabla_\theta V_\pi(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\psi(s)d s \\\\ &=\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s
\end{align}
where the second step is due to \eqref{eq:fn.1}.$\tag*{$\Box$}$

It can be seen that the stochastic policy gradient theorem specified in both discounted episodic and continuing settings have the same formulation. In particular, if we replace the state distribution $\rho_\pi$ in start-state formulation \eqref{eq:spgt.1} by the stationary distribution $\bar{\rho}\_\pi$ (also with new definition of the value functions), we obtain average-reward formulation \eqref{eq:spgt.8}. Thus, in the remaining of this note, we will be considering the episodic and discounted setting.

## (Stochastic) Actor-Critic{#stochastic-ac}
Based on the policy gradient theorem, a (stochastic) **actor-critic algorithm** consists of two elements:
<ul class='number-list'>
	<li>
		<b>Actor</b> learns a parameter $\theta$ of the stochastic policy $\pi_\theta$ by iteratively update $\theta$ by SGA using the policy gradient in \eqref{eq:spgt.2}.
	</li>
	<li>
		<b>Critic</b> estimates the value function $Q_\pi(s,a)$ by an state-action value function approximation $Q_w(s,a)$ parameterized by a vector $w$.
	</li>
</ul>

### Policy Gradient with Function Approximation{#pg-func-approx}
Let $Q_w(s,a)$ be a function approximation parameterized by $w\in\mathbb{R}^n$ of the state-action value function $Q_\pi(s,a)$ for a stochastic policy $\pi_\theta$ parameterized by $\theta\in\mathbb{R}^n$. Then if $Q_w(s,a)$ is **compatible** with the policy parameterization in the sense that
<ul class='roman-list'>
	<li>
		<span id='prop-i'>$Q_w(s,a)=\nabla_\theta\log\pi_\theta(a\vert s)^\text{T}w$.</span>
	</li>
	<li>
		<span id='prop-ii'>The parameters $w$ are chosen to minimize the mean-squared error (MSE)</span>
		\begin{equation}
		\epsilon^2(w)=\mathbb{E}_{\rho_\pi,\pi_\theta}\Big[\big(Q_w(s,a)-Q_\pi(s,a)\big)^2\Big]
		\end{equation}
	</li>
</ul>

then
\begin{equation}
\nabla_\theta J(\pi_\theta)=\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_w(s,a)\Big]
\end{equation}

**Proof**  
Taking the gradient w.r.t $w$ of both sides of the equation given in property [(i)](#prop-i) gives us
\begin{equation}
\nabla_w Q_w(s,a)=\nabla_\theta\log\pi_\theta(a\vert s)
\end{equation}
On the other hand, consider the gradient of the MSE, $\epsilon^2(w)$, w.r.t $w$, we have
\begin{align}
\nabla_w\epsilon^2(w)&=\nabla_w\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\big(Q_w(s,a)-Q_\pi(s,a)\big)^2\Big] \\\\ &=\nabla_w\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\big[Q_w(s,a)-Q_\pi(s,a)\big]^2 da\hspace{0.1cm}d s \\\\ &=2\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\big[Q_w(s,a)-Q_\pi(s,a)\big]\nabla_w Q_w(s,a)da\hspace{0.1cm}d s \\\\ &=2\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\big[Q_w(s,a)-Q_\pi(s,a)\big]\nabla_\theta\log\pi_\theta(a\vert s)da\hspace{0.1cm}d s \\\\ &=2\left(\int_\mathcal{S}\rho_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\log\pi_\theta(a\vert s)Q_w(s,a)da\hspace{0.1cm}d s-\nabla_\theta J(\pi_\theta)\right) \\\\ &=2\left(\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_w(s,a)\Big]-\nabla_\theta J(\pi_\theta)\right)
\end{align}
Moreover, property [(ii)](#prop-ii) claims that this gradient w.r.t $w$ must be zero due to the fact that $w$ minimizes $\epsilon^2(w)$. And thus, we obtain
\begin{equation}
\nabla_\theta J(\pi_\theta)=\mathbb{E}\_{\rho_\pi,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_w(s,a)\Big]
\end{equation}

### Off-policy Actor-Critic{#off-pac}
Consider off-policy methods, which learn a target policy $\pi_\theta$ using trajectories sampled according to a behavior policy $\beta(a\vert s)\neq\pi_\theta(a\vert s)$. In this setting, the performance objective is given as the value function of the target policy, averaged over $\beta$, as
\begin{align}
J_\beta(\pi_\theta)&=\int_\mathcal{S}\rho_\beta(s)V_\pi(s)d s \\\\ &=\int_\mathcal{S}\rho_\beta(s)\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s
\end{align}
The off-policy policy gradient then be given by utilizing importance sampling
\begin{align}
\nabla_\theta J_\beta(\pi_\theta)&=\nabla_\theta\int_\mathcal{S}\rho_\beta(s)\int_\mathcal{A}\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\rho_\beta(s)\int_\mathcal{A}\Big(\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)+\color{red}{\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)}\Big)da\hspace{0.1cm}d s\label{eq:offpac.1} \\\\ &\overset{\text{(i)}}{\approx}\int_\mathcal{S}\rho_\beta(s)\int_\mathcal{A}\nabla_\theta\pi_\theta(a\vert s)Q_\pi(s,a)da\hspace{0.1cm}d s \\\\ &=\int_\mathcal{S}\rho_\beta(s)\int_\mathcal{A}\pi_\theta(a\vert s)\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)d a\hspace{0.1cm}d s \\\\ &=\mathbb{E}\_{\rho_\beta,\pi_\theta}\Big[\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)\Big] \\\\ &=\mathbb{E}\_{\rho_\beta,\beta}\left[\frac{\pi_\theta(a\vert s)}{\beta(a\vert s)}\nabla_\theta\log\pi_\theta(a\vert s)Q_\pi(s,a)\right],
\end{align}
where to get the approximation in step (i), we have removed the $\color{red}{\pi_\theta(a\vert s)\nabla_\theta Q_\pi(s,a)}$ part in \eqref{eq:offpac.1}. This approximation is good enough to guarantee the policy improvement and eventually achieve the true local optima due to justification proofs proposed in [Off-PAC paper](#offpac-paper), which stands for **Off-policy Actor-Critic**.

## Deterministic Policy Gradient Theorem{#dpg-theorem}
Now let us consider the case of deterministic policy $\mu_\theta$, in which the performance objective function is also defined as the expected return of the start state. Thus we also have that
\begin{equation}
J(\mu_\theta)=\mathbb{E}\_{\rho_0,\mu_\theta}\big[r_0^\gamma\big]=\mathbb{E}\_{s_0\sim\rho_0}\big[V_\mu(s_0)\big]\label{eq:dpgt.1}
\end{equation}
The **Deterministic Policy Gradient Theorem** thus states that
\begin{align}
\nabla_\theta J(\mu_\theta)&=\int_\mathcal{S}\rho_\mu(s)\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\big\vert_{a=\mu_\theta(s)}d s \\\\ &=\mathbb{E}\_{\rho_\mu}\Big[\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\big\vert_{a=\mu_\theta(s)}\Big]\label{eq:dpgt.2}
\end{align}
where along with the assumption we have made in stochastic case, we additionally assume that $\nabla_a p(s'\vert s,a),\mu_\theta(s),\nabla_\theta\mu_\theta(s),\nabla_a r(s,a)$ are continues for all $\theta$ and $s,s'\in\mathcal{S},a\in\mathcal{A}$. These also imply the existence of $\nabla_a Q_\mu(s,a)$.

**Proof**  
This proof will be quite similar to what we have used in the stochastic case. Specifically, also starting with the gradient of the value function w.r.t $\theta$, we have
\begin{align}
&\hspace{-0.5cm}\nabla_\theta V_\mu(s)\nonumber \\\\ &\hspace{-0.5cm}=\nabla_\theta Q_\mu(s,\mu_\theta(s)) \\\\ &\hspace{-0.5cm}=\nabla_\theta\left[r(s,\mu_\theta(s))+\int_\mathcal{S}\gamma p(s'\vert s,\mu_\theta(s))V_\mu(s')d s'\right] \\\\ &\hspace{-0.5cm}=\nabla_\theta\mu_\theta(s)\nabla_a r(s,a)\vert_{a=\mu_\theta(s)}+\nabla_\theta\int_\mathcal{S}\gamma p(s'\vert s,\mu_\theta(s))V_\mu(s')d s' \\\\ &\hspace{-0.5cm}=\nabla_\theta\mu_\theta(s)\nabla_a r(s,a)\vert_{a=\mu_\theta(s)}\nonumber \\\\ &\hspace{1.5cm}+\int_\mathcal{S}\gamma\nabla_\theta\mu_\theta(s)\nabla_a p(s'\vert s,a)\vert_{a=\mu_\theta(s)}V_\mu(s')+\gamma p(s'\vert s,a)\nabla_\theta V_\mu(s')d s' \\\\ &\hspace{-0.5cm}=\nabla_\theta\mu_\theta(s)\nabla_a\Big(\underbrace{r(s,a)+\int_\mathcal{S}p(s'\vert s,a)V_\mu(s')d s'}\_{Q_\mu(s,a)}\Big)\Big\vert_{a=\mu_\theta(s)}+\int_\mathcal{S}\gamma p(s'\vert s,a)\nabla_\theta V_\mu(s')d s' \\\\ &\hspace{-0.5cm}=\underbrace{\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}}\_{\psi(s)}+\int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta V_\mu(s')d s' \\\\ &\hspace{-0.5cm}=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta V_\mu(s')d s'
\end{align}
which is in the same form as equation \eqref{eq:spgt.9}. Thus after repeated unrolling, we also end up with
\begin{align}
\nabla_\theta V_\mu(s)&=\psi(s)+\int_\mathcal{S}\gamma p(s\to s',1,\mu_\theta)\nabla_\theta V_\mu(s')d s' \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\mu_\theta)\psi(\tilde{s})d\tilde{s} \\\\ &=\int_\mathcal{S}\sum_{k=0}^{\infty}\gamma^k p(s\to\tilde{s},k,\mu_\theta)\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}d\tilde{s}\label{eq:dpgt.3}
\end{align}
Consider the definition of $J(\mu_\theta)$ given in \eqref{eq:dpgt.1}, taking gradient of both sides w.r.t $\theta$, combined with \eqref{eq:dpgt.3} gives us
\begin{align}
&\nabla_\theta J(\mu_\theta)\nonumber \\\\ &=\nabla_\theta\mathbb{E}\_{s_0\sim\rho_0}\big[V_\mu(s_0)\big] \\\\ &=\int_\mathcal{S}\rho_0(s_0)\nabla_\theta V_\mu(s_0)d s_0 \\\\ &=\int_{s_0\in\mathcal{S}}\rho_0(s_0)\int_{s\in\mathcal{S}}\sum_{k=0}^{\infty}\gamma^k p(s_0\to s,k,\mu_\theta)\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}d s\hspace{0.1cm}d s_0 \\\\ &=\int_{s\in\mathcal{S}}\left(\int_{s_0\in\mathcal{S}}\sum_{k=0}^{\infty}\gamma^k\rho_0(s_0)p(s_0\to s,k,\mu_\theta)d s_0\right)\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}d s \\\\ &=\int_\mathcal{S}\rho_\mu(s)\nabla_\theta\mu_\theta(s)\nabla a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}d s,
\end{align}
where in the forth step, the **Fubini's theorem** has helped us exchange the order of integration.$\tag*{$\Box$}$

It is worth remarking that we can consider a deterministic policy $\mu_\theta$ as a special case of the stochastic policy, in which $\pi_\theta(\cdot\vert s)$ becomes the **Kronecker delta function**, which takes the value of $1$ at only one point $a\in\mathcal{A}$ and $0$ elsewhere.

To be more specific, in the [original paper](#dpg-paper), the authors have shown that by rewriting the stochastic policy as $\pi_{\mu_\theta,\sigma}$, which is parameterized by a deterministic policy $\mu_\theta:\mathcal{S}\to\mathcal{A}$ and a variance parameter $\sigma$ such that for $\sigma=0$, we have that $\pi_{\mu_\theta,0}\equiv\mu_\theta$; then as $\sigma\to 0$, they have proved that the stochastic policy gradient converges to the deterministic one.

This critical result allows us to apply the deterministic policy gradient to common policy gradient frameworks, for example actor-critic approaches.

## Deterministic Actor-Critic{#deterministic-ac}

### On-policy Deterministic Actor-Critic{#on-policy-deterministic-ac}
Analogous to the stochastic approach, the deterministic actor learns a parameter $\theta$ by using SGA to iteratively update the parameter vector according to the deterministic policy gradient direction \eqref{eq:dpgt.2} while the critic estimates the state-action value function by a function approximation $Q_w(s,a)$ using a policy evaluation method such as [TD-learning]({{<ref"td-learning">}}).

For instance, a deterministic actor-critic method with a [Sarsa]({{<ref"td-learning#sarsa">}}) critic has the following update in each time-step $t$
\begin{align}
\delta_t&=r_{t+1}+\gamma Q_w(s_{t+1},a_{t+1})-Q_w(s_t,a_t) \\\\ w_{t+1}&=w_t+\alpha_w\delta_t\nabla_w Q_w(s_t,a_t) \\\\ \theta_{t+1}&=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_a Q_w(s_t,a_t)\vert_{a=\mu_\theta(s)},
\end{align}
where $\delta_t$ as specified before, are known as TD errors.

### Off-policy Deterministic Actor-Critic{#off-policy-deterministic-ac}
Analogy to stochastic methods, let $\beta(a\vert s)$ denote the behavior policy that generates trajectories used for updating the deterministic target policy $\mu_\theta(s)$, the performance objective $J(\mu_\theta)$ is then given as
\begin{align}
J_\beta(\mu_\theta)&=\int_\mathcal{S}\rho_\beta(s)V_\mu(s)d s \\\\ &=\int_\mathcal{S}\rho_\beta(s)Q_\mu(s,\mu_\theta(s))d s
\end{align}
It is noticeable that the deterministic policy allows us to explicitly replace the integration over action space $\mathcal{A}$ by $Q_\mu(s,\mu_\theta(s))$, and thus we do not need to use importance sampling in the actor. Hence, we have that
\begin{align}
\nabla_\theta J_\beta(\mu_\theta)&=\nabla_\theta\int_\mathcal{S}\rho_\beta(s)Q_\mu(s,\mu_\theta(s))d s \\\\ &\approx\int_\mathcal{S}\rho_\beta(s)\nabla_\theta\mu_\theta(a\vert s)Q_\mu(s,a)d s \\\\ &=\mathbb{E}\_{\rho_\beta}\Big[\nabla_\theta\mu_\theta(s)\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}\Big]
\end{align}
We can also avoid using importance sampling in critic by using Q-learning 
as our policy evaluation In particular, the off-policy deterministic actor-critic with a Q-learning critic has the form of
\begin{align}
\delta_t&=r_{t+1}+\gamma Q_w(s_{t+1},\mu_\theta(s))-Q_w(s_t,a_t)\label{eq:opdac.1} \\\\ w_{t+1}&=w_t+\alpha_w\delta_t\nabla_w Q_w(s_t,a_t) \\\\ \theta_{t+1}&=\theta_t+\alpha_\theta\nabla_\theta\mu_\theta(s_t)\nabla_a Q_w(s_t,a_t)\vert_{a_t=\mu_\theta(s_t)},
\end{align}
where the greedy policy, $\underset{a}{\text{argmax}}Q_w(s,a)$, in the usual Q-learning update has been replaced by the newly-updated deterministic policy, $\mu_\theta(s)$, in \eqref{eq:opdac.1}, i.e. $\mu_\theta\equiv\mu_{\theta_k}$.

### Compatible Function Approximation with Deterministic Policy{#compatible-func-approx-deterministic}
From what we have mentioned in the stochastic case, we can also define an appropriate form of function approximation $Q_w$ which preserves the direction of true gradient.

In particular, A $w$-parameterized $Q_w(s,a)$ is referred as a **compatible function approximator** of the state-action value function $Q_\mu$ for deterministic policy $\mu_\theta$ in the sense that
<ul class='roman-list'>
	<li>
		<span id='prop-i-det'>$\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}=\nabla_\theta\mu_\theta(s)^\text{T}w$.</span>
	</li>
	<li>
		<span id='prop-ii-det'>Parameters $w$ minimize the mean-squared error</span>
		\begin{equation}
		\text{MSE}(\theta,w)=\mathbb{E}\big[\epsilon(\theta,w,s)^\text{T}\epsilon(\theta,w,s)\big],
		\end{equation}
		where
		\begin{equation}
		\epsilon(\theta,w,s)\doteq\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}-\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}
		\end{equation}
	</li>
</ul>

then
\begin{equation}
\nabla_\theta J(\mu_\theta)=\mathbb{E}\big[\nabla_\theta\mu_\theta(s)\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}\big]\label{eq:cfad.1}
\end{equation}

**Proof**  
We follow the procedure used in stochastic case. Specifically, starting with the property [(i)](#prop-i-det) and by definition of $\epsilon$, we have that
\begin{align}
\nabla_w\epsilon(\theta,w,s)&=\nabla_w\big[\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}-\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}\big] \\\\ &=\nabla_w\big(\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}\big) \\\\ &=\nabla_w\big(\nabla_\theta\mu_\theta(s)^\text{T}w\big) \\\\ &=\nabla_\theta\mu_\theta(s)
\end{align}
Using this result, the gradient of $\text{MSE}(\theta,w)$ w.r.t $w$ is thus given as
\begin{align}
\nabla_w\text{MSE}(\theta,w)&=\nabla_w\mathbb{E}\big[\epsilon(\theta,w,s)^\text{T}\epsilon(\theta,w,s)\big] \\\\ &=\mathbb{E}\big[2\epsilon(\theta,w,s)\nabla_w\epsilon(\theta,w,s)\big] \\\\ &=2\mathbb{E}\Big[\big(\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}-\nabla_a Q_\mu(s,a)\vert_{a=\mu_\theta(s)}\big)\nabla_\theta\mu_\theta(s)\Big] \\\\ &=2\Big[\mathbb{E}\big[\nabla_\theta\mu_\theta(s)\nabla_a Q_w(s,a)\vert_{a=\mu_\theta(s)}\big]-J(\mu_\theta)\Big],
\end{align}
which lets our claim, \eqref{eq:cfad.1}, follows due to the property [(ii)](#prop-ii-det), which means that $\nabla_w\text{MSE}(\theta,w)=0$.$\tag*{$\Box$}$

## Deep Deterministic Policy Gradient{#ddpg}


## References
[1] <span id='dpg-paper'>David Silver, Guy Lever, Nicolas Heess, Thomas Degris, Daan Wierstra, Martin Riedmiller. [Deterministic Policy Gradient Algorithms](http://proceedings.mlr.press/v32/silver14.pdf). JMLR 2014</span>.

[2] Richard S. Sutton, Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

[3] Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour. [Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). NIPS 1999.

[4] Elias M. Stein, Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). Princeton University Press, 2007.

[5] <span id='offpac-paper'>Thomas Degris, Martha White, Richard S. Sutton. [Off-Policy Actor-Critic](https://icml.cc/2012/papers/268.pdf). ICML 2012</span>.

[6] Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra. [Continuous control with deep reinforcement learning](https://arxiv.org/pdf/1509.02971.pdf). ICLR 2016.

## Footnotes
[^1]: To simplify the notation, we have let $\rho_\pi\doteq\rho\_{\pi_\theta}$ and $Q_\pi\doteq Q_{\pi_\theta}$ implicitly. As a result, we will also denote $V_{\pi_\theta}$ by $V_\pi$.
[^2]: The notation $r(\pi)$ of the **average reward** is just a notation-abused and should not be confused with notation $r$ of the reward function.
[^3]: This means that if we keep selecting action according to $\pi_\theta$, we remains in the same state distribution $\bar{\rho}\_\pi$, i.e.
\begin{equation}
\int_\mathcal{S}\bar{\rho}\_\pi(s)\int_\mathcal{A}\pi_\theta(a\vert s)p(s'\vert s,a)da\hspace{0.1cm}d s=\bar{\rho}\_\pi(s')\label{eq:fn.1}
\end{equation}
