---
title: "Temporal consistency loss & Ape-X DQfD"
date: 2024-03-12T18:39:34+07:00
tags: [reinforcement-learning, model-free, dqn, my-rl]
math: true
eqn-number: true
---
An algorithm consists of three components: the transformed Bellman operator, the temporal consistency (TC) loss and the combination of Ape-X DQN and DQfD to learn a more consistent human-level policy.
<!--more-->

### DQN
Consider a finite, discrete-time MDP $(\mathcal{S},\mathcal{A},R,P,\gamma)$ where $\mathcal{S}$ is the state space, $\mathcal{A}$ the action space, $r:\mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}$ the reward function, $P$ the transition probability and $\gamma\in[0,1]$ the discount factor. To measure how good a policy $\pi$ is, we use the state value function, $V^\pi:\mathcal{S}\mapsto\mathbb{R}$, and state-action value function, $Q^\pi:\mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}$, defined as
\begin{align}
V^\pi(s)&=\mathbb{E}\_\pi\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\Big\vert s_0=s\right] \\\\ Q^\pi(s,a)&=\mathbb{E}\_\pi\left[\sum_{t=0}^{\infty}\gamma^t r(s_t,a_t)\Big\vert s_0=s,a_0=a\right]
\end{align}
Recall that a policy $\pi$ is better than some $\pi'$ if for all $s\in\mathcal{S}$, we have that $V^\pi(s)\geq V^{\pi'}(s)$. Our goal is to find an optimal policy $\pi^\*$ that maximizes the state value function, $V^\pi$.
\begin{equation}
V^{\pi^\*}\geq\sup_\pi V^\pi(s),\hspace{1cm}\forall s\in\mathcal{S}
\end{equation}
Such policy [always exists]({{< ref "optimal-policy-existence" >}}). Also, [recall]({{< ref "mdp-bellman-eqn#opt-vf" >}}) that while there might be more than one optimal policy, they all share the same state and state-action value functions, known as optimal state value function and optimal state-action value function, denoted $V^\*$ and $Q^\*$. These functions are connected via the following equations
\begin{align}
V^\*(s)&=\sup_{a\in\mathcal{A}}Q^\*(s,a),\hspace{4.9cm}\forall s\in\mathcal{S} \\\\ Q^\*(s,a)&=r(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\vert s,a)V^\*(s'),\hspace{1cm}\forall s\in\mathcal{S},a\in\mathcal{A}
\end{align}
The optimal state and state-action value function satisfy the fixed-point equation, here we consider $Q^*$ only
\begin{equation}
Q^\*(s,a)=r(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\vert s,a)\sup_{a'\in\mathcal{A}}Q(s',a')\hspace{1cm}\forall(s,a)\in\mathcal{S}\times\mathcal{A}
\end{equation}
Rewriting the above equation with the Bellman optimality operator, $\mathcal{T}:\mathbb{R}^{\mathcal{S}\times\mathcal{A}}\mapsto\mathbb{R}^{\mathcal{S}\times\mathcal{A}}$, we have that $Q^\*$ is the unique solution to
\begin{equation}
(\mathcal{T}Q)(s,a)=r(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\vert s,a)\sup_{a'\in\mathcal{A}}Q(s',a'),\hspace{1cm}\forall(s,a)\in\mathcal{S}\times\mathcal{A}
\end{equation}
for any $Q:\mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}$. Since $\mathcal{T}$ is a [$\gamma$-contraction]({{< ref "optimal-policy-existence#bellman-op-contraction" >}}), we can learn $Q^\*$ using a fixed point iteration, starting at some initial $Q^{(0)}$ and then iterating $Q^{(k)}\doteq\mathcal{T}Q^{(k-1)}$ for $k\in\mathbb{N}$ to generate a sequence of functions that converge to $Q^\*$.

DQN uses a neural network $f_\theta$ as a function approximator of the optimal state-action value function $Q^\*$. It starts with some initial weight $\theta^{(0)}$ and then iterates
\begin{equation}
\theta^{(k)}\doteq\underset{\theta}{\text{argmin}}\hspace{0.1cm}\mathbb{E}\_{s,a}\big[\mathcal{L}(f_\theta(s,a))-(\mathcal{T}f_{\theta^{(k-1)}})(s,a)\big],
\end{equation}
where the expectation is taken w.r.t a random sample of states and actions and $\mathcal{L}:\mathbb{R}\mapsto\mathbb{R}$ is the Huber loss, defined as
\begin{equation}
\mathcal{L}(x)=\begin{cases}\frac{1}{2}x^2&\text{if }\vert x\vert\geq 1 \\\\ \vert x\vert-\frac{1}{2}&\text{otherwise}\end{cases}
\end{equation}

### Transformed Bellman operator
In the original DQN paper, the authors clip the reward distribution to the interval $[-1,1]$ to reduce the variance of the optimization target $\mathcal{T}f_{\theta^{(k-1)}}$. This trick, however, changes the set of optimal policies. We can instead use a function $h:\mathbb{R}\mapsto\mathbb{R}$ that reduces the scale of the state-action value function. The new operator $\mathcal{T}\_h$ is defined as, for all $(s,a)\in\mathcal{S}\times\mathcal{A}$
\begin{align}
(\mathcal{T}\_h Q)(s,a)&=h\left(r(s,a)+\gamma\sum_{s'\in\mathcal{S}}P(s'\vert s,a)\sup_{a'\in\mathcal{A}}h^{-1}\big(Q(s',a'))\right) \\\\ &=\mathbb{E}\_{s'\sim P(\cdot\vert s,a)}\Bigg[h\Big(r(s,a)+\gamma\max_{a'\in\mathcal{A}}h^{-1}\big(Q(s',a')\big)\Big)\Bigg]
\end{align}

**Proposition 1**. *Let $Q^\*$ be the fixed point of $\mathcal{T}$ and $Q:\mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}$, then*
<ul id='roman-list' style='font-style: italic'>
	<li>
		If $h(z)=\alpha z$ for $\alpha>0$, then $\mathcal{T}_h^k Q\xrightarrow[]{\hspace{0.1cm}k\to\infty\hspace{0.1cm}}h\circ Q^*=\alpha Q^*$.
	</li>
	<li>
		If $h$ is strictly monotonically increasing and the MDP is deterministic (i.e., $P(\cdot\vert s,a)$ and $r(s,a)$ are point measures for all $s,a\in\mathcal{S}\times\mathcal{A}$), then $\mathcal{T}_h^k Q\xrightarrow[]{\hspace{0.1cm}k\to\infty\hspace{0.1cm}}h\circ Q^*$.
	</li>
</ul>

<!-- **Proof**
<ul id='roman-list'>
	<li>
		When $h(z)=\alpha z$, we have that
		\begin{align}
		(\mathcal{T}_h Q)(s,a)&=\mathbb{E}_{s'\sim P}\Bigg[\alpha\Bigg(r(s,a)+\frac{\gamma}{\alpha}\max_{a'\in\mathcal{A}}Q(s',a')\Bigg)\Bigg] \\
		\end{align}
	</li>
</ul> -->


The proposition shows that when $h$ is either linear or the MDP is deterministic, $\mathcal{T}\_h$ has the unique fixed point $h\circ Q^\*$. Hence, if $h$ is an invertible contraction, we can use $\mathcal{T}\_h$ instead of $\mathcal{T}$ in the DQN algorithm to reduce the variance of our optimization target while still learning an optimal policy.

**Proposition 2**. *Let $h$ be strictly monotonically increasing, Lipschitz continuous with Lipschitz constant $L_h$, and have a Lipschitz continuous inverse $h^{-1}$ with Lipschitz constant $L_{h^{-1}}$. For $\gamma<\frac{1}{L_h L_{h^{-1}}}$, $\mathcal{T}\_h$ is a contraction*.

**Proof**  
We first recall the Lipschitz property that if $f:\mathbb{R}\mapsto\mathbb{R}$ are Lipschitz continuous with Lipschitz constants $K$, we have for all $x_1,x_2\in\mathbb{R}$
\begin{equation}
\vert f(x_1)-f(x_2)\vert\leq K\vert x_1-x_2\vert\label{eq:tbo.1}
\end{equation}
Consider $Q_1,Q_2:\mathcal{S}\times\mathcal{A}\mapsto\mathbb{R}$ arbitrary, we have
\begin{align}
\hspace{-1.3cm}\Vert\mathcal{T}\_hQ_1-\mathcal{T}\_hQ_2\Vert_\infty&=\max_{s,a\in\mathcal{S}\times\mathcal{A}}\left\vert\mathbb{E}\_{s'\sim P(\cdot\vert s,a)}\left[h\left(r(s,a)+\gamma\max_{a'\in\mathcal{A}}h^{-1}\big(Q_1(s',a')\big)\right)\right.\right.\nonumber \\\\ &\hspace{1cm}\left.\left.-h\left(r(s,a)+\gamma\max_{a'\in\mathcal{A}}h^{-1}\big(Q_2(s',a')\big)\right)\right]\right\vert \\\\ &\overset{\text{(i)}}{\leq}\max_{s,a\in\mathcal{S}\times\mathcal{A}}\mathbb{E}\_{s'\sim P}\left[\left\vert h\left(r(s,a)+\gamma\max_{a'\in\mathcal{A}}h^{-1}\big(Q_1(s',a')\big)\right)\right.\right.\nonumber \\\\ &\hspace{1cm}\left.\left.-h\left(r(s,a)+\gamma\max_{a'\in\mathcal{A}}h^{-1}\big(Q_2(s',a')\big)\right)\right\vert\right] \\\\ &\overset{\text{(ii)}}{\leq}L_h\gamma\max_{s,a\in\mathcal{S}\times\mathcal{A}}\mathbb{E}\_{s'\sim P}\left[\left\vert\max_{a'\in\mathcal{A}}h^{-1}\big(Q_1(s',a')\big)-\max_{a'\in\mathcal{A}}h^{-1}\big(Q_2(s',a')\big)\right\vert\right] \\\\ &\leq L_h\gamma\max_{s,a\in\mathcal{S}\times\mathcal{A}}\mathbb{E}\_{s'\sim P}\left[\max_{a'\in\mathcal{A}}\Big\vert h^{-1}\big(Q_1(s',a')\big)-h^{-1}\big(Q_2(s',a')\big)\Big\vert\right] \\\\ &\overset{\text{(iii)}}{\leq} L_h L_{h^{-1}}\gamma\max_{s,a\in\mathcal{S}\times\mathcal{A}}\mathbb{E}\_{s'\sim P}\Big[\max_{a'\in\mathcal{A}}\Big\vert Q_1(s',a')-Q_2(s',a')\Big\vert\Big] \\\\ &\leq L_h L_{h^{-1}}\gamma\Vert Q_1-Q_2\Vert_\infty\lt\Vert Q_1-Q_2\Vert_\infty,
\end{align}
which implies that $\mathcal{T}\_h$ is a contraction. Note that in the above derivation, specifically we used Jensen's inequality in (ii) and the Lipschitz property \eqref{eq:tbo.1} of $h$ and $h^{-1}$ in (i) and (iii).

In the [Ape-X DQfD paper](#apex-dqfd-paper), the authors choose
\begin{equation}
h(z)=\text{sign}(z)\left(\sqrt{\vert z\vert+1}-1\right)+\epsilon z,
\end{equation}
with $\epsilon=0.01$. This function can be proved that strictly monotonically increasing, Lipschitz continuous and so is its closed form inverse.

### Temporal consistency loss

### References
[1] <span id='apex-dqfd-paper'>Tobias Pohlen et al. [Observe and Look Further: Achieving Consistent Performance on Atari](https://arxiv.org/abs/1805.11593). arXiv preprint, arXiv:1805.11593, 2018.</span>

[2] Csaba Szepesvári. [Algorithms for Reinforcement Learning](http://dx.doi.org/10.2200/S00268ED1V01Y201005AIM009). Synthesis Lectures on Artificial Intelligence and Machine Learning 4, 2010.

### Footnotes