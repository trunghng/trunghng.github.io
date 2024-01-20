---
title: "MuZero"
date: 2024-01-02T11:52:40+07:00
tags: [deep-reinforcement-learning, model-based, mcts, my-rl]
math: true
eqn-number: true
hideSummary: true
---

## MuZero
Prediction are made at each time step $t$, for each of $k=0,\ldots,K$ steps, by a model $\mu_\theta$, parameterized by $\theta$, conditioned on past observations $o_1,\ldots,o_t$ and on future actions $a_{t+1},\ldots,a_{t+k}$ for $k>0$.<br>
The model $\mu_\theta$ predicts three futures quantities:
<ul id='number-list'>
	<li>
		the policy $p_t^k\approx\pi(a_{t+k+1}\vert o_1,\ldots,o_t,a_{t+1},\ldots,a_{t+k})$;
	</li>
	<li>
		the value function $v_t^k\approx\mathbb{E}\big[u_{t+k+1}+\gamma u_{t+k+2}+\ldots\vert o_1,\ldots,o_t,a_{t+1},\ldots,a_{t+k}\big]$;
	</li>
	<li>
		the immediate reward $r_t^k\approx u_{t+k}$,
	</li>
</ul>

where $u$ is the true, observed reward, $\pi$ is the policy used to select real actions and $\gamma$ is the discount function of the environment.

### Planning
Internally, at each time step $t$, the model is represented by the combination of a representation function, a dynamics function and a prediction function.
<ul id='number-list'>
	<li>
		The (deterministic) <b>dynamics function</b> $g_\theta$, is a recurrent process, $(r_t^k,s_t^k)=g_\theta(s_t^{k-1},a_t^k)$, that computes, at each hypothetical step $k$, an immediate reward $r_t^k$ and an internal state $s_t^k$.
	</li>
	<li>
		The <b>prediction function</b> $f_\theta$ computes the policy and value functions from the internal state $s^k$, $(p_t^k,v_t^k)=f_\theta(s_t^k)$, similar to the joint policy and value network of <a href={{<ref"alphazero#neural-network-training">}}>AlphaZero</a>.
	</li>
	<li>
		The <b>representation function</b> $h_\theta$ initializes the root state $s_t^0$ by encoding past observations, $s_t^0=h_\theta(o_1,\ldots,o_t)$.
	</li>
</ul>

### Acting
Given such a model, it is possible to search over hypothetical future trajectories $a^1,\ldots,a^k$ given past observations $o_t,\ldots,o_t$. Generally, we can apply any MDP planning algorithm (MCTS, dynamic programing, etc) to the internal rewards and state space induced by the dynamics function.

MuZero uses an MCTS algorithm similar to [AlphaZero]({{<ref"alphazero#search-algorithm">}})'s search (with generalized to single-agent and nonzero intermediate reward).

#### Search algorithm
Every node in the search tree is associated with an internal state $s$. Corresponding to each action $a$ from $s$ is an edge $(s,a)$ storing a set of statistics
\begin{equation}
\\{N(s,a),P(s,a),Q(s,a),R(s,a),S(s,a)\\}
\end{equation}
respectively representing visit count $N$, policy $P$, mean value $Q$, reward $R$ and state transition $S$. Analogous to AlphaZero, the algorithm iterates over three stages for a number of simulations.
<ul id='number-list'>
	<li>
		<b>Selection</b>. The stage begins at the internal root state $s^0$ and finishes when reaching a leaf node $s^l$. For each hypothetical time step $k=1,\ldots,l$ of the simulation, an action $a^k$ that maximizes over a PUCT bound is selected according to the statistics stored at $s^{k-1}$.
		\begin{equation}
		a^k=\underset{a}{\text{argmax}}\big(\bar{Q}(s^{k-1},a)+U(s^{k-1},a)\big),
		\end{equation}
		where
		\begin{align}
		\bar{Q}(s^{k-1},a)&=\frac{Q(s^{k-1},a)-\underset{s',a'\in\text{Tree}}{\min}Q(s',a')}{\underset{s',a'\in\text{Tree}}{\max}Q(s',a')-\underset{s',a'\in\text{Tree}}{\min}Q(s',a')} \\ U(s^{k-1},a)&=c_\text{puct}P(s^{k-1},a)\frac{\sqrt{\sum_b N(s^{k-1},b)}}{1+N(s^{k-1},a)} \\ c_\text{puct}&=c_1+\log\frac{\sum_b N(s^{k-1},b)+c_2+1}{c_2},
		\end{align}
		where $c_1$ and $c_2$ are constants controlling the influence of the policy $P(s^{k-1},a)$ relative to the value $Q(s^{k-1},a)$ as nodes are visited more often.<br>
		For $k\lt l$, the next state and reward are looked up in the state transition and reward table
		\begin{align}
		s^k&=S(s^{k-1},a^k) \\ r^k&=R(s^{k-1},a^k)
		\end{align}
	</li>
	<li>
		<b>Expansion</b>. At the final step $l$ of the simulation, the reward and state are computed by the dynamics function $g_\theta$.
		\begin{equation}
		(r^l,s^l)=g_\theta(s^{l-1},a^l)
		\end{equation}
		and stored in the corresponding tables
		\begin{align}
		R(s^{l-1},a^l)&=r^l \\ S(s^{l-1},a^l)&=s^l
		\end{align}
		The policy and value function are computed by the prediction function $f_\theta$.
		\begin{equation}
		(p^l,v^l)=f_\theta(s^l)
		\end{equation}
		A new node, corresponding to the state $s^l$ is added to the search tree and each edge $(s^l,a)$ is initialized to
		\begin{equation}
		\{N(s^l,a)=0,Q(s^l,a)=0,P(s^l,a)=p^l\}
		\end{equation}
	</li>
	<li>
		<b>Backup</b>. For $k=l,\ldots,0$, we form an $l-k$-step estimate of the cumulative discounted reward, bootstrapping from the value function $v^l$.
		\begin{equation}
		G^k=\sum_{\tau=0}^{l-1-k}\gamma^\tau r_{k+1+\tau}+\gamma^{l-k}v^l
		\end{equation}
		The edge statistics are then updated in a backward pass through each step $k\leq l$. Specifically, for $k=l,\ldots,1$, the statistics corresponding to each edge $(s^{k-1},a^k)$ in the simulation path are updated as
		\begin{align}
		Q(s^{k-1},a^k)&\leftarrow\frac{N(s^{k-1},a^k)Q(s^{k-1},a^k)+G^k}{N(s^{k-1},a^k)+1} \\ N(s^{k-1},a^k)&\leftarrow N(s^{k-1},a^k)+1
		\end{align}
	</li>
</ul>

The MCTS can be viewed as a pair of a search policy and a search value function $(\pi_t,\nu_t)$
\begin{align}
\pi_t&=Pr(a_{t+1}\vert o_1,\ldots,o_t) \\\\ \nu_t&\approx\mathbb{E}\big[u_{t+1}+\gamma u_{t+2}+\ldots\vert o_1,\ldots,o_t\big]
\end{align}
that both selects an action and predicts cumulative discounted reward given past observations $o_1,\ldots,o_t$. At each internal node, it makes use of the policy $p$, value function $v$ and reward estimate $G$ produced by the current model parameter $\theta$, and combines these values together using lookahead search to produce an improved policy $\pi_t$ and improved value function $\nu_t$ at the root of the search tree.

The next action $a_{t+1}$ is then selected according to the search policy $\pi_t$. With the action received, the environment generates a new observation $o_{t+1}$ and reward $u_{t+1}$. At the end of episode, the trajectory data are stored into a replay buffer.

### Training
The model training in MuZero proceeds as
<ul id='number-list'>
	<li></li>
</ul>

<figure>
	<img src="/images/muzero/muzero-planning-acting-training.png" alt="planning, acting, training with a learned model" width="80%" height="80%"/>
	<figcaption style="text-align: center"><b>Figure 1</b>: (taken from <a href='#muzero-paper'>MuZero paper</a>) <b>Planning, acting, training with a learned model</b></figcaption>
</figure>

## Sampled MuZero

## Stochastic MuZero

## References
[1] <span id='muzero-paper'>Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, David Silver et al. [Mastering Atari, Go, chess and shogi by planning with a learned model](https://doi.org/10.1038/s41586-020-03051-4). Nature 588, 604–609, 2020.</span>

[2] <span id='sampled-muzero-paper'>Thomas Hubert, Julian Schrittwieser, Ioannis Antonoglou, Mohammadamin Barekatain, Simon Schmitt, David Silver. [Learning and Planning in Complex Action Spaces](http://proceedings.mlr.press/v139/hubert21a/hubert21a.pdf). ICML, 2021.</span>

[3] <span id='stochastic-muzero-paper'>Ioannis Antonoglou, Julian Schrittwieser, Sherjil Ozair, Thomas Hubert, David Silver. [Planning in stochastic environments with a learned model](https://openreview.net/forum?id=X6D9bAHhBQ1). ICLR, 2022.</span>

[4] <span id='gumbel-muzero-paper'>Ivo Danihelka, Arthur Guez, Julian Schrittwieser, David Silver. [Policy improvement by planning with Gumbel](https://openreview.net/forum?id=bERaNdoegnO). ICLR, 2022.</span>

[5] Richard S. Sutton, Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

## Footnotes
