---
title: "AlphaZero"
date: 2024-01-02T10:23:22+07:00
tags: [deep-reinforcement-learning, my-rl]
math: true
eqn-number: true
hideSummary: true
---
### AlphaGo
The training pipeline used in **AlphaGo** can be divided into following stages:
<ul id='number-list'>
	<li>
		Using a dataset of human experts positions, a <b>supervised learning (SL) policy network</b> $p_\sigma$ and, a <b>rollout policy</b> $p_\pi$, which can sample actions rapidly, are trained by classification to predict player moves.
	</li>
	<li>
		Initializing with the <b>SL policy network</b> $p_\sigma$, it uses policy gradient to train a <b>reinforcement learning (RL) policy network</b> $p_\rho$ with the goal to maximize the winning outcome against previous versions of the policy network. This process generates a dataset of self-play games.
	</li>
	<li>
		Via the dataset of self-play moves, a <b>value network</b> $v_\theta$ is trained by regression to predict the expected outcome (win or lose).
	</li>
</ul>

<figure>
	<img src="/images/alphazero/alphago-training-pipeline.png" alt="AlphaGo training pipeline" width="80%" height="80%"/>
	<figcaption style="text-align: center"><b>Figure 1</b>: (taken from <a href='#alphago-paper'>AlphaGo paper</a>) <b>AlphaGo neural network training pipeline</b></figcaption>
</figure>

#### SL policy network $p_\sigma$, rollout network $p_\pi$
The policy network $p_\sigma(a\vert s)$ takes as its input a simple representation of the board state $s$ and outputs a probability distribution over all legal moves $a$. The network is trained to maximize the likelihood of the human move $a$ selected in state $s$ by using SGA
\begin{equation}
\Delta\sigma\propto\frac{\partial\log p_\sigma(a\vert s)}{\partial\sigma}
\end{equation}
The rollout network $p_\pi(a\vert s)$ is trained using a linear softmax of small pattern features. This network is less accurate but faster selecting action than $p_\sigma$.

#### RL policy network $p_\rho$
The RL policy network $p_\rho$ has the same architecture as $p_\sigma$ and its weights are also initialized at $\rho=\sigma$.

The games are between the current policy $p_\rho$ and a randomly selected previous iteration $p_{\rho^-}$ of its, which prevents overfitting to the current policy. The outcome $z_t=\pm r(s_T)$ of each game is the terminal reward at the end of the game from the perspective, where $r(s)$ is the reward function which is zero for all non-terminal step $t<T$, i.e. $z_t=+1$ for winning and $-1$ for losing.

The weights $\rho$ are updated by SDA in the direction that maximizes expected outcome
\begin{equation}
\Delta\rho\propto\frac{\partial\log p_\rho(a_t\vert s_t)}{\partial\rho}z_t
\end{equation}

#### Value network $v_\theta$
For each RL policy network $p$, we define its corresponding value function to be
\begin{equation}
v^{p}(s)=\mathbb{E}\big[z_t\vert s_t=s,a_{t\ldots T}\sim p\big]
\end{equation}
Ideally, we wish for the optimal policy $p^\*$ with value function $v^\*$. In tabular problem, this can be done via [dynamic programming]({{< ref "dp-in-mdp#policy-iter" >}}). However, in large state space problem like Go, we instead use an estimation of $v^{p_\rho}$. Specifically, we approximate the value function using a value network $v_\theta(s)$ parameterized by $\theta$, $v_\theta(s)\approx v^{p_\rho}(s)\approx v^*(s)$.

This network has a similar structure to the policy network $p_\rho$, but outputs a scalar value instead of a probability distribution. The weights $\theta$ is updated via SGD in the direction that minimizes the MSE between the predicted value $v_\theta(s)$ and the corresponding outcome $z$ by using regression on state-outcome pair $(s,z)$[^1]
\begin{equation}
\Delta\theta\propto\frac{\partial v_\theta(s)}{\partial\theta}(z-v_\theta(s))
\end{equation}

#### Searching with policy and value networks
Once trained, the policy network $p_\rho$ and value network $v_\theta$ then are combined with a **Monte Carlo tree search** (**MCTS**) to provide a lookahead search.
<figure>
	<img src="/images/alphazero/alphago-mcts.png" alt="MCTS in AlphaGo" width="100%" height="100%"/>
	<figcaption style="text-align: center"><b>Figure 2</b>: (taken from <a href='#alphago-paper'>AlphaGo paper</a>) <b>MCTS in AlphaGo</b></figcaption>
</figure>

Each node $s$ in the search tree contains edges $(s,a)$ for all legal action $a\in\mathcal{A}(s)$. Each edges stores a set of statistics
\begin{equation}
\\{P(s,a),N_v(s,a),N_r(s,a),W_v(s,a),W_r(s,a),Q(s,a)\\}
\end{equation}

### AlphaGo Zero
**AlphaGo Zero** differs from its predecessor, AlphaGo, in various aspects
<ul id='roman-list'>
	<li>
		It is trained only via self-play reinforcement learning, starting from random play, without supervised.
	</li>
	<li>
		It uses only the black and white stones from the board as input features.
	</li>
	<li>
		It uses a single neural network, rather than separate policy and value networks.
	</li>
	<li>
		It uses a simpler tree search to evaluate positions and sample moves, without doing any MC rollouts.
	</li>
</ul>

#### Self-play algorithm
The self-play mechanism used by AlphaGo Zero proceeds as:
- The AlphaGo Zero agent plays a game $s_1,\ldots,s_T$ against itself.
- In each position $s_t$, an MCTS $\alpha_\theta$ is executed using the latest neural network $f_\theta$.
- Moves are selected according to the search probabilities outputed by the the MCTS, $a_t\sim\boldsymbol{\pi}_t$.
- The terminal position $s_T$ is scored according to the rules of the game to compute the game winner $z$.
<figure>
	<img src="/images/alphazero/alphagozero-selfplay.png" alt="Self-play in AlphaGo Zero" width="70%" height="70%"/>
	<figcaption style="text-align: center"><b>Figure 3</b>: (taken from <a href='#alphagozero-paper'>AlphaGo Zero paper</a>) <b>Self-play reinforcement learning in AlphaGo Zero</b></figcaption>
</figure>

The self-play algorithm in AlphaGo Zero can be understood as a [policy iteration]({{< ref "dp-in-mdp#policy-iter" >}}) procedure. Indeed, in each position $s$, an MCTS $\alpha_\theta$ is performed guided by the neural network $f_\theta$. The move probabilities $\boldsymbol{\pi}$ computed by $\alpha_\theta$ is usually much effective than the one, $\mathbf{p}$, produced by $f_\theta$ (i.e. selecting stronger moves). Thus, MCTS then can be observed as a [policy improvement]({{< ref "dp-in-mdp#policy-imp" >}}) operator. Moreover, self-play with search, using the improved policy $\boldsymbol{\pi}$ to select move, then using the game winner $z$ as a sample of the value, might be observed as a [policy evaluation]({{< ref "dp-in-mdp#policy-eval" >}}) operator.

The neural network's parameters $\theta$ are updated to make the move probabilities and value $(\mathbf{p},v)$ more closely match the improved search probabilities and self-play winner $(\boldsymbol{\pi},z)$. These new parameters are used in the next iteration of self-play to make the search even stronger.

#### Neural network training
AlphaGo Zero utilizes a neural network $f_\theta$ parameterized by $\theta$. This network takes the board position $s$ as an input and outputs a vector of move probabilities $\mathbf{p}$ with components $p_a=Pr(a\vert s)$ for each action $a$ and a scalar value $v$ estimating the expected outcome $z$ of the game from position $s$, $v\approx\mathbb{E}\big[z\vert s\big]$
\begin{equation}
(\mathbf{p},v)=f_\theta(s)
\end{equation}
The training of $f_\theta$ proceeds as
<ul id='number-list'>
	<li>
		The network is initialized to random weights $\theta_0$.
	</li>
	<li>
		At each subsequent iteration $i\geq 1$, games of self-play are generated.
	</li>
	<li>
		At each time-step $t$, an MCTS search $\boldsymbol{\pi}_t=\alpha_{\theta_{i-1}}(s_t)$ is executed using the previous iteration of neural network $f_{\theta_{i-1}}$ and a move is played by sampling from $\boldsymbol{\pi}_t$.
	</li>
	<li>
		A game terminates at step $T$ when the terminal condition is reached (both player pass, the search value drops below a resignation threshold or the game exceeds a maximum length), the game is then scored to give a final reward of $r_T\in\{-1,+1\}$.
	</li>
	<li>
		The data for each time-step $t$ is stored as $(s_t,\boldsymbol{\pi}_t,z_t)$ where $z_t=\pm r_T$ is the game winner from the perspective of the current player at step $t$.<br>
		In the mean time, new network parameters $\theta_i$ are trained from data $(s,\boldsymbol{\pi},z)$ sampled uniformly among all time-steps of the last iteration of self-play.
	</li>
	<li>
		The neural network $(\mathbf{p},v)$ is adjusted to minimize the error between the predicted value $v$ and the self-play winner $z$ while concurrently maximize the similarity of $\mathbf{p}$ to the search probability $\boldsymbol{\pi}$. Specifically, the parameters $\theta$ is updated by gradient descent on a loss function $l$ that sums over the MSE and cross-entropy losses
		\begin{equation}
		l=(z-v)^2-\boldsymbol{\pi}^\text{T}\log\mathbf{p}+c\Vert\theta\Vert^2,
		\end{equation}
		where $c$ is the L2 weight regularization parameter.
	</li>
</ul>
<figure>
	<img src="/images/alphazero/alphagozero-nn-training.png" alt="NN training in AlphaGo Zero" width="70%" height="70%"/>
	<figcaption style="text-align: center"><b>Figure 4</b>: (taken from <a href='#alphagozero-paper'>AlphaGo Zero paper</a>) <b>Neural network training in AlphaGo Zero</b></figcaption>
</figure>

#### Search algorithm
Each node in the search tree contains edges $(s,a)$ for all legal action $a\in\mathcal{A}(s)$. Each edge stores a set of statistics
\begin{equation}
\\{N(s,a),W(s,a),Q(s,a),P(s,a)\\}
\end{equation}
where $N(s,a)$ is the visit count, $W(s,a)$ is the total action value, $Q(s,a)$ is the mean action value and $P(s,a)$ is the prior probability of selecting the edge. The algorithm proceeds as
<ul id='number-list'>
	<li>
		<b>Select</b>. The selection phase begins at the root node $s_0$ and finishes when the simulation reaches a leaf node $s_L$ at time-step $L$. At each of these time-step, $t< L$, an action $a_t$ that maximizes the UCB is selected
		\begin{align}
		a_t&=\underset{a}{\text{argmax}}(Q(s_t,a)+U(s_t,a)) \\ U(s,a)&=c_\text{puct}P(s,a)\frac{\sqrt{\sum_b N(s,b)}}{1+N(s,a)},
		\end{align}
		where $c_\text{puct}$ is the UCT constant that controls the level exploration.
	</li>
	<li>
		<b>Expand and evaluate</b>. The leaf node $s_L$ is added to the queue for neural network evaluation. The leaf node is then expanded and each edge $(s_L,a)$ is initialized to
		\begin{equation}
		\{N(s_L,a)=0,W(s_L,a)=0,Q(s_L,a)=0,P(s_L,a)=p_a\},
		\end{equation}
		and the value $v$ is then backed up.
	</li>
	<li>
		<b>Backup</b>. The edge statistics are updated in a backward pass through each step $t\leq L$.
		\begin{align}
		N(s_t,a_t)&=N(s_t,a_t)+1 \\ W(s_t,a_t)&=W(s_t,a_t)+v \\ Q(s_t,a_t)&=\frac{W(s_t,a_t)}{N(s_t,a_t)}
		\end{align}
	</li>
	<li>
		<b>Play</b>. At the end of the search, the best child node corresponding to a move $a$ is selected from
		\begin{equation}
		\pi(a\vert s_0)=\frac{N(s_0,a)^{1/\tau}}{\sum_b N(s_0,b)^{1/\tau}},
		\end{equation}
		where $\tau$ is the temperature parameter that controls the level of exploration.
	</li>
</ul>

<figure>
	<img src="/images/alphazero/alphagozero-mcts.png" alt="MCTS in AlphaGo Zero" width="100%" height="100%"/>
	<figcaption style="text-align: center"><b>Figure 5</b>: (taken from <a href='#alphagozero-paper'>AlphaGo Zero paper</a>) <b>MCTS in AlphaGo Zero</b></figcaption>
</figure>

### AlphaZero


### References
[1] <span id='alphago-paper'>David Silver, Aja Huang, Chris J. Maddison et al. [Mastering the game of Go with deep neural networks and tree search](https://doi.org/10.1038/nature16961). Nature 529, 484–489, 2016.</span>

[2] <span id='alphagozero-paper'>David Silver, Julian Schrittwieser, Karen Simonyan et al. [Mastering the game of Go without human knowledge](https://doi.org/10.1038/nature24270). Nature 550, 354–359, 2017.</span>

[3] <span id='alphazero-paper'>David Silver, Thomas Hubert, Julian Schrittwieser et al. [A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play](DOI:10.1126/science.aar6404). Science 362, 1140-1144, 2018.</span>

[4] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.

## Footnotes
[^1]: This approach avoids overfitting, which occurs when training on outcomes from data consisting of complete games (since successive positions are strongly correlated). 