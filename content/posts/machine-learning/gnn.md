---
title: "Graph Neural Network"
date: 2024-01-18T11:32:05+07:00
tags: [machine-learning, deep-learning, neural-network]
draft: true
math: true
eqn-number: true
hideSummary: true
---

### The Graph Neural Network model
A graph $G=(N,E)$ is a tuple of the set of nodes $N$ and the set of edges $E$, which give rise to following notations
<ul id='number-list'>
	<li>
		$\text{ne}[n]$ is the neighbors of node $n$.
	</li>
	<li>
		$\text{co}[n]$ is the set of arcs linked to node $n$
	</li>
	<li>
		The labels attached to node $n$ and edge $(n_1,n_2)$ are denoted by $l_n\in\mathbb{R}^{\mathbf{l}_N}$ and $l_{(n_1,n_2)}\in\mathbb{R}^{\mathbf{l}_E}$; The vector consisting of all labels of the graph is represented as $\mathbf{l}$.
	</li>
	<li>
		A positional graph is defined as, for each node $n$ in the graph, there exists an injective function $\nu_n:\text{ne}[n]\mapsto\{1,\ldots\vert N\vert\}$, which assigns to each other $u$ of $n$ a position $\nu_n(u)$, indicated by a unique integer.
	</li>
</ul>

Let $\mathcal{D}=\mathcal{G}\times\mathcal{N}$ denote the pairs of a graph and a node, where $\mathcal{G}$ is a set of graphs and $\mathcal{N}$ is a subset of their nodes. Consider a learning set
\begin{equation}
\mathcal{L}=\Big\\{(G_i,n_{i,j},t_{i,j})\big\vert G_i=(N_i,E_i)\in\mathcal{G};n_{i,j}\in N_i;t_{i,j}\in\mathbb{R}^m,1\leq i\leq p,1\leq j\leq q_i\Big\\},
\end{equation}
where $n_{i,j}\in N_i$ denotes the $j$th node of $N_i\in\mathcal{N}$; $t_{i,j}$ is the desired target corresponded to $n_{i,j}$; and where $p\leq\vert\mathcal{G}\vert,q_i\leq\vert N_i\vert$.

It is worth noting that all of the graphs of the learning set can be combined into a unique disconnected graph. And thus, the learning set can be viewed as the pair $\mathcal{L}=(G,\mathcal{T})$ where $G=(N,E)$ is a graph and where
\begin{equation}
\mathcal{T}=\big\\{(n_i,t_i)\big\vert n_i\in N,t_i\in\mathbb{R}^m,1\leq i\leq q\big\\}
\end{equation}

#### Model
Let $f_w$ be the **local transition function** that expresses the dependence of a node $n$ on its neighborhood and let $g_w$ be the **logical output function** that describes how the output is produced. We have
\begin{align}
x_n&=f_w(l_n,\mathbf{l}\_{\text{co}[n]},\mathbf{x}\_{\text{ne}[n]},\mathbf{l}\_{\text{ne}[n]})\nonumber \\\\ o_n&=g_w(x_n,l_n),
\end{align}
where $x_n$ is known as the state and $o_n$ is the output produced by that output. The above definitions can be written in compact forms
\begin{align}
\mathbf{x}&=F_w(\mathbf{x},\mathbf{l})\nonumber \\\\ \mathbf{o}&=G_w(\mathbf{x},\mathbf{l}\_N)\label{eq:m.1}
\end{align}
where $\mathbf{x},\mathbf{o}$ denote the vectors containing all the states and the outputs, and where $F_w,G_w$ are also the vectorized version of $f_w$ and $g_w$.

Recall that the **Banach fixed-point theorem** states that, let $(X,d)$ be a non-empty complete metric space, and let $T:X\mapsto X$ be a contraction mapping on $X$[^1], then $T$ admits a fixed-point $x^*$ in $X$[^2]. Applying the theorem to \eqref{eq:m.1}, we have that \eqref{eq:m.1} has a unique solution if $F_w$ is a contraction map w.r.t the state, i.e., there exists $\mu\in[0,1)$ such that the following holds for any $\mathbf{x},\mathbf{y}$
\begin{equation}
\Vert F_w(\mathbf{x},\mathbf{l})-F_w(\mathbf{y},\mathbf{l})\Vert\leq\mu\Vert\mathbf{x}-\mathbf{y}\Vert,
\end{equation}
where $\Vert\cdot\Vert$ denote a vector norm.

#### Computation of the State



### References
[1] F. Scarselli, M. Gori, A. C. Tsoi, M. Hagenbuchner and G. Monfardini. [The Graph Neural Network Model](https://ieeexplore.ieee.org/document/4700287). IEEE Transactions on Neural Networks, vol. 20, no. 1, pp. 61-80, Jan. 2009.

[2] [Banach fixed-point theorem]().

### Footnotes
[^1]: This happens if there exists $\mu\in[0,1)$ such that for all $x,y\in X$, we always have that
\begin{equation}
d(T(x),T(y))\leq\mu d(x,y)
\end{equation}
[^2]: i.e., $T(x^\*)=x^*$