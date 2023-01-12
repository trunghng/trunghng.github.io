---
title: "Probabilistic Graphical Model - Representation"
date: 2023-01-11T17:55:57+07:00
draft: true
tags: [machine-learning]
math: true
eqn-number: true
---
Notes on Representation in Probabilistic Graphical Models.
<!--more-->

## Graphs
A **graph**, denoted $\mathcal{K}$ is a tuple  of $\mathcal{X}$ and $\mathcal{E})$ where $\mathcal{X}=\\{X_1,\ldots,X_n\\}$ is the sets of **nodes** (or **vertices**) and $\mathcal{E}$ is the set of **edges**.
\begin{equation}
\mathcal{K}=(\mathcal{X},\mathcal{E})
\end{equation}

### Nodes, Edges
Any pair of nodes $X_i,X_j$, for $i\neq j$ is connected by either a **directed edge** $X_i\rightarrow X_j$ or an **undirected edge** $X_i-X_j$[^1]. We use the notation $X_i\rightleftharpoons X_j$ to denote that $X_i$ is connected to $X_j$ via some edge, whether directed (in any direction) or undirected.

If the graph contains directed edges only, we call it a **directed graph**, denoted $\mathcal{G}$, else if the graph established by undirected edge only, it is referred as **undirected graph**, denoted $\mathcal{H}$.
<figure>
	<img src="/images/pgm-representation/graph-eg.png" alt="Graph example" style="display: block; margin-left: auto; margin-right: auto; width: 70%; height: 70%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b>: (taken from the <a href='#pgm-book'>PGM book</a>) Example of a partially directed graph $\mathcal{K}$</figcaption>
</figure>

Following are some necessary notations:
<ul id='number-list'>
	<li>
		If $X_i\rightarrow X_j\in\mathcal{E}$, we say that $X_i$ is the <b>parent</b> of $X_j$ while $X_j$ is the <b>child</b> of $X_i$.<br>
		E.g. node $I$ is a child of nodes $C,E$ and $H$ while $D$ is a parent of $G$.
	</li>
	<li>
		If $X_i-X_j\in\mathcal{E}$, we say that $X_i$ is a <b>neighbor</b> of $X_j$, and vice versa.<br>
		E.g. node $F$ is a neighbor of $G$.
	</li>
	<li>
		If $X\rightleftharpoons Y\in\mathcal{E}$, we say that $X$ and $Y$ are adjacent.<br>
		E.g. nodes $A$ and $C$ are adjacent, while $D$ is adjacent to $E$.
	</li>
	<li>
		We use $\text{Pa}_X$ to denote the set of parents of $X$, $\text{Ch}_X$ to denote the set of its children and $\text{Nb}_X$ to denote its neighbors. The set $\text{Boundary}_X\doteq\text{Pa}_X\cup\text{Nb}_X$ is known as the <b>boundary</b> of $X$.<br>
		E.g. $\text{Pa}_I=\{C,E,H\}$ and $\text{Boundary}_F=\text{Pa}_F\cup\text{Nb}_F=\{C\}\cup\{G\}=\{C,G\}$.
	</li>
	<li>
		The <b>degree of a node</b> $X$ is the number of edges in which it participates; its <b>indegree</b> is the number of directed edges $Y\rightarrow X$. The <b>degree of a graph</b> is the maximal degree of a node in the graph.<br>
		E.g. node $D$ has degree of $3$, indegree of $0$; the graph $\mathcal{K}$ has degree of $3$.
	</li>
</ul>

### Subgraphs
Consider the graph $\mathcal{K}=(\mathcal{X},\mathcal{E})$ and let $\mathbf{X}\subset\mathcal{X}$ be a subset of nodes in $\mathcal{K}$. Then:
<ul id='number-list'>
	<li>
		The <b>induced subgraph</b> of $\mathcal{K}$, denoted $\mathcal{K}[\mathbf{X}]$ is defined as the graph $(\mathbf{X},\mathcal{E}')$ where
		\begin{equation}
		\mathcal{E}'=\{X\rightleftharpoons Y:X,Y\in\mathbf{X}\}
		\end{equation}
	</li>
	<li>
		A subgraph over $\mathbf{X}$ is <b>complete</b> if every two nodes in $\mathbf{X}$ are connected via some edges. The set $\mathbf{X}$ is known as a <b>clique</b>; or even a <b>maximal clique</b> if for any set of nodes $\mathbf{Y}\supset\mathbf{X}$, $\mathbf{Y}$ is not a clique, i.e.
		\begin{equation}
		\{\mathbf{Y}\text{ clique}:\mathbf{Y}\supset\mathbf{X}\}=\emptyset
		\end{equation}
	</li>
	<li>
		The set $\mathbf{X}$ is called <b>upward closed</b> in $\mathbf{K}$ if for any $X\in\mathbf{X}$, we have that
		\begin{equation}
		\text{Boundary}_X\subset\mathbf{X}
		\end{equation}
		The <b>upward closure</b> of $\mathbf{X}$ is the minimal closed subset $\mathbf{Y}$ covering $\mathbf{X}$, i.e.
		\begin{equation}
		\mathbf{Y}=\sup\{\bar{\mathbf{Y}}\text{ upward closed in }\mathcal{K}:\bar{\mathbf{Y}}\supset\mathbf{X}\}
		\end{equation}
		The <b>upwardly closed subgraph</b> of $\mathbf{X}$, denoted $\mathcal{K}^+[\mathbf{X}]$, is the induced subgraph over $\mathbf{Y}$, $\mathcal{K}[\mathbf{Y}]$.
	</li>
</ul>

### Paths, Trails
Consider the graph $\mathcal{K}=(\mathcal{X},\mathcal{E})$, the basic notion of edges gives rise to following definitions:
<ul id='number-list'>
	<li>
		$X_1,\ldots,X_k$ form a <b>path</b> in $\mathcal{K}$ if for every $i=1,\ldots,k-1$, we have that either $X_i\rightarrow X_{i+1}$ or $X_i-X_{i+1}$. A path is <b>directed</b> if there exists a directed edge $X_i\rightarrow X_{i+1}$.
	</li>
	<li>
		$X_1,\ldots,X_k$ form a <b>trail</b> in $\mathcal{K}$ if for every $i=1,\ldots,k-1$, we have that $X_i\rightleftharpoons X_{i+1}$.
	</li>
	<li>
		$\mathcal{K}$ is <b>connected</b> if for every pair $X_i,X_j$ there is a trail between $X_i$ and $X_j$.
	</li>
	<li>
		$X$ is an <b>ancestor</b> of $Y$ and correspondingly $Y$ is a <b>descendant</b> of $X$ in $\mathcal{K}$ if there exists a directed path $X_1,\ldots,X_k$ with $X_1=X$ and $X_k=Y$.
	</li>
	<li>
		An ordering of nodes $X_1,\ldots,X_n$ is a <b id='topo-order'>topological ordering</b> relative to $\mathbf{K}$ if whenever we have $X_i\rightarrow X_j\in\mathcal{E}$, then $i\lt j$. This gives rise to critical results that for each node $X_i$, we have that
		\begin{equation}
		\text{Pa}_{X_i}\subset\{X_1,\ldots,X_{i-1}\},
		\end{equation}
		and
		\begin{equation}
		\text{Ch}_{X_i}\subset\{X_{i+1},\ldots,X_n\}
		\end{equation}
	</li>
</ul>

### Cycles, Loops
<ul id='number-list'>
	<li>
		A <b>cycle</b> in $\mathcal{K}$ is a directed path $X_1,\ldots,X_k$ where $X_1=X_k$. $\mathcal{K}$ is <b>acyclic</b> if it contains no cycles.
	</li>
	<li>
		$\mathcal{K}$ is a <b>directed acyclic graph</b> (or <b>DAG</b>) if it is both directed and acyclic.
	</li>
	<li>
		An acyclic graph containing both directed and undirected edges is known as a <b>partially directed acyclic graph</b> (or <b>PDAG</b>).<br>
		Let $\mathcal{K}$ be a PDAG over $\mathcal{X}$ and let $\mathbf{K}_1,\ldots,\mathbf{K}_\ell$ be a disjoint partition of $\mathcal{X}$ such that:
		<ul id='roman-list'>
			<li>
				the induced subgraph over $\mathbf{K}_i$ contains no directed edges;
			</li>
			<li>
				for any pair of nodes $X\in\mathbf{K}_i$ and $Y\in\mathbf{K}_j$ for $i\lt j$, an edge between $X$ and $Y$ can only be a directed edge $X\rightarrow Y$.
			</li>
		</ul>
		Each component $\mathbf{K}_i$ is called a <b>chain component</b>, while $\mathcal{K}$ is referred as a <b>chain graph</b>.
	</li>
	<li>
		A <b>loop</b> in $\mathcal{K}$ is a trail $X_1,\ldots,X_k$ where $X_1=X_k$. A graph is <b>singly connected</b> if it contains no loops. A node in a singly connected graph is called a <b>leaf</b> if it has exactly one adjacent node.
	</li>
	<li>
		A singly connected directed graph is called a <b>polytree</b>, while a singly connected undirected graph is known as a <b>forest</b>; if it is also connected, it is called a <b>tree</b>.
	</li>
	<li>
		A directed graph is a <b>forest</b> if each node has at most one parent. A directed forest is a <b>tree</b> if it is also connected.
	</li>
</ul>

## Bayesian Networks

### Bayesian Network Structure
A **Bayesian network structure** (or **Bayesian network graph**, **BN graph**) is a DAG, denoted $\mathcal{G}=(\mathcal{X},\mathcal{E})$ with $\mathcal{X}=\\{X_1,\ldots,X_n\\}$ where
<ul id='number-list'>
	<li>
		Each node $X_i\in\mathcal{X}$ represents a random variable.
	</li>
	<li>
		Each node $X_i\in\mathcal{X}$ is associated with a conditional independencies assumption, called <b>local independencies</b>, denoted $\mathcal{I}_\ell(\mathcal{G})$, which says that $X_i$ is conditionally independent of its non-descendants given its parent, i.e.
		\begin{equation}
		(X_i\perp\text{NonDescendants}_{X_i}\vert\hspace{0.1cm}\text{Pa}_{X_i}),
		\end{equation}
		where $\text{NonDescendants}_{X_i}$ denotes the set of non-descendant nodes of $X_i$.
	</li>
</ul>

### I-Maps
Let $P$ be a distribution over $\mathcal{X}$, we define $\mathcal{I}(P)$ to be the set of independence assertions of the form $(X\perp Y\hspace{0.1cm}\vert Z)$ that hold in $P$.

Let $\mathcal{K}$ be a graph associated with aa set of independencies $\mathcal{I}(\mathcal{K})$, then $\mathcal{K}$ is an **I-map** for a set of independencies $\mathcal{I}$ if $\mathcal{I}(\mathcal{K})\subset\mathcal{I}$.

Hence, if $P$ satisfies the local dependencies associated with $\mathcal{G}$, we have
\begin{equation}
\mathcal{I}\_\ell(\mathcal{G})\subset\mathcal{I}(P),
\end{equation}
which implies that $\mathcal{G}$ is an I-map for $\mathcal{I}(P)$, or simply an I-map for $P$.

### Factorization
Let $\mathcal{G}$ is a BN graph over $X_1,\ldots,X_n\in\mathcal{X}$. A distribution $P$ over $\mathcal{X}$ is said to **factorize** according to $\mathcal{G}$ if $P$ can be expressed as a product
\begin{equation}
P(X_1,\ldots,X_n)=\prod_{i=1}^{n}P(X_i\vert\text{Pa}\_{X_i})
\end{equation}
This equation is known as the **chain rule for Bayesian networks**. Each individual factor $P(X_i\vert\text{Pa}\_{X_i})$, which is a conditional probability distribution (CPD), is called the **local probabilistic model**.

### I-Map - Factorization Connection
**Theorem 1**: Let $\mathcal{G}$ be a BN graph over a set of random variables $\mathcal{X}$ and let $P$ be a joint distribution over $\mathcal{X}$. Then $\mathcal{G}$ is an I-map for $P$ if and only if $P$ factorizes over $\mathcal{G}$.

**Proof**
<ul id='roman-list'>
	<li>
		($\Rightarrow$)<br>
		Without loss of generality, let $X_1,\ldots,X_n$ be a <a href='#topo-order'>topological ordering</a> of the variables in $\mathcal{X}$.<br>
		Let us consider an arbitrary $X_i$ for $i\in\{1,\ldots,n\}$. As mentioning above, the topological ordering implies that
		\begin{align}
		\text{Pa}_{X_i}&\subset\{X_1,\ldots,X_{i-1}\}, \\ \text{Ch}_{X_i}&\subset\{X_{i+1},\ldots,X_n\}
		\end{align}
		Consequently, none of descendants of $X_i$ is in $\{X_1,\ldots,X_{n-1}\}$. Thus, if we denote the set of all non-descendant nodes of $X_i$ as $\text{NonDescentdants}_{X_i}$ and let $\mathbf{Z}\subset\text{NonDescentdants}_{X_i}$, then
		\begin{equation}
		\mathbf{Z}\cup\text{Pa}_{X_i}=\{X_1,\ldots,X_{i-1}\}
		\end{equation}
		Moreover, the local independencies for $X_i$ implies that
		\begin{equation}
		(X_i\perp\mathbf{Z}\vert\text{ Pa}_{X_i})
		\end{equation}
		Therefore, since $\mathcal{G}$ is an I-map for $P$ we obtain
		\begin{equation}
		P(X_i\vert X_1,\ldots,X_{i-1})=P(X_i\vert\text{Pa}_{X_i})
		\end{equation}
		Thus, by Bayes theorem, we have
		\begin{equation}
		P(X_1,\ldots,X_n)=\prod_{i=1}^{n}P(X_i\vert X_1,\ldots,X_{i-1})=\prod_{i=1}^{n}P(X_i\vert\text{Pa}_{X_i})
		\end{equation}
	</li>
	<li>
		($\Leftarrow$)<br>
		To prove that $\mathcal{G}$ is an I-map according to $P$, we need to show that $\mathcal{I}_\ell(\mathcal{G})$ holds in $P$.<br>
		Consider an arbitrary node $X_i$ for $i\in\{1,\ldots,n\}$, and the local independencies $(X_i\perp\text{NonDescendants}_{X_i}\vert\text{Pa}_{X_i})$, our problem remains to prove that
		\begin{equation}
		P(X_i\vert\mathcal{X}\backslash X_i)=P(X_i\vert\text{Pa}_{X_i})
		\end{equation}
		since
		\begin{equation}
		P(X_i\vert\mathcal{X}\backslash X_i)=P(X_i\vert\text{NonDescendants}_{X_i}\cup\text{Pa}_{X_i})
		\end{equation}
		By factorization, we have that
		\begin{align}
		P(\mathcal{X}\backslash X_i)&=\sum_{X_i}P(X_1,\ldots,X_n) \\ &=\sum_{X_i}\prod_{j=1}^{n}P(X_j\vert\text{Pa}_{X_j}) \\ &=\left(\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j})\right)\underbrace{\sum_{X_i}P(X_i\vert\text{Pa}_{X_i})}_{=1} \\ &=\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j}),
		\end{align}
		which implies that by Bayes rules
		\begin{align}
		P(X_i\vert\mathcal{X}\backslash X_i)&=\frac{P(X_1,\ldots,X_n)}{P(\mathcal{X}\backslash X_i)} \\ &=\frac{\prod_{j=1}^{n}P(X_j\vert\text{Pa}_{X_j})}{\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j})} \\ &=P(X_i\vert\text{Pa}_{X_i})
		\end{align}
	</li>
</ul>

### Bayesian Network Definition
A **Bayesian network** is a tuple $\mathcal{B}=(\mathcal{G},P)$ where $P$ factorizes according to $\mathcal{G}$, and where $P$ is specified as a set of CPDs associated with $\mathcal{G}$'s nodes.

## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes
[^1]: Note that $X_i\rightarrow X_j\equiv X_j\leftarrow X_i$ but $X_i\rightarrow X_j\not\equiv X_i\leftarrow X_j$, while $X_i-X_j\equiv X_j-X_i$. 
