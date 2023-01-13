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
	<img src="/images/pgm-representation/graph-eg.png" alt="Graph example" style="display: block; margin-left: auto; margin-right: auto; width: 50%; height: 50%"/>
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
Let $P$ be a distribution over $\mathcal{X}$, we define $\mathcal{I}(P)$ to be the set of independence assertions of the form $(X\perp Y\hspace{0.1cm}\vert Z)$ that hold in $P$, i.e.
\begin{equation}
P(X,Y\vert Z)=P(X\vert Z)P(Y\vert Z)
\end{equation}
Let $\mathcal{K}$ be a graph associated with aa set of independencies $\mathcal{I}(\mathcal{K})$, then $\mathcal{K}$ is an **I-map** (for independence map) for a set of independencies $\mathcal{I}$ if $\mathcal{I}(\mathcal{K})\subset\mathcal{I}$.

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
		I-map $\Rightarrow$ Factorization<br>
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
		Thus, by the chain rule for probabilities, we have
		\begin{equation}
		P(X_1,\ldots,X_n)=\prod_{i=1}^{n}P(X_i\vert X_1,\ldots,X_{i-1})=\prod_{i=1}^{n}P(X_i\vert\text{Pa}_{X_i})
		\end{equation}
	</li>
	<li>
		I-map $\Leftarrow$ Factorization<br>
		To prove that $\mathcal{G}$ is an I-map according to $P$, we need to show that $\mathcal{I}_\ell(\mathcal{G})$ holds in $P$. Consider an arbitrary node $X_i$ and the local independencies $(X_i\perp\text{NonDescendants}_{X_i}\vert\text{Pa}_{X_i})$, our problem remains to prove that
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

### D-separation
Let $\mathcal{G}$ be a BN structure, $X_1\rightleftharpoons\ldots\rightleftharpoons X_n$ be a trail in $\mathcal{G}$ and let $\mathbf{Z}$ be a subset of observed variables. The trail $X_1\rightleftharpoons\ldots\rightleftharpoons X_n$ is **active** if
<ul id='roman-list'>
	<li>
		Whenever we have a <b>v-structure</b> $X_{i-1}\rightarrow X_i\leftarrow X_{i+1}$, $X_i$ or one of its descendants are in $\mathbf{Z}$;
	</li>
	<li>
		No other node along the trail are in $\mathbf{Z}$.
	</li>
</ul>

<figure>
	<img src="/images/pgm-representation/two-edge-trails.png" alt="Two-edge trails" style="display: block; margin-left: auto; margin-right: auto; width: 70%; height: 70%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 2</b>: (taken from the <a href='#pgm-book'>PGM book</a>) The four possible two-edge trails from $X$ to $Y$ via $Z$: (a) Causal trail; (b) Evidential trail; (c) Common cause trail; (d) Common effect trail</figcaption>
</figure>

<span id='two-edge-trail'>Consider the trails forming from two edges as illustrated above</span>:
<ul id='alpha-list'>
	<li>
		The trail $X\rightarrow Z\rightarrow Y$ is active $\Leftrightarrow$ $Z$ is not observed.
	</li>
	<li>
		The trail $X\leftarrow Z\leftarrow Y$ is active $\Leftrightarrow$ $Z$ is not observed.
	</li>
	<li>
		The trail $X\leftarrow Z\leftarrow Y$ is active $\Leftrightarrow$ $Z$ is not observed.
	</li>
	<li>
		The trail $X\rightarrow Z\leftarrow Y$ is active $\Leftrightarrow$ either $Z$ or one of its descendants is observed.
	</li>
</ul>

#### D-separated
Let $\mathbf{X},\mathbf{Y},\mathbf{Z}$ be sets of nodes in $\mathbf{G}$. Then $\mathbf{X}$ and $\mathbf{Y}$ are said to be **d-separated** given $\mathbf{Z}$, denoted $\text{d-sep}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})$, if there is no active trail between any node $X\in\mathbf{X}$ and $Y\in\mathbf{Y}$ given $\mathbf{Z}$.

We use $\mathcal{I}(\mathcal{G})$ to denote the dependencies correspond to d-separation:
\begin{equation}
\mathcal{I}(\mathcal{G})=\\{(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z}):\text{d-sep}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})\\}
\end{equation}
The set is also known as **global Markov independencies**.

#### Soundness, Completeness
A distribution $P$ is **faithful** to $\mathcal{G}$ if whenever $(X\perp Y\vert\mathbf{Z})\in\mathcal{I}(P)$, then $\text{d-sep}(X;Y\vert\mathbf{Z})$.

**Theorem 2** (Soundness of d-separation) *If a distribution $P$ factorizes according to $\mathcal{G}$, then*
\begin{equation}
\mathcal{I}(\mathcal{G})\subset\mathcal{I}(P)
\end{equation}

**Theorem 3** (Completeness of d-separation) *Let $\mathcal{G}$ be a BN graph. If $X$ and $Y$ are not d-separated given $\mathbf{Z}$, then $X$ and $Y$ are dependent given $\mathbf{Z}$ in some distribution $P$ that factorizes over $\mathcal{G}$*.

**Theorem 4**: *For almost all distributions $P$ that factorize over $\mathcal{G}$, i.e. for all distributions except for a set of measure zero in the space of CPD parameterizations, we have that*
\begin{equation}
\mathcal{I}(\mathcal{G})=\mathcal{I}(P)
\end{equation}

These results state that for almost all parameterizations $P$ of the graph $\mathcal{G}$, the d-separation test precisely characterizes the independencies that hold for $P$.

### I-Equivalence
The **skeleton** of a BN graph $\mathcal{G}$ over $\mathcal{X}$ is an undirected graph over $\mathcal{X}$ containing an edge $\\{X,Y\\}$ for every edge $(X,Y)$ in $\mathcal{G}$.

Two graph $\mathcal{K}_1$ and $\mathcal{K}_2$ over $\mathcal{X}$ are said to be **I-equivalent** if they encode the same set of conditional independencies assertions, i.e.
\begin{equation}
\mathcal{I}(\mathcal{K}\_1)=\mathcal{I}(\mathcal{K}\_2)
\end{equation}
This implies that any distribution $P$ that factorizes over $\mathcal{K}_1$ also factorizes according to $\mathcal{K}_2$ and vice versa.

**Theorem 5** (skeleton + v-structures $\Rightarrow$ I-equivalence) *Let $\mathcal{G}_1$ and $\mathcal{G_2}$ be two graphs over $\mathcal{X}$. If $\mathcal{G}_1,\mathcal{G}_2$ both have the same skeleton and the same set of v-structures then they are I-equivalent*.[^2]
<figure>
	<img src="/images/pgm-representation/I-equivalence.png" alt="I-equivalent graphs" style="display: block; margin-left: auto; margin-right: auto; width: 80%; height: 80%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: (taken from the <a href='#pgm-book'>PGM book</a>) Two graphs have the same skeleton and set of v-structures, i.e. $\{X\rightarrow Y\leftarrow Z\}$, and thus are I-equivalent</figcaption>
</figure>

#### Immorality
A v-structure $X\rightarrow Z\leftarrow Y$ is an **immorality** if there is no direct edge between. If there is such an edge, it is called a **covering edge** for the v-structure.

It is easily seen that not every v-structure is an immorality, which implies that two networks with the same set of immoralities do not necessarily have the same set of v-structures.

**Theorem 6** (skeleton + immoralities $\Leftrightarrow$ I-equivalence) *Let $\mathcal{G}_1$ and $\mathcal{G_2}$ be two graphs over $\mathcal{X}$. If $\mathcal{G}_1,\mathcal{G}_2$ both have the same skeleton and the same set of immoralities iff they are I-equivalent*.

**Proof**  
To prove the theorem, we first introduce the notion of **minimal active trail** and **triangle**.

**Definition** (Minimal active trail) An active trail $X_1,\ldots,X_m$ is **minimal** if there is no other active trail from $X_1$ to $X_m$ that shortcuts some of the nodes, i.e. there is no active trail
\begin{equation}
X_1\rightleftharpoons X_{i_1}\rightleftharpoons\ldots X_{i_k}\rightleftharpoons X_m\hspace{1cm}\text{for }1\lt i_1\lt\ldots\lt i_k\lt m
\end{equation}
**Definition** (Triangle in trail) Any three consecutive nodes in a trail $X_1,\ldots,X_m$ are called a **triangle** if their skeleton is fully connected, i.e. forms a 3-clique.

Our first step is to prove that the only possible triangle in minimal active trail is the one having form of
<ul id='roman-list'>
	<li>
		$X_{i-1}\leftarrow X_i\rightarrow X_{i+1}$
	</li>
	<li>
		Either $X_{i-1}\rightarrow X_{i+1}$ or $X_{i-1}\leftarrow X_{i+1}$.
	</li>
</ul>

Consider a two-edge trail from $X_{i-1}$ to $X_{i+1}$ via $X_i$, which as being [mentioned](#two-edge-trail) above, has four possible forms
<ul id='number-list'>
	<li>
		$X_{i-1}\rightarrow X_i\rightarrow X_{i+1}$<br>
		It is easily seen that $X_i$ has to be not observed to make the trail active. If $X_{i-1}$ is connected to $X_{i+1}$ via $X_{i-1}\rightarrow X_{i+1}$, this gives rise to a shortcut. On the other hand, if they are connected by $X_{i-1}\leftarrow X_{i+1}$, the triangle now induces a cycle.
	</li>
	<li>
		$X_{i-1}\leftarrow X_i\leftarrow X_{i+1}$<br>
		This case is symmetrically identical to the previous one, and thus is not viable.
	</li>
	<li>
		$X_{i-1}\leftarrow X_i\rightarrow X_{i+1}$<br>
		The first observation is that $X_i$ has to be not given. The second observation is $X_{i-1}$ and $X_{i+1}$ are symmetric through $X_i$, so we only need to consider some specific cases of $X_{i-1}$ and the same logic is applied to $X_{i+1}$ analogously.<br>
		Let us examine the two-edge trail $X_{i-2},X_{i-1},X_i$. On the one hand, if we have $X_{i-2}\rightarrow X_{i-1}$, $X_{i-1}$ then has to be given, which implies that
		<ul>
			<li>
				If $X_{i-1}\leftarrow X_{i+1}$ exists, it will create a shortcut, which is not allowed.
			</li>
			<li>
				If $X_{i-1}\rightarrow X_{i+1}$ exists, no shortcut appears, $X_{i-1},X_i,X_{i+1}$ satisfies the condition of a triangle in the minimal active trail $X_1,\ldots,X_m$.
			</li>
		</ul>
		On the other hand, if we have $X_{i-2}\leftarrow X_{i-1}$, then $X_{i-1}$ is not observed, analogously, we instead have
		<ul>
			<li>
				If $X_{i-1}\leftarrow X_{i+1}$ exists, no shortcut is formed, $X_{i-1},X_i,X_{i+1}$ create a triangle.
			</li>
			<li>
				If $X_{i-1}\rightarrow X_{i+1}$ exists, $X_{i-1},X_i,X_{i+1}$ is do not form a triangle due to the appearance of a shortcut through $X_{i-1}$ to $X_{i+1}$.
			</li>
		</ul>
	</li>
	<li>
		$X_{i-1}\rightarrow X_i\leftarrow X_{i+1}$<br>
		In this case, $X_i$ or one of its descendant is observed. Using the similar procedure to previous case gives us no viable triangle formed by $X_{i-1},X_i,X_{i+1}$.
	</li>
</ul>



#### Covered edge
An edge $X\rightarrow Y$ in a graph $\mathcal{G}$ is said to be **covered** if
\begin{equation}
\text{Pa}\_Y=\text{Pa}\_X\cup\\{X\\}
\end{equation}

**Theorem 7**: *Two graphs $\mathcal{G}$ and $\mathcal{G}'$ are I-equivalent iff there exists a sequence of networks $\mathcal{G}=\mathcal{G}_1,\ldots,\mathcal{G}_k=\mathcal{G}'$ that are all I-equivalent to $\mathcal{G}$ such that the only difference between $\mathcal{G}\_i$ and $\mathcal{G}\_{i+1}$ is a single reversal of a covered edge*.

### Minimal I-Maps

### Perfect Maps



## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes
[^1]: Note that $X_i\rightarrow X_j\equiv X_j\leftarrow X_i$ but $X_i\rightarrow X_j\not\equiv X_i\leftarrow X_j$, while $X_i-X_j\equiv X_j-X_i$.
[^2]: Note that the backward path is not true.
