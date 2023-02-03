---
title: "Probabilistic Graphical Model - Representation"
date: 2022-12-10T17:55:57+07:00
tags: [machine-learning]
math: true
eqn-number: true
---
Notes on Bayesian Networks and Markov Random Fields.
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
		If $X_i-X_j\in\mathcal{E}$, we say that $X_i$ is a <b id='neighbor'>neighbor</b> of $X_j$, and vice versa.<br>
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
		A subgraph over $\mathbf{X}$ is <b>complete</b> if every two nodes in $\mathbf{X}$ are connected via some edges. The set $\mathbf{X}$ is known as a <b id='clique'>clique</b>; or even a <b>maximal clique</b> if for any set of nodes $\mathbf{Y}\supset\mathbf{X}$, $\mathbf{Y}$ is not a clique, i.e.
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
		$X_1,\ldots,X_k$ form a <b id='path'>path</b> in $\mathcal{K}$ if for every $i=1,\ldots,k-1$, we have that either $X_i\rightarrow X_{i+1}$ or $X_i-X_{i+1}$. A path is <b>directed</b> if there exists a directed edge $X_i\rightarrow X_{i+1}$.
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

## Directed Graphical Model{#dgm}
A **Directed Graphical Model** (or **Bayesian network**) is a tuple $\mathcal{B}=(\mathcal{G},P)$ where
- $\mathcal{G}$ is a **Bayesian network structure**,
- $P$ **factorizes** according to $\mathcal{G}$,
- $P$ is specified as a set of CPDs associated with $\mathcal{G}$'s nodes.

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
P\models(X\perp Y\vert Z),
\end{equation}
or
\begin{equation}
P(X,Y\vert Z)=P(X\vert Z)P(Y\vert Z)
\end{equation}
Let $\mathcal{K}$ be a graph associated with a set of independencies $\mathcal{I}(\mathcal{K})$, then $\mathcal{K}$ is an **I-map** (for independence map) for a set of independencies $\mathcal{I}$ if $\mathcal{I}(\mathcal{K})\subset\mathcal{I}$.

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
<ul id='number-list'>
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
		Moreover, the local independence for $X_i$ implies that
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
		P(\mathcal{X}\backslash X_i)&=\sum_{X_i}P(X_1,\ldots,X_n) \\ &=\sum_{X_i}\prod_{j=1}^{n}P(X_j\vert\text{Pa}_{X_j}) \\ &=\left(\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j})\right)\sum_{X_i}P(X_i\vert\text{Pa}_{X_i}) \\ &=\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j}),
		\end{align}
		where in the last step, we use the fact that the conditional probability function $P(X_i\vert\text{Pa}_{X_i})$ sum to $1$ over the sample space of $X_i$. This implies that by Bayes rules
		\begin{align}
		P(X_i\vert\mathcal{X}\backslash X_i)&=\frac{P(X_1,\ldots,X_n)}{P(\mathcal{X}\backslash X_i)} \\ &=\frac{\prod_{j=1}^{n}P(X_j\vert\text{Pa}_{X_j})}{\prod_{j=1,j\neq i}^{n}P(X_j\vert\text{Pa}_{X_j})} \\ &=P(X_i\vert\text{Pa}_{X_i})
		\end{align}
	</li>
</ul>

### Independencies in Bayesian Network

#### D-separation
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

Let $\mathbf{X},\mathbf{Y},\mathbf{Z}$ be sets of nodes in $\mathcal{G}$. Then $\mathbf{X}$ and $\mathbf{Y}$ are said to be **d-separated** given $\mathbf{Z}$, denoted $\text{d-sep}_\mathcal{G}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})$, if there is no active trail between any node $X\in\mathbf{X}$ and $Y\in\mathbf{Y}$ given $\mathbf{Z}$.

We define the **global Markov independencies** associated with $\mathcal{G}$, denoted $\mathcal{I}(\mathcal{G})$, to be the set of all independencies that correspond to d-separation in $\mathcal{G}$
\begin{equation}
\mathcal{I}(\mathcal{G})=\big\\{(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z}):\text{d-sep}\_\mathcal{G}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})\big\\}
\end{equation}

#### Soundness, Completeness{#soundness-completeness-bn}
**Theorem 2** (Soundness of d-separation) *If a distribution $P$ factorizes according to $\mathcal{G}$, then*
\begin{equation}
\mathcal{I}(\mathcal{G})\subset\mathcal{I}(P)
\end{equation}
The soundness property says that if $\text{d-sep}_\mathcal{G}(X;Y\vert\mathbf{Z})$ then they are conditional independent given $\mathbf{Z}$, or $(X\perp Y\vert\mathbf{Z})$.

**Theorem 3** (Completeness of d-separation) *If two variables $X$ and $Y$ are independent given $\mathbf{Z}$, then they are d-separated.*

The completeness property says that d-separation detects all possible independencies.

**Theorem 4**: *Let $\mathcal{G}$ be a BN graph. If $X$ and $Y$ are not d-separated given $\mathbf{Z}$, then $X$ and $Y$ are dependent given $\mathbf{Z}$ in some distribution $P$ that factorizes over $\mathcal{G}$*.

**Theorem 5**: *For almost all distributions $P$ that factorize over $\mathcal{G}$, i.e. for all distributions except for a set of measure zero in the space of CPD parameterizations, we have that*
\begin{equation}
\mathcal{I}(\mathcal{G})=\mathcal{I}(P)
\end{equation}
These results state that for almost all parameterizations $P$ of the graph $\mathcal{G}$, the d-separation test precisely characterizes the independencies that hold for $P$.

#### I-Equivalence
Two graph $\mathcal{K}_1$ and $\mathcal{K}_2$ over $\mathcal{X}$ are said to be **I-equivalent** if they encode the same set of conditional independencies assertions, i.e.
\begin{equation}
\mathcal{I}(\mathcal{K}\_1)=\mathcal{I}(\mathcal{K}\_2)
\end{equation}
This implies that any distribution $P$ that factorizes over $\mathcal{K}_1$ also factorizes according to $\mathcal{K}_2$ and vice versa.

The **skeleton** of a BN graph $\mathcal{G}$ over $\mathcal{X}$ is an undirected graph over $\mathcal{X}$ containing an edge $\\{X,Y\\}$ for every edge $(X,Y)$ in $\mathcal{G}$.

**Theorem 6** (skeleton + v-structures $\Rightarrow$ I-equivalence) *Let $\mathcal{G}_1$ and $\mathcal{G_2}$ be two graphs over $\mathcal{X}$. If $\mathcal{G}_1,\mathcal{G}_2$ both have the same skeleton and the same set of v-structures then they are I-equivalent*.[^2]
<figure>
	<img src="/images/pgm-representation/I-equivalence.png" alt="I-equivalent graphs" style="display: block; margin-left: auto; margin-right: auto; width: 80%; height: 80%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 3</b>: (taken from the <a href='#pgm-book'>PGM book</a>) Two graphs have the same skeleton and set of v-structures, i.e. $\{X\rightarrow Y\leftarrow Z\}$, and thus are I-equivalent</figcaption>
</figure>

##### Immorality
A v-structure $X\rightarrow Z\leftarrow Y$ is an **immorality** if there is no direct edge between. If there is such an edge, it is called a **covering edge** for the v-structure.

It is easily seen that not every v-structure is an immorality, which implies that two networks with the same set of immoralities do not necessarily have the same set of v-structures.

**Theorem 7** (skeleton + immoralities $\Leftrightarrow$ I-equivalence) *Let $\mathcal{G}_1$ and $\mathcal{G_2}$ be two graphs over $\mathcal{X}$. If $\mathcal{G}_1,\mathcal{G}_2$ both have the same skeleton and the same set of immoralities iff they are I-equivalent*.

**Proof**
<ul id='number-list'>
	<li>
		To prove the theorem, we first introduce the notion of <b>minimal active trail</b> and <b>triangle</b>.<br>
		<b>Definition</b> (Minimal active trail) An active trail $X_1,\ldots,X_m$ is <b>minimal</b> if there is no other active trail from $X_1$ to $X_m$ that shortcuts some of the nodes, i.e. there is no active trail
		\begin{equation}
		X_1\rightleftharpoons X_{i_1}\rightleftharpoons\ldots X_{i_k}\rightleftharpoons X_m\hspace{1cm}\text{for }1\lt i_1\lt\ldots\lt i_k\lt m
		\end{equation}
		<b>Definition</b> (Triangle) Any three consecutive nodes in a trail $X_1,\ldots,X_m$ are called a <b>triangle</b> if their skeleton is fully connected, i.e. forms a 3-clique.<br><br>
		Our attention now is to prove that the only possible triangle in minimal active trail is the one having form of $X_{i-1}\leftarrow X_i\rightarrow X_{i+1}$ and either $X_{i-1}\rightarrow X_{i+1}$ or $X_{i-1}\leftarrow X_{i+1}$.<br>
		Consider a two-edge trail from $X_{i-1}$ to $X_{i+1}$ via $X_i$, which as being <a href='#two-edge-trail'>mentioned</a> above, has four possible forms
		<ul id='alpha-list'>
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
		Given these results, we are now ready for the main part. Let us begin with the forward path.
	</li>
	<li>
		Skeleton + Immoralities $\Rightarrow$ I-equivalence<br>
		Assume that there exists node $X,Y,Z$ such that
		\begin{align}
		(X\perp Y\vert Z)&\in\mathcal{I}(\mathcal{G}_1), \\ (X\perp Y\vert Z)&\not\in\mathcal{I}(\mathcal{G}_2),
		\end{align}
		which implies that there is an active trail through $X,Y$ and $Z$ in the graph $\mathcal{G}_2$. Let us consider the minimal one and continue by examining two cases that whether $Z$ is observed.
		<ul id='alpha-list'>
			<li>
				If $Z$ is observed, in $\mathcal{G}_1$, we have $X\rightarrow Z\rightarrow Y$, or $X\leftarrow Z\leftarrow Y$, or $X\leftarrow Z\rightarrow Y$, while we have $X\rightarrow Z\leftarrow Y$ in $\mathcal{G}_2$, which is a v-structure. To assure that both graphs have the same set of moralities, there exist an edge that directly connects $X$ and $Y$, or in other words, $X,Y,Z$ form a triangle. This contradicts to the claim we have proved in the previous part.
			</li>
			<li>
				If $Z$ is not observed, thus in $\mathcal{G}_1$, $X,Y,Z$ now must form a v-structure $X\rightarrow Z\leftarrow Y$. And also, to guarantee that both graphs have the same moralities, there exists an edge, without loss of generality, we assume $X\rightarrow Y$. However, this edge will active the trail $X,Y,Z$, or in other words, in $\mathcal{G}_1$, we now have $(X\not\perp Y\vert Z)$, which is a contradiction of our assumption.
			</li>
		</ul>
	</li>
	<li>
		Skeleton + Immoralities $\Leftarrow$ I-equivalence<br>
		Consider two I-equivalent graphs $\mathcal{G}_1$ and $\mathcal{G}_2$.
		<ul id='alpha-list'>
			<li>
				First assuming that they do not have that same skeleton. This implies without loss of generality that there exists a trail in $\mathcal{G}_1$ that does not appear in $\mathcal{G}_2$, which induces a conditional independence in $\mathcal{G}_1$ but not in $\mathcal{G}_2$, contradicts to the fact that they two graphs are I-equivalent.
			</li>
			<li>
				Now assuming that two graphs do not have the same set of moralities.
			</li>
		</ul>
	</li>
</ul>

### From Distributions to Graphs{#dist2graph-bn}
Given a distribution $P$, how can we represent the independencies of $P$ with a graph $\mathcal{G}$?

#### Minimal I-Maps{#min-imap}
A graph $\mathcal{K}$ is a **minimal I-map** for a set of independencies $\mathcal{I}$ if it is an I-map for $\mathcal{I}$, and removing one edge from $\mathcal{K}$ makes it no longer be an I-map.

#### Perfect Maps
A graph $\mathcal{K}$ is a **perfect map** (or **P-map**) for a set of independencies $\mathcal{I}$ if we have that $\mathcal{I}(\mathcal{K})=\mathcal{I}$; and if $\mathcal{I}(\mathcal{K})=\mathcal{I}(P)$, $\mathcal{K}$ is said to be a **perfect map** for $P$.

## Undirected Graphical Model{#ugm}
Similar to the directed case, each node in an undirected graphical model represents a random variable. However, as indicated from the name, each edge that connects two nodes is now undirected, and thus can not describe causal relationship between those nodes as in Bayesian network.

### Markov Random Fields{#mrf}
An **Undirected Graphical Model** (or **Markov Random Field**, or **Markov network**) represents a probability distribution $P$ over variables $X_1,\ldots,X_n$ defined by an undirected graph $\mathcal{H}$ in which each node correspond to a variable $X_i$, and a set of positive **potential functions** $\psi_c$ associated with the [cliques](#clique) of $\mathcal{H}$ such that
\begin{equation}
P(X_1,\ldots,X_n)=\frac{1}{Z}\prod_{c\in C}\psi_c(\mathbf{X}\_c),\label{eq:mrf.1}
\end{equation}
where
- $P$ is also called a **Gibbs distribution**
- $C$ denotes the set of cliques of $\mathcal{H}$
- $\mathbf{X}_c$ represents the set of nodes within clique $c$
- $Z$ is known as the **partition function**, given by
\begin{equation}
Z=\sum_{X_1,\ldots,X_n}\prod_{c\in C}\psi_c(\mathbf{X}\_c)
\end{equation}

From the above definition, we say that a distribution $P$ is a Gibbs distribution that factorizes over $\mathcal{H}$ if it can be expressed as a normalized product over all potential functions defined on the cliques of $\mathcal{H}$, as in \eqref{eq:mrf.1}.

### Independencies in Markov Network
Analogy to Bayesian network, the graph structure in a Markov Random Field can be viewed as encoding a set of independence assumptions, which can be specified by considering the undirected paths through nodes.

#### Separation
Let $\mathcal{H}$ be a Markov network structure and let $X_1-\ldots-X_k$ be a [path](#path) in $\mathcal{H}=(\mathcal{X},\mathcal{E})$. Let $\mathbf{Z}\subset\mathcal{X}$ be a set of observed variables. Then $X_1-\ldots-X_k$ is **active** given $\mathbf{Z}$ if none of $X_1,\ldots,X_k$ is in $\mathbf{Z}$.

Let $\mathbf{X},\mathbf{Y}$ be set of nodes in $\mathcal{H}$. We say that $\mathbf{Z}$ **separates** $\mathbf{X}$ and $\mathbf{Y}$, denoted $\text{sep}_\mathcal{H}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})$ if there is no active path between any node $X\in\mathbf{X}$ and $Y\in\mathbf{Y}$ given $\mathbf{Z}$.

#### Global Markov Independencies
As in the case of Bayesian network, we define the **global Markov independencies** associated with $\mathcal{H}$, denoted $\mathcal{I}(\mathcal{H})$, to be the set of all independencies correspond to separation in $\mathcal{H}$
\begin{equation}
\mathcal{I}(\mathcal{H})=\big\\{(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z}):\text{sep}\_\mathcal{H}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})\big\\}
\end{equation}

#### Local Markov Independencies
Let $\mathcal{H}$ be a Markov network. We define the **pairwise independencies** associated with $\mathcal{H}$ to be
\begin{equation}
\mathcal{I}\_p(\mathcal{H})=\big\\{(X\perp Y\vert\mathcal{X}\backslash\\{X,Y\\}):X-Y\notin\mathcal{H}\big\\}
\end{equation}
Or in other words, the pairwise independencies states that $X$ and $Y$ are independent given all other nodes in $\mathcal{H}$.

For a given graph $\mathcal{H}$ and for an arbitrary node $X$ of $\mathcal{H}$, the set of [neighbors](#neighbor) of $X$ is also called the **Markov blanket** of $X$, denoted $\text{MB}_\mathcal{H}(X)$. With this notion, we define the **local independencies**, or **Markov local independencies**, associated with $\mathcal{H}$ to be
\begin{equation}
\mathcal{I}\_\ell(\mathcal{H})=\big\\{\big(X\perp\mathcal{X}\backslash(\\{X\\}\cup\text{MB}\_\mathcal{H}(X))\vert\text{MB}\_\mathcal{H}(X)\big):X\in\mathcal{X}\big\\}
\end{equation}
Or in other words, the Markov local independencies says that $X$ is independent of the rest of the nodes in $\mathcal{H}$ given its neighbors.

The definition of **Markov blanket** can also be rewritten using independencies assertions:  
A set $\mathbf{U}$ is a **Markov blanket** of $X$ in a distribution $P$ if $X\notin\mathbf{U}$ and if $\mathbf{U}$ is a minimal set of nodes such that
\begin{equation}
\big(X\perp\mathcal{X}\backslash(\\{X\\}\cup\mathbf{U})\vert\mathbf{U}\big)\in\mathcal{I}(P)
\end{equation}

#### Markov Independencies Relationships
**Theorem 8**: *Let $\mathcal{H}$ be a Markov network and $P$ be a positive distribution. The following three statement are then equivalent:*
<ul id='roman-list' style='font-style: italic;'>
	<li>
		$P\models\mathcal{I}_\ell(\mathcal{H})$.
	</li>
	<li>
		$P\models\mathcal{I}_p(\mathcal{H})$.
	</li>
	<li>
		$P\models\mathcal{I}(\mathcal{H})$.
	</li>
</ul>

**Proof**
<ul id='number-list'>
	<li>
		(i) $\Rightarrow$ (ii)<br>
		Consider an arbitrary node $X$ in $\mathcal{H}$. Let $Y\in\mathcal{X}$ such that $X-Y\notin\mathcal{H}$, then $Y\notin\text{MB}_\mathcal{H}(X)$, or in other words
		\begin{equation}
		Y\in\mathcal{X}\backslash(\{X\}\cup\text{MB}_\mathcal{H}(X))
		\end{equation}
		Moreover, since $P\models\mathcal{I}_\ell(\mathcal{H})$, we have that $P$ satisfies
		\begin{equation}
		\big(X\perp\mathcal{X}\backslash(\{X\}\cup\text{MB}_\mathcal{H}(X))\vert\text{MB}_\mathcal{H}(X)\big),
		\end{equation}
		which implies that
		\begin{equation}
		\big(X\perp Y\vert\text{MB}_\mathcal{X}\cup\mathcal{X}\backslash(\{X,Y\}\cup\text{MB}_\mathcal{H}(X))\big)
		\end{equation}
		holds for $P$. Or in other words, for any $X,Y$ such that $X-Y\notin\mathcal{H}$, we have
		\begin{equation}
		P\models(X\perp Y\vert\mathcal{X}\backslash\{X,Y\}),
		\end{equation}
		which proves our claim.
	</li>
	<li>
		(ii) $\Rightarrow$ (iii)
	</li>
	<li>
		(iii) $\Rightarrow$ (i)<br>
		This follows directly from the fact that if two nodes $X$ and $Y$ are not connected, then they are necessarily separated by all remaining nodes of the graph.
	</li>
</ul>

#### Soundness, Completeness{#soundness-completeness-mn}
**Theorem 9** (Soundness of separation) *Let $\mathcal{H}=(\mathcal{X},\mathcal{E})$ be a Markov network structure and let $P$ be a distribution over $\mathcal{X}$. If $P$ is a Gibbs distribution that factorizes over $\mathcal{H}$, then $\mathcal{H}$ is an I-map for $P$.*

**Proof**  
Let $\mathbf{X},\mathbf{Y}$ and $\mathbf{Z}$ be any disjoint subsets of $\mathcal{X}$ such that $\text{sep}\_\mathcal{H}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})$. We need to show that
\begin{equation}
P\models(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z})
\end{equation}
We begin by considering the case that $\mathbf{X}\cup\mathbf{Y}\cup\mathbf{Z}=\mathcal{X}$. Since $\mathbf{Z}$ separates $\mathbf{X}$ and $\mathbf{Y}$, then for any $X\in\mathbf{X}$ and for any $Y\in\mathbf{Y}$, there is no direct edge between $X,Y$. This implies that any clique in $\mathcal{H}$ is either in $\mathbf{X}\cup\mathbf{Z}$ or in $\mathbf{Y}\cup\mathbf{Z}$.

Let $C_\mathbf{X}$ denote the set of cliques within $\mathbf{X}\cup\mathbf{Z}$ and let $C_\mathbf{Y}$ represent the set of cliques in $\mathbf{Y}\cup\mathbf{Z}$. Thus, as $P$ factorizes over $\mathcal{H}$, we have that
\begin{align}
P(X_1,\ldots,X_n)&=\frac{1}{Z}\left(\prod_{c\in C_\mathbf{X}}\psi(X_c)\right)\left(\prod_{c\in C_\mathbf{Y}}\psi(X_c)\right) \\\\ &=\frac{1}{Z}\psi_\mathbf{X}(\mathbf{X},\mathbf{Z})\psi_\mathbf{Y}(\mathbf{Y},\mathbf{Z}),
\end{align}
where $\psi_\mathbf{X},\psi_\mathbf{Y}$ are some factor with scopes $\mathbf{X}\cup\mathbf{Z}$ and $\mathbf{Y}\cup\mathbf{Z}$ respectively. Hence, it follows that
\begin{equation}
P\models(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z})
\end{equation}
Now consider the case that $\mathbf{X}\cup\mathbf{Y}\cup\mathbf{Z}\subset\mathcal{X}$. Let $\mathbf{U}=\mathcal{X}\backslash(\mathbf{X}\cup\mathbf{Y}\cup\mathbf{Z})$. We can divide $\mathbf{U}$ into two disjoint sets $\mathbf{U}_1$ and $\mathbf{U}_2$ such that
\begin{equation}
\text{sep}\_\mathcal{H}(\mathbf{X}\cup\mathbf{U}\_1;\mathbf{Y}\cup\mathbf{U}\_2\vert\mathbf{Z})
\end{equation}
And since $(\mathbf{X}\cup\mathbf{U}_1)\cup(\mathbf{Y}\cup\mathbf{U}_2)\cup\mathbf{Z}=\mathcal{X}$, we can follow the previous procedure to conclude that
\begin{equation}
P\models(\mathbf{X}\cup\mathbf{U}\_1\perp\mathbf{Y}\cup\mathbf{U}\_2\vert\mathbf{Z}),
\end{equation}
which implies that
\begin{equation}
P\models(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z})
\end{equation}

**Theorem 10** (Hammersley-Clifford) *Let $\mathcal{H}=(\mathcal{X},\mathcal{E})$ be a Markov network structure and let $P$ be a positive distribution over $\mathcal{X}$. If $\mathcal{H}$ is an I-map for $P$, then $P$ is a Gibbs distribution that factorizes over $\mathcal{H}$.*

**Corollary 11**: *The positive distribution $P$ factorizes a Markov network $\mathcal{H}$ iff $\mathcal{H}$ is an I-map for $P$*.

**Theorem 12** (Completeness of separation) *Let $\mathcal{H}$ be a Markov network structure. If $X$ and $Y$ are not separated given $\mathbf{Z}$ in $\mathcal{H}$, then $X$ and $Y$ are dependent given $\mathbf{Z}$ in some distribution $P$ that factorizes over $\mathcal{H}$.*

### From Distributions to Graphs{#dist2graph-mn}
As [mentioned](#min-imap) above, the notion of minimal I-map lets us encode the independencies of a given distribution $P$ using a graph structure.

In particular, for a distribution $P$, we can construct the minimal I-map based on either the pairwise independencies or the local independencies.

**Theorem 13**: *Let $P$ be a positive distribution and $\mathcal{H}$ be a Markov network defined by including an edge $X-Y$ for all $X,Y$ such that $P\not\models(X\perp Y\vert\mathcal{X}\backslash\\{X,Y\\})
$. Then $\mathcal{H}$ is  the unique minimal I-map for $P$.*

**Theorem 14**: *Let P be a positive distribution and let $\mathcal{H}$ be a Markov network defined by including an edge $X-Y$ for all $X$ and all $Y\in\text{MB}_\mathcal{H}(X)$. Then $\mathcal{H}$ is the unique minimal I-map for $P$.*

**Remark**: Not every distribution has a perfect map as UGM (proof by contradiction).

### Factor Graphs
A **factor graph** $\mathcal{F}$ is an undirected graph whose nodes are divided into two groups: variable nodes (denoted as ovals) and factor nodes (denoted as squares) and whose edges only connect each factor (potential function) $\psi_c$ to its dependent nodes $X\in X_c$.
<figure>
	<img src="/images/pgm-representation/factor-graphs.png" alt="Same Markov network different factor graphs" style="display: block; margin-left: auto; margin-right: auto"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 4</b>: (based on figure from the <a href='#pgm-book'>PGM book</a>) Different factor graphs for the same Markov network (a) A Markov network consists of nodes $X_1,X_2,X_3$; (b) A factor graph with a factor $\psi_{1,2,3}$ connected to each $X_1,X_2,X_3$; (c) A factor graph with three pairwise factors $\psi_{1,2}$ (connected to $X_1,X_2$), $\psi_{1,3}$ (connected to $X_1,X_3$) and $\psi_{2,3}$ (connected to $X_2,X_3$)</figcaption>
</figure>

### Log-Linear Models
A distribution $P$ is a **log-linear model** over a Markov network $\mathcal{H}$ if it is associated with
<ul id='number-list'>
	<li>
		a set of features $\mathcal{F}=\{\phi_1(\mathbf{X}_1),\ldots,\phi_k(\mathbf{X}_k)\}$ where each $\mathbf{X}_i$ is a complete subgraph in $\mathcal{H}$,
	</li>
	<li>
		a set of weight $w_1,\ldots,w_k$, such that
		\begin{equation}
		P(X_1,\ldots,X_n)=\frac{1}{Z}\exp\left[-\sum_{i=1}^{k}w_i\phi_i(\mathbf{X}_i)\right]
		\end{equation}
	</li>
</ul>

The function $\phi_i$ are called **energy functions**.

### Conditional Random Fields{#crf}
A **conditional random field**, or **CRF**, is an undirected graph $\mathcal{H}$ whose nodes correspond to $\mathbf{X}\cup\mathbf{Y}$ where $\mathbf{X}$ is a set of observed variables and $\mathbf{Y}$ is a (disjoint) set of target variables which specifies a conditional distribution (instead of a joint distribution)
\begin{equation}
P(\mathbf{Y}\vert\mathbf{X})=\frac{1}{Z(\mathbf{X})}\prod_{c\in C}\psi_c(\mathbf{X}\_c,\mathbf{Y}\_c),
\end{equation}
where the partition function $Z(\mathbf{X})$ now depends on $\mathbf{X}$ (rather than being a constant)
\begin{equation}
Z(\mathbf{X})=\sum_\mathbf{Y}\prod_{c\in C}\psi_c(\mathbf{X}\_c,\mathbf{Y}\_c)
\end{equation}

#### Example - Naive Markov
Consider a CRF over the binary-value variables $\mathbf{X}=\\{X_1,\ldots,X_k\\}$ and $\mathbf{Y}=\\{Y\\}$, and a pairwise potential between $Y$ and each $X_i$. The model is referred as a **naive Markov model**. Assume that the pairwise potentials defined via the log-linear model
\begin{equation}
\psi_i(X_i,Y)=\exp\big[w_i\mathbf{1}\\{X_i=1,Y=1\\}\big]
\end{equation}
Additionally, we also have a single-node potential
\begin{equation}
\psi_0(Y)=\exp\big[w_0\mathbf{1}\\{Y=1\\}\big]
\end{equation}
We then have that
\begin{equation}
P(Y=1\vert x_1,\ldots,x_k)=\frac{1}{1+\exp\left[-\left(w_0+\sum_{i=1}^{k}w_i x_i\right)\right]}
\end{equation}

## Local Probabilistic Models

### Tabular CPDs
For a space of discrete-valued random variables only, we can encode the CPDs $P(X\vert\text{Pa}_X)$ as a table where each entry corresponds to a pair of $X,\text{Pa}_X$.

It is easily seen that this representation raises a disadvantage that the number of parameters required grows exponentially in the number of parents. Also, it is impossible to store the conditional probability corresponding to a continuous-valued random variables.

To avoid these problems, instead of viewing CPDs as tables listing all of the conditional probabilities $P(x\vert\text{pa}_X)$, we should consider them as functions that given $\text{pa}_X$ and $x$, return the conditional probability $P(x\vert\text{pa}_X)$.

### Deterministic CPDs
The simplest type of non-tabular CPD corresponds to a variable $X$ being a deterministic function of its parents $\text{Pa}_X$. It means, there exists $f:\text{Val}(Pa_X)\mapsto\text{Val}(X)$ such that
\begin{equation}
P(x\vert\text{pa}\_X)=\begin{cases}1&\hspace{1cm}x=f(\text{pa}\_X) \\\\ 0&\hspace{1cm}\text{otherwise}\end{cases}
\end{equation}
Deterministic variables are denoted as double-line ovals, as illustrated in the following example
<figure>
	<img src="/images/pgm-representation/det-cpd.png" alt="Network with a deterministic CPD" style="display: block; margin-left: auto; margin-right: auto; width: 30%; height: 30%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 5</b>: (taken from the <a href='#pgm-book'>PGM book</a>) A network with $C$ being a deterministic function of $A$ and $B$</figcaption>
</figure>

Consider the above figure, as $C$ being a deterministic function of $A$ and $B$, we can deduce that $C$ is fully observed if $A$ and $B$ are both observed. In other words, we have that
\begin{equation}
(D\perp E\vert A,B)
\end{equation}

**Theorem 15**: *Let $\mathcal{G}$ be a network structure, and let $\mathbf{X},\mathbf{Y},\mathbf{Z}$ be sets of variables, $\mathbf{D}$ be set of deterministic variables. If $\mathbf{X}$ is **deterministically separated** from $\mathbf{Y}$ given $\mathbf{Z}$[^3], then for all distributions $P$ such that $P\models\mathcal{I}_\ell(\mathcal{G})$ and where, for each $X\in\mathbf{D}$, $P(X\vert\text{Pa}_X)$ is a deterministic CPD, we have that $P\models(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z})$*.

**Theorem 16**: *Let $\mathcal{G}$ be a network structure, and let $\mathbf{X},\mathbf{Y},\mathbf{Z}$ be sets of variables, $\mathbf{D}$ be set of deterministic variables. If $\mathbf{X}$ is not deterministically separated from $\mathbf{Y}$ given $\mathbf{Z}$, then there exists a distribution $P$ such that $P\models\mathcal{I}_\ell(\mathcal{G})$ and where, for each $X\in\mathbf{D}$, $P(X\vert\text{Pa}_X)$ is a deterministic CPD, but we instead have $P\not\models(\mathbf{X}\perp\mathbf{Y}\vert\mathbf{Z})$*.

It is worth remarking that particular deterministic CPD might imply additional independencies. For instance, let us consider the following examples

**Example 1**: Consider the following Bayesian network
<figure>
	<img src="/images/pgm-representation/complex-det-cpd.png" alt="Network with a deterministic CPD" style="display: block; margin-left: auto; margin-right: auto; width: 40%; height: 40%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 6</b>: (taken from the <a href='#pgm-book'>PGM book</a>) Another Bayesian network with $C$ being a deterministic function of $A$ and $B$</figcaption>
</figure>

In the above figure, if $C=A\text{ XOR }B$, we have that $A$ is fully determined given $C$ and $B$. In other words, we have that
\begin{equation}
(D\perp E\vert B,C)
\end{equation}
**Example 2**: Consider **Figure 5**, with $C=A\text{ OR }B$. Assume that we are given $A=a^1$, it is then immediately that $C=c^1$ without taking into account the value of $B$. Or in other words, we have that
\begin{equation}
P(D\vert B,a^1)=P(D\vert a^1)
\end{equation}
On the other hand, given $A=a^0$, the value of $C$ is not fully determined, and still depend the value of $B$.

#### Context-Specific Independence
Let $\mathbf{X},\mathbf{Y},\mathbf{Z}$ be pairwise disjoint sets of variables, let $\mathbf{C}$ be a set of variable (which might overlap with $\mathbf{X}\cup\mathbf{Y}\cup\mathbf{Z}$), and let $\mathbf{c}\in\text{Val}(\mathbf{C})$. We say that $X$ and $\mathbf{Y}$ are **contextually independent** given $\mathbf{Z}$ and the context $\mathbf{C}$ denoted $(\mathbf{X}\perp_c\mathbf{Y}\vert\mathbf{Z},\mathbf{c})$ if
\begin{equation}
P(\mathbf{Y},\mathbf{Z},\mathbf{c})>0\Rightarrow P(\mathbf{X}\vert\mathbf{Y},\mathbf{Z},\mathbf{c})=P(\mathbf{X}\vert\mathbf{Z},\mathbf{c})
\end{equation}
Given this definition, let us examine some examples.

**Example 3**: Given the Bayesian network in **Figure 5** with $C$ being a deterministic function $\text{OR}$ of $A$ and $B$. By properties of $OR$ function, we can conclude some independence assertions
\begin{align}
&(C\perp_c B\vert\hspace{0.1cm}a^1), \\\\ &(D\perp_c B\vert\hspace{0.1cm}a^1), \\\\ &(A\perp_c B\vert\hspace{0.1cm}c^0), \\\\ &(D\perp_c E\vert\hspace{0.1cm}c^0), \\\\ &(D\perp_c E\vert\hspace{0.1cm}b^0,c^1)
\end{align}
**Example 4**: Given the Bayesian network in **Figure 6** with $C$ being the exclusive or of $A$ and $B$. We can also conclude some independence assertions using properties of $\text{XOR}$ function
\begin{align}
&(D\perp_c E\vert\hspace{0.1cm}b^1,c^0), \\\\ &(D\perp_c E\vert\hspace{0.1cm}b^0,c^1)
\end{align}

### Context-specific CPDs

#### Tree-CPDs
A **tree-CPD** representing a CPD for variable $X$ is a rooted tree, where:
<ul id='number-list'>
	<li>
		each leaf node is labeled with a distribution $P(X)$;
	</li>
	<li>
		each internal node is labeled with some variable $Z\in\text{Pa}_X$;
	</li>
	<li>
		each edge from an internal node, which is labeled as some $Z$, to its child nodes corresponds to a $z_i\in\text{Val}(Z)$.
	</li>
</ul>
<figure>
	<img src="/images/pgm-representation/tree-cpd.png" alt="Tree-CPD" style="display: block; margin-left: auto; margin-right: auto; width: 30%; height: 30%"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 7</b>: (taken from the <a href='#pgm-book'>PGM book</a>) A tree CPD for $P(J\vert A,S,L)$</figcaption>
</figure>

The structure is common in cases where a variable can depend on a set of r.v.s but we have uncertainty about which r.v.s it depends on. For example, in the above tree-CDP representing $P(J\vert A,S,L)$, we have that
\begin{align}
&(J\perp_c L,S\vert\hspace{0.1cm}a^0), \\\\ &(J\perp_c L\vert\hspace{0.1cm}a^1,s^1), \\\\ &(J\perp_c L\vert\hspace{0.1cm}s^1)
\end{align}

##### Multiplexer CPD
A CPD $P(Y\vert A,Z_1,\ldots,Z_k)$ is said to be a **multiplexer CPD** if $\text{Val}(A)=\\{1,\ldots,k\\}$ and
\begin{equation}
P(Y\vert a,Z_1,\ldots,Z_k)=\mathbf{1}\\{Y=Z_a\\},
\end{equation}
where $a$ is the value of $A$. The variable $A$ is referred as the **selector variable** for the CPD.
<figure>
	<img src="/images/pgm-representation/multiplexer-cpd.png" alt="Multiplexer-CPD" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 8</b>: (based on figure from the <a href='#pgm-book'>PGM book</a>) (a) A Bayesian network for $P(J,C,L_1,L_2)$; (b) Tree-CPD for $P(J\vert C,L_1,L_2)$; (c) Modified network with additional variable $L$ acting as a multiplexer CPD</figcaption>
</figure>

#### Rule CPDs
A **rule** $\rho$ is a pair $(\mathbf{c},p)$ where $\mathbf{c}$ is an assignment to some subset of variables $\mathbf{C}$ and $p\in[0,1]$. $\mathbf{C}$ is then referred as the **scope** of $\rho$, denoted $\mathbf{C}=\text{Scope}(\rho)$.

This representation decomposes a tree-CPD into its most basic elements.

**Example 5**: Consider the tree-CPD given in **Figure 7**. The tree defines eight rules
\begin{Bmatrix}(a^0,j^0;0.8), \\\\ (a^0,j^1;0.2), \\\\ (a^1,s^0,l^0,j^0;0.9), \\\\ (a^1,s^0,l^0,j^1;0.1), \\\\ (a^1,s^0,l^1,j^0;0.4), \\\\ (a^1,s^0,l^1,j^1;0.6), \\\\ (a^1,s^1,j^0;0.1), \\\\ (a^1,s^1,j^1;0.9)\end{Bmatrix}
It is necessary that each conditional distribution $P(X\vert\text{Pa}_X)$ is specified by exactly one rule. Or in other words, the rules in a tree-CPD must be mutually exclusive and exhaustive.

##### Rule-based CPD
A **rule-based CPD** $P(X\vert\text{Pa}_X)$ is a set of rules $\mathcal{R}$ such that
<ul id='roman-list'>
	<li>
		For each $\rho\in\mathcal{R}$, we have that
		\begin{equation}
		\text{Scope}(\rho)\subset\{X\}\cup\text{Pa}_X
		\end{equation}
	</li>
	<li>
		For each assignment $(x,\mathbf{u})$ to $\{X\}\cup\text{Pa}_X$, we have exactly one rule $(\mathbf{c};p)\in\mathcal{R}$ such that $\mathbf{c}$ is compatible with $(x,\mathbf{u})$. And we say that
		\begin{equation}
		P(X=x\vert\text{Pa}_X=\mathbf{u})=p
		\end{equation}
	</li>
	<li>
		$\sum_x P(x\vert\mathbf{u})=1$.
	</li>
</ul>

**Example 6**: Let $X$ be a variable with $\text{Pa}_X=\\{A,B,C\\}$ with $X$'s CPD is defined by sets of rules
\begin{Bmatrix}\rho_1:(a^1,b^1,x^0;0.1), \\\\ \rho_2:(a^1,b^1,x^1;0.9), \\\\ \rho_3:(a^0,c^1,x^0;0.2), \\\\ \rho_4:(a^0,c^1,x^1;0.8), \\\\ \rho_5:(b^0,c^0,x^0;0.3), \\\\ \rho_6:(b^0,c^0,x^1;0.7), \\\\ \rho_7:(a^1,b^0,c^1,x^0;0.4), \\\\ \rho_8:(a^1,b^0,c^1,x^1;0.6), \\\\ \rho_9:(a^0,b^1,c^0;0.5)\end{Bmatrix}
The tree-CPD corresponds to the above rule-based CPD $P(X\vert A,B,C)$ is given as:
<figure>
	<img src="/images/pgm-representation/rule-based-cpd.png" alt="Rule-based-CPD" style="display: block; margin-left: auto; margin-right: auto; width: 50%; height: 50%"/>
	<figcaption></figcaption>
</figure>
It is worth noticing that both CPD entries $P(x^0\vert a^0,b^1,c^0)$ and $P(x^1\vert a^0,b^1,c^0)$ are determined by rule $\rho_9$ only. This kind of rule only works for uniform distribution.

#### Independencies in Context-specific CPDs

### Independence of Causal Influence

#### Noisy-Or Model

#### Generalized Linear Models


## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

[2] Michael I. Jordan. [An Introduction to Probabilistic Graphical Models](http://people.eecs.berkeley.edu/~jordan/prelims/). In preparation.

[3] Eric P.Xing. [10-708: Probabilistic Graphical Model](https://www.cs.cmu.edu/~epxing/Class/10708-20/). CMU Spring 2020.

[4] Stefano Ermon. [CS228: Probabilistic Graphical Model](https://cs.stanford.edu/~ermon/cs228/index.html). Stanford Winter 2017-2018.

## Footnotes
[^1]: Note that $X_i\rightarrow X_j\equiv X_j\leftarrow X_i$ but $X_i\rightarrow X_j\not\equiv X_i\leftarrow X_j$, while $X_i-X_j\equiv X_j-X_i$.
[^2]: Note that the inverse is not true.
[^3]: This can be specified by doing the procedure
	> Let $\mathbf{Z}^+\leftarrow\mathbf{Z}$  
	> While $\exists X_i$ such that $X_i\in\mathbf{D}$  and $\text{Pa}_{X_i}\subset\mathbf{Z}^+$  
	> $\hspace{1cm}\mathbf{Z}^+\leftarrow\mathbf{Z}\cup\\{X_i\\}$  
	> return $\text{d-sep}(\mathbf{X};\mathbf{Y}\vert\mathbf{Z})$