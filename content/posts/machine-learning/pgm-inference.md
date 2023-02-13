---
title: "Probabilistic Graphical Models - Inference"
date: 2023-02-02T15:51:13+07:00
tags: [machine-learning, probabilistic-graphical-model]
math: true
eqn-number: true
---
Notes on Inference in PGMs.
<!--more-->

## Exact Inference
**Problem setting**: We need to effectively compute the **conditional probability query**, which is the conditional probability of variables $\mathbf{Y}$ given evidence $\mathbf{E}=\mathbf{e}$
\begin{equation}
P(\mathbf{Y}\vert\mathbf{E}=\mathbf{e})
\end{equation}

### Variable Elimination

#### Intuition
Consider a simple chain $A\rightarrow B\rightarrow C\rightarrow D$ with our interest of computing $P(D)$, which can be computed by
\begin{equation}
P(D)=\sum_C\sum_B\sum_A P(A,B,C,D)\label{eq:int.1}
\end{equation}
This computation is ineffective, since it grows exponentially with the number of variables $n$. Specifically, if each of variables $A,B,C,D$ ($n=4$) takes $k$ possible values, the time complexity required is $\mathcal{O}(k^{n-1})$.

Using the chain rule of probability, we can instead calculate $P(D)$ as
\begin{align}
P(D)&=\sum_C\sum_B\sum_A P(A,B,C,D) \\\\ &=\sum_C P(D\vert C)\sum_B P(C\vert B)\color{red}{\sum_A P(A)P(B\vert A)} \\\\ &=\sum_C P(D\vert C)\color{red}{\sum_B P(C\vert B)P(B)} \\\\ &=\color{red}{\sum_C P(D\vert C)P(C)},
\end{align}
which has computation cost of $\mathcal{O}((n-1)k(k-1))=\mathcal{O}(nk^2)$ only due to each $\color{red}{summation}$ takes $k(k-1)$ computations.

**Remark**: The later computation is more effective than the former one due to
<ul id='roman-list'>
	<li>
		The local independencies in Bayesian networks allow us to compactly rewrite $P(X\vert\mathcal{X}\backslash X)=P(X\vert\text{Pa}_X)$.
	</li>
	<li>
		At each step, by computing the $\color{red}{summations}$ first, we cached their values to use in the outer summations.
	</li>
</ul>

#### Basic Elimination
As in **Representation** task, using the notion of **factors**, we then can formally define an algorithm that applies to both Bayesian networks and Markov networks.

##### Factor Marginalization
Let $\mathbf{X}$ be a set of variables, and $Y\notin\mathbf{X}$. Let $\phi(\mathbf{X},Y)$ be a factor. By marginalizing out $Y$, we obtain the **factor marginalization** of $Y$ in $\phi$, which is a factor $\psi$ over $\mathbf{X}$ that
\begin{equation}
\psi(\mathbf{X})=\sum_Y\phi(\mathbf{X},Y)
\end{equation}
In addition, let us recall some useful identities of factors:
<ul id='number-list'>
	<li>
		Commutative of products: $\phi_1\cdot\phi_2=\phi_2\cdot\phi_1$.
	</li>
	<li>
		Commutative of summations: $\sum_X\sum_Y\phi=\sum_Y\sum_X\phi$.
	</li>
	<li>
		Associative of products: $(\phi_1\cdot\phi_2)\cdot\phi_3=\phi_1\cdot(\phi_2\cdot\phi_3)$.
	</li>
	<li>
		If $X\notin\text{Scope}(\phi_1)$, then
		\begin{equation}
		\sum_X(\phi_1\cdot\phi_2)=\phi_1\cdot\sum_X\phi_2\label{eq:fm.1}
		\end{equation}
	</li>
</ul>

##### The Variable Elimination Algorithm
Using identity \eqref{eq:fm.1}, we can write
\begin{equation}
P(A,B,C,D)=\phi_A\cdot\phi_B\cdot\phi_C\cdot\phi_D,
\end{equation}
which allows us to continue deriving \eqref{eq:int.1} as
\begin{align}
P(D)&=\sum_C\sum_B\sum_A P(A,B,C,D) \\\\ &=\sum_C\sum_B\sum_A\phi_A\cdot\phi_B\cdot\phi_C\cdot\phi_D \\\\ &=\sum_C\sum_B\phi_C\cdot\phi_D\cdot\left(\sum_A\phi_A\cdot\phi_B\right) \\\\ &=\sum_C\phi_D\cdot\left(\sum_B\phi_C\cdot\left(\sum_A\phi_A\cdot\phi_B\right)\right),
\end{align}
which is justified by the limited scope of the CPD factors, e.g. the second step is due to the fact that $A\notin\text{Scope}(\phi_C)\cup\text{Scope}(\phi_D)$. This computation suggests that in general, our problem is to computing the value of an expression of the form
\begin{equation}
\sum_\mathbf{Z}\prod_{\phi\in\Phi}\phi,
\end{equation}
which is referred as the **sum-product inference** task. This gives rise to the algorithm called **Sum-Product Variable Elimination**, as given in the following pseudocode.
<figure>
	<img src="/images/pgm-inference/sum-product-ve.png" alt="Sum-Product VE" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption></figcaption>
</figure>

**Theorem 1**: Let $\mathbf{X}$ be some set of variables, and let $\Phi$ be a set of factors such that for each $\phi\in\Phi$, $\text{Scope}(\phi)\subset\mathbf{X}$. Let $\mathbf{Y}\subset\mathbf{X}$ be a set of query variables, and let $\mathbf{Z}=\mathbf{X}\backslash\mathbf{Y}$. Then for any ordering $\prec$ over $\mathbf{Z}$, $\text{Sum-Product-VE}(\Phi,\mathbf{Z},\prec)$ returns a factor $\phi^\*(\mathbf{Y})$ such
\begin{equation}
\phi^\*(\mathbf{Y})=\sum_\mathbf{Z}\prod_{\phi\in\Phi}\phi
\end{equation}
**Remark**:
<ul id='number-list'>
	<li>
		To apply the algorithm to a Bayesian network $\mathcal{B}$ for computing $P_\mathcal{B}(\mathbf{Y})$, we begin by instantiating $\Phi$ to comprise all of the CPDs
		\begin{equation}
		\Phi=\{\phi_{X_i}\}_{i=1,\ldots,n},
		\end{equation}
		where $\phi_{X_i}=P(X_i\vert\text{Pa}_{X_i})$. Then, we apply the algorithm to the set $\mathbf{Z}=\mathcal{X}\backslash\mathbf{Y}$.
	</li>
	<li>
		Similarly, with a Markov network $\mathcal{H}$, we begin by instantiating $\Phi$ as the set of potential functions
		\begin{equation}
		\Phi=\{\phi_c\}_{c\in C},
		\end{equation}
		where $C$ be a set of cliques of $\mathcal{H}$. Analogously, we then apply the algorithm to the set $\mathbf{Z}=\mathcal{X}\backslash\mathbf{Y}$, which now returns an unnormalized factor over $\mathbf{Y}$. By dividing the partition function, we obtain $P_\mathcal{H}(\mathbf{Y})$.
	</li>
</ul>

**Example 2**: Consider the following Bayesian network with the goal is to computing the probability that the student got the job.
<figure>
	<img src="/images/pgm-inference/student-bn.png" alt="Student BN" width="40%" height="40%"/>
	<figcaption><b>Figure 1</b> (taken from the <a href='#pgm-book'>PGM book</a>) <b>A Bayesian network for Student scenario</b></figcaption>
</figure>

By chain rule of probability, we have that
\begin{align}
P(C,D,I,G,S,L,J,H)&=P(C)P(D\vert C)P(I)P(G\vert D,I)P(S\vert I)P(L\vert G)\nonumber \\\\ &\hspace{2cm}P(J\vert L,S)P(H\vert G,J) \\\\ &=\phi\_C(C)\phi\_D(D,C)\phi\_I(I)\phi\_G(G,D,I)\phi\_S(S,I)\phi\_L(L,G)\nonumber \\\\ &\hspace{2cm}\phi\_J(J,L,S)\phi\_H(H,G,J)
\end{align}
Consider the ordering: $C,D,I,H,G,S,L$. We step by step do the elimination procedure to each variable
<ul id='number-list'>
	<li>
		Eliminating $C$. We have
		\begin{align}
		\psi_1(D,C)&=\phi_C(C)\cdot\phi_D(D,C) \\ \tau_1(D)&=\sum_C\psi_1(D,C)
		\end{align}
	</li>
	<li>
		Eliminating $D$. We have
		\begin{align}
		\psi_2(G,D,I)&=\phi_G(G,D,I)\cdot\tau_1(D) \\ \tau_2(G,I)&=\sum_{D}\psi_2(G,D,I)
		\end{align}
	</li>
	<li>
		Eliminating $I$. We have
		\begin{align}
		\psi_3(G,I,S)&=\phi_I(I)\cdot\phi_S(S,I)\cdot\tau_2(G,I) \\ \tau_3(G,S)&=\sum_I\psi_3(G,I,S)
		\end{align}
	</li>
	<li>
		Eliminating $H$. We have
		\begin{align}
		\psi_4(H,G,J)&=\phi_H(H,G,J) \\ \tau_4(G,J)&=\sum_H\psi_4(H,G,J),
		\end{align}
		which is equal to $1$, since $\sum_H P(H\vert G,J)=1$
	</li>
	<li>
		Eliminating $G$. We have
		\begin{align}
		\psi_5(G,J,L,S)&=\phi_L(L,G)\cdot\tau_3(G,S)\cdot\tau_4(G,J) \\ \tau_5(L,S,J)&=\sum_G\psi_5(G,J,L,S)
		\end{align}
	</li>
	<li>
		Eliminating $S$. We have
		\begin{align}
		\psi_6(J,L,S)&=\phi_J(J,L,S)\cdot\tau_5(L,S,J) \\ \tau_6(J,L)&=\sum_S\psi_6(J,L,S)
		\end{align}
	</li>
	<li>
		Eliminating $L$. We have
		\begin{align}
		\psi_7(J,L)&=\tau_6(J,L) \\ \tau_7(J)&=\sum_L\psi_7(J,L)
		\end{align}
	</li>
</ul>

##### Semantics of Factors
Consider a factor produced as a product of some of the CPDs in a Bayesian network $\mathcal{B}$
\begin{equation}
\tau(\mathbf{W})=\prod_{i=1}^{k}P(Y_i\vert\text{Pa}\_{Y_i}),
\end{equation}
where
\begin{equation}
\mathbf{W}=\bigcup_{i=1}^{k}(\\{Y_i\\}\cup\text{Pa}\_{Y_i})
\end{equation}
Then, we can construct a Bayesian network $\mathcal{B}'$ and a disjoint partition $\mathbf{W}=\mathbf{Y}\cup\mathbf{Z}$ such that
\begin{equation}
\tau(\mathbf{W})=P_{\mathcal{B}'}(\mathbf{Y}\vert\mathbf{Z})
\end{equation}

#### Dealing with Evidence

### Complexity & Graph Structure of Variable Elimination

#### Complexity

#### Graph Structure
Since the inputs to $\text{Sum-Product-VE}$ is a set of factors $\Phi$, set of variables to eleminated $\mathbf{Z}$ with some ordering $\prec$, the algorithm does take into account whether the graph generating factors is directed, undirected or partly directed. Hence, it is simplest to consider the algorithm as acting on an undirected graph $\mathcal{H}$.

Let $\Phi$ be a set of factors, we define
\begin{equation}
\text{Scope}(\Phi)\doteq\bigcup_{\phi\in\Phi}\text{Scope}(\phi)
\end{equation}
to be the set of all variables appearing in any of the factors in $\Phi$. We define $\mathcal{H}\_\Phi$ to be the undirected graph whose nodes correspond to variables in $\text{Scope}(\Phi)$ and where we have an edge $X_i-X_j\in\mathcal{H}\_\Phi$ iff there exists a factor $\phi\in\Phi$ such that $X_i,X_j\in\text{Scope}(\phi)$.

In other words, $\mathcal{H}\_\Phi$ introduces a fully connected subgraph over the scope of each factor $\phi\in\Phi$. Hence, $\mathcal{H}_\Phi$ the minimal I-map for the distribution induced by $\Phi$.

**Proposition 2**: Let $P$ be a distribution defined by multiplying the factors in $\Phi$ then normalizing, i.e. let $\mathbf{X}=\text{Scope}(\Phi)$,
\begin{equation}
P(\mathbf{X})=\frac{1}{Z}\prod_{\phi\in\Phi}\phi,
\end{equation}
where
\begin{equation}
Z=\sum_\mathbf{X}\prod_{\phi\in\Phi}\phi
\end{equation}
Then $\mathcal{H}_\Phi$ is the minimal Markov network I-map for $P$, and $\Phi$ is a parameterization of this network that defines the distribution $P$.

**Proof**  
This follows direcly from [Theorem 16]({{< ref "pgm-representation#theorem16" >}}).

**Remark**: Using the arguments specified for converting from [Bayesian networks to Markov networks]({{< ref "pgm-representation#bn-2-mrf" >}}), it is worth remarking that
<ul id='number-list'>
	<li>
		For a set of factors $\Phi$ defined by a Bayesian network $\mathcal{G}$ (in this case, the partition function $Z=1$), $\mathcal{H}_\Phi$ is exactly the moralized graph of $\mathcal{G}$.
	</li>
	<li>
		In more specific, $\mathcal{H}_\Phi$ is the Markov network induced by a set of factors $\Phi[\mathbf{e}]$ defined by the reduction of the factors in a Bayesian network to some context $\mathbf{E}=\mathbf{e}$. In this case,
		<ul>
			<li>
				$\mathbf{X}=\text{Scope}(\Phi[\mathbf{e}])=\mathcal{X}\backslash\mathbf{E}$;
			</li>
			<li>
				the unnormalized product of factors is $P(\mathbf{X},\mathbf{e})$;
			</li>
			<li>
				the partition function is $P(\mathbf{e})$.
			</li>
		</ul>
	</li>
</ul>

##### The Induced Graph
Let $\Phi$ be a set of factors over $\mathcal{X}=\\{X_1,\ldots,X_n\\}$ and let $\prec$ be an elimination ordering for some $\mathbf{X}\subset\mathcal{X}$. The **induced graph** $\mathcal{I}_{\Phi,\prec}$ is an undirected graph over $\mathcal{X}$, where $X_i$ and $X_j$ are connected via an edge if they both appear in some intermediate factor $\psi$ generated by VE using $\prec$ as an elimination ordering.

**Theorem 3**: *Let $\mathcal{I}_{\Phi,\prec}$ be the induced graph for a set of factor $\Phi$ and some elimination ordering $\prec$. Then:*
<ul id='number-list' style='font-style: italic;'>
	<li>
		The scope of every factor generated during the VE process is a clique in $\mathcal{I}_{\Phi,\prec}$.
	</li>
	<li>
		Every <a href={{< ref "pgm-representation#max-clique" >}}>maximal clique</a> in $\mathcal{I}_{\Phi,\prec}$ is the scope of some intermediate factor in the computation.
	</li>
</ul>

**Proof**
<ul id='number-list'>
	<li>
		Consider a factor $\psi(Y_1,\ldots,Y_k)$ generated during the VE procedure. By definition of the induced graph, there must be an edge between each pair $(Y_i,Y_j)$, which implies that $Y_1,\ldots,Y_k$ form a clique.
	</li>
	<li>
		
	</li>
</ul>

###### Induced Width, Tree-width
The **width**, denoted $w$, of an induced graph is the number of nodes in the largest clique in the graph minus $1$.

The **induced width** $w_{\mathcal{K},\prec}$ of an ordering $\prec$ relative to a graph $\mathcal{K}$ (directed or undirected) is defined to be the width of the graph $\mathcal{I}_{\mathcal{I},\prec}$ induced by applying VE to $\mathcal{K}$ using the ordering $\prec$.

The **tree-width** of a graph $\mathcal{K}$ is defined to be its minimal induced width
\begin{equation}
w_\mathcal{K}^\*=\min_\prec w(\mathcal{I}\_{\mathcal{K},\prec})
\end{equation} 

### Clique Trees

## Variational Inference

## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes
