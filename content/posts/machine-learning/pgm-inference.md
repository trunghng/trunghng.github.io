---
title: "Probabilistic Graphical Models - Inference"
date: 2023-02-02T15:51:13+07:00
tags: [machine-learning, probabilistic-graphical-model]
math: true
eqn-number: true
---

## Exact Inference
**Problem setting**: We need to effectively compute the **conditional probability query**, which is the conditonal probability of variables $\mathbf{Y}$ given evidence $\mathbf{E}=\mathbf{e}$
\begin{equation}
P(\mathbf{Y}\vert\mathbf{E}=\mathbf{e})
\end{equation}

### Variable Elimination

#### Intuition
Consider a simple chain $A\rightarrow B\rightarrow C\rightarrow D$ with our iterest of computing $P(D)$, which can be computed by
\begin{equation}
P(D)=\sum_C\sum_B\sum_A P(A,B,C,D)\label{eq:int.1}
\end{equation}
This computation is ineffective, since it grows exponentially with the number of variables $n$. Speciffically, if each of variables $A,B,C,D$ ($n=4$) takes $k$ possible values, the time complexity required is $\mathcal{O}(k^{n-1})$.

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
which is justied by the limited scope of the CPD factors, e.g. the second step is due to the fact that $A\notin\text{Scope}(\phi_C)\cup\text{Scope}(\phi_D)$. This computation suggests that in general, our problem is to computing the value of an expression of the form
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
		To apply the algorithm to a Bayesian network $\mathcal{B}$ for computing $P_\mathcal{B}(\mathbf{Y})$, we begin by instatinating $\Phi$ to comprise all of the CPDs
		\begin{equation}
		\Phi=\{\phi_{X_i}\}_{i=1,\ldots,n},
		\end{equation}
		where $\phi_{X_i}=P(X_i\vert\text{Pa}_{X_i})$. Then, we apply the algorithm to the set $\mathbf{Z}=\mathcal{X}\backslash\mathbf{Y}$.
	</li>
	<li>
		Similarly, with a Markov network $\mathcal{H}$, we begin by instatiating $\Phi$ as the set of potential functions
		\begin{equation}
		\Phi=\{\phi_c\}_{c\in C},
		\end{equation}
		where $C$ be a set of cliques of $\mathcal{H}$. Analogously, we then apply the algorithm to the set $\mathbf{Z}=\mathcal{X}\backslash\mathbf{Y}$, which now returns an unnormalized factor over $\mathbf{Y}$. By dividing the partition function, we obtain $P_\mathcal{H}(\mathbf{Y})$.
	</li>
</ul>

**Example 2**: Consider the following Bayesian network with the goal is to computing the probability that the student got the job.
<figure>
	<img src="/images/pgm-inference/student-bn.png" alt="Student BN" style="display: block; margin-left: auto; margin-right: auto; width: 40%; height: 40%;"/>
	<figcaption><b>Figure 1</b> (taken from the <a href='#pgm-book'>PGM book</a>) A Bayesian network</figcaption>
</figure>

By chain rule of probability, we have that
\begin{align}
P(C,D,I,G,S,L,J,H)&=P(C)P(D\vert C)P(I)P(G\vert D,I)P(S\vert I)P(L\vert G)\nonumber \\\\ &\hspace{2cm}P(J\vert L,S)P(H\vert G,J) \\\\ &=\phi\_C(C)\phi\_D(D,C)\phi\_I(I)\phi\_G(G,D,I)\phi\_S(S,I)\phi\_L(L,G)\nonumber \\\\ &\hspace{2cm}\phi\_J(J,L,S)\phi\_H(H,G,J)
\end{align}
Consider the ordering: $C,D,I,H,G,S,L$. We step by step do the elimation procedure to each variable
<ul id='number-list'>
	<li>
		Eliminating $C$. We have
		\begin{equation}
		\tau_1(D)=\sum_C\phi_C(C)\cdot\phi_D(D,C)
		\end{equation}
	</li>
	<li>
		Eliminating $D$. We have
		\begin{equation}
		\tau_2(G,I)=\sum_{D}\phi_G(G,D,I)\cdot\tau_1(D)
		\end{equation}
	</li>
	<li>
		Eliminating $I$. We have
		\begin{equation}
		\tau_3(G,S)=\sum_I\phi_I(I)\cdot\phi_S(S,I)\cdot\tau_2(G,I)
		\end{equation}
	</li>
	<li>
		Eliminating $H$. We have
		\begin{equation}
		\tau_4(G,J)=\sum_H\phi_H(H,G,J),
		\end{equation}
		which is equal to $1$, since $\sum_H P(H\vert G,J)=1$
	</li>
	<li>
		Eliminating $G$. We have
		\begin{equation}
		\tau_5(L,S,J)=\sum_G\phi_L(L,G)\cdot\tau_3(G,S)\cdot\tau_4(G,J)
		\end{equation}
	</li>
	<li>
		Eliminating $S$. We have
		\begin{equation}
		\tau_6(J,L)=\sum_S\phi_J(J,L,S)\cdot\tau_5(L,S,J)
		\end{equation}
	</li>
	<li>
		Eliminating $L$. We have
		\begin{equation}
		\tau_7(J)=\sum_L\tau_6(J,L)
		\end{equation}
	</li>
</ul>

### Clique Trees

## Variational Inference

## References
[1] <span id='pgm-book'>Daphne Koller, Nir Friedman. [Probabilistic Graphical Models](https://mitpress.mit.edu/9780262013192/probabilistic-graphical-models/). The MIT Press.</span>

## Footnotes

