---
layout: post
title:  "Optimal Policy Existence"
date:   2021-07-10 13:03:00 +0700
tags: reinforcement-learning mathematics bellman-equation my-rl
description: Proof of the existence of optimal policy in finite Markov Decision Processes (MDPs)
comments: true
---
> In the previous post about [**Markov Decision Processes, Bellman equations**]({% post_url 2021-06-27-mdp-bellman-eqn %}), we mentioned that there exists a policy $\pi_\*$ that is better than or equal to all other policies. And now, we are here to prove it.  

<!-- excerpt-end -->
- [Preliminaries](#preliminaries)
	- [Norms](#norms)
	- [Contractions](#contractions)
	- [Banach's Fixed-point Theorem](#banach-fixed-pts)
	- [Bellman Operator](#bellman-operator)
- [Proof of the existence](#proof)
- [References](#references)
- [Footnotes](#footnotes)


Before catching the pokémon, we need to prepare ourselves some pokéball.  

## Preliminaries
{: #preliminaries}

### Norms
**Definition** (*Norm*)  
Given a vector space $\mathcal{V}\subseteq\mathbb{R}^d$, a function $f:\mathcal{V}\to\mathbb{R}^+\_0$ is a *norm* if and only if
1. If $f(v)=0$ for some $v\in\mathcal{V}$, then $v=0$
2. For any $\lambda\in\mathbb{R},v\in\mathcal{V},f(\lambda v)=\|\lambda\|v$
3. For any $u,v\in\mathbb{R}, f(u+v)\leq f(u)+f(v)$

**Examples** (*Norm*)
1. $\ell^p$ norms: for $p\geq 1$,
\begin{equation}
\Vert v\Vert\_p=\left(\sum_{i=1}^{d}\|v_i\|^p\right)^{1/p}
\end{equation}
2. $\ell^\infty$ norms:
\begin{equation}
\Vert v\Vert_\infty=\max_{1\leq i\leq d}\|v_i\|
\end{equation}
3. $\ell^{\mu,p}$: the weighted variants of these norm are defined as
\begin{equation}
\Vert v\Vert_p=\begin{cases}\left(\sum_{i=1}^{d}\frac{\|v_i\|^p}{w_i}\right)^{1/p}&\text{if }1\leq p\<\infty\\\\ \max_{1\leq i\leq d}\frac{\|v_i\|}{w_i}&\text{if }p=\infty\end{cases}
\end{equation}
4. $\ell^{2,P}$: the matrix-weighted 2-norm is defined as
\begin{equation}
\Vert v\Vert^2\_P=v^\intercal Pv
\end{equation}
Similarly, we can define norms over spaces of functions. For example, if $\mathcal{V}$ is the vector space of functions over domain $\mathcal{X}$ which are *uniformly bounded*[^1], then
\begin{equation}
\Vert f\Vert_\infty=\sup_{x\in\mathcal{X}}\vert f(x)\vert
\end{equation}

**Definition** (*Convergence in norm*)  
Let $\mathcal{V}=(\mathcal{V},\Vert\cdot\Vert)$ be a *normed vector space*[^2]. Let $v_n\in\mathcal{V}$ is a sequence of vectors ($n\in\mathbb{N}$). The sequence ($v_n,n\geq 0$) is said to *converge to* $v\in\mathcal{V}$ in the norm $\Vert\cdot\Vert$, denoted as $v_n\to\_{\Vert\cdot\Vert}v$ if
\begin{equation}
\lim_{n\to\infty}\Vert v_n-v\Vert=0,
\end{equation}
<br/>

**Definition** (*Cauchy sequence*[^3])  
Let ($v_n;n\geq 0$) be a sequence of vectors of a normed vector space $\mathcal{V}=(\mathcal{V},\Vert\cdot\Vert)$. Then $v_n$ is called a *Cauchy sequence* if
\begin{equation}
\lim_{n\to\infty}\sup_{m\geq n}\Vert v_n-v_m\Vert=0
\end{equation}
Normed vector spaces where all Cauchy sequences are convergent are special: we can find examples of normed vector spaces such that some of the Cauchy sequences in the vector space do not have a limit.  
<br/>

**Definition** (*Completeness*)  
A normed vector space $\mathcal{V}=(\mathcal{V},\Vert\cdot\Vert)$ is called *complete* if every Cauchy sequence in $\mathcal{V}$ is convergent in the norm of the vector space.  


### Contractions
**Definition** (*Lipschitz function*, *Contraction*)   
Let $\mathcal{V}=(\mathcal{V},\Vert\cdot\Vert)$ be a normed vector space. A mapping $\mathcal{T}:\mathcal{V}\to\mathcal{V}$ is called *L-Lipschitz* if for any $u,v\in\mathcal{V}$,
\begin{equation}
\Vert\mathcal{T}u-\mathcal{T}v\Vert\leq L\Vert u-v\Vert
\end{equation}
A mapping $\mathcal{T}$ is called a *non-expansion* if it is *Lipschitzian* with $L\leq 1$. It is called a *contraction* if it is *Lipschitzian* with $L<1$. In this case, $L$ is called the *contraction factor of* $\mathcal{T}$ and $\mathcal{T}$ is called an *L-contraction*.  

**Remark**  
If $\mathcal{T}$ is *Lipschitz*, it is also continuous in the sense that if $v_n\to_{\Vert\cdot\Vert}v$, then also $\mathcal{T}v_n\to_{\Vert\cdot\Vert}\mathcal{T}v$. This is because $\Vert\mathcal{T}v_n-\mathcal{T}v\Vert\leq L\Vert v_n-v\Vert\to 0$ as $n\to\infty$.  


### Banach's Fixed-point Theorem
{: #banach-fixed-pts}
**Definition** (*Banach space*)  
A complete, normed vector space is called a *Banach space*.  
<br/>

**Definition** (*Fixed point*)  
Let $\mathcal{T}:\mathcal{V}\to\mathcal{V}$ be some mapping. The vector $v\in\mathcal{V}$ is called a *fixed point of* $\mathcal{T}$ if $\mathcal{T}v=v$.  
<br/>

**Theorem** (*Banach's fixed-point*)[^4]      
Let $\mathcal{V}$ be a Banach space and $\mathcal{T}:\mathcal{V}\to\mathcal{V}$ be a $\gamma$-contraction mapping. Then
1. $\mathcal{T}$ admits a *unique fixed point* $v$.
2. For any $v_0\in\mathcal{V}$, if $v_{n+1}=\mathcal{T}v_n$, then $v_n\to_{\Vert\cdot\Vert}v$ with a *geometric convergence rate*:
\begin{equation}
\Vert v_n-v\Vert\leq\gamma^n\Vert v_0-v\Vert
\end{equation}


### Bellman Operator
Previously, we defined [Bellman equation]({% post_url 2021-06-27-mdp-bellman-eqn %}#bellman-equations) for state-value function $v_\pi(s)$ as:
\begin{align}
v_\pi(s)&=\sum_{a\in\mathcal{A}}\pi(a|s)\sum_{s'\in\mathcal{S},r}p(s',r|s,a)\left[r+\gamma v_\pi(s')\right] \\\\\text{or}\quad v_\pi(s)&=\sum_{a\in\mathcal{A}}\pi(a|s)\left(\mathcal{R}^a_s+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_\pi(s')\right)\tag{1}\label{1}
\end{align}
If we let
\begin{align}
\mathcal{P}^\pi_{ss'}&=\sum_{a\in\mathcal{A}}\pi(a|s)\mathcal{P}^a_{ss'}; \\\\\mathcal{R}^\pi_s&=\sum_{a\in\mathcal{A}}\pi(a|s)\mathcal{R}^a_s
\end{align}
then we can rewrite \eqref{1} in another form as
\begin{equation}
v_\pi(s)=\mathcal{R}^\pi_s+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}v_\pi(s')\tag{2}\label{2}
\end{equation}
<br/>
**Definition** (*Bellman operator*)  
We define the *Bellman operator* underlying $\pi,\mathcal{T}:\mathbb{R}^\mathcal{S}\to\mathbb{R}^\mathcal{S}$, by:
\begin{equation}
(\mathcal{T}^\pi v)(s)=\mathcal{R}^\pi_s+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}v(s')
\end{equation}
<br/>
With the help of $\mathcal{T}^\pi$, equation \eqref{2} can be rewrite as:
\begin{equation}
\mathcal{T}^\pi v_\pi=v_\pi\tag{3}\label{3}
\end{equation}
Similarly, we can rewrite the *Bellman optimality equation for* $v_\*$
\begin{align}
v_\*(s)&=\max_{a\in\mathcal{A}}\sum_{s'\in\mathcal{S},r}p(s',r|s,a)\left[r+\gamma v_\*(s')\right] \\\\ &=\max_{a\in\mathcal{A}}\left(\mathcal{R}^a_s+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v_\*(s')\right)\tag{4}\label{4}
\end{align}
and thus, we can define the *Bellman optimality operator* $\mathcal{T}^\*:\mathcal{R}^\mathcal{S}\to\mathcal{R}^\mathcal{S}$, by:
\begin{equation}
(\mathcal{T}^\* v)(s)=\max_{a\in\mathcal{A}}\left(\mathcal{R}^a_s+\gamma\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}v(s')\right)
\end{equation}
And thus, with the help of $\mathcal{T}^\*$, we can rewrite the equation \eqref{4} as:
\begin{equation}
\mathcal{T}^\*v_\*=v_\*\tag{5}\label{5}
\end{equation}
<br/>
Now everything is all set, we can move on to the next step.  

## Proof of the existence
{: #proof}
Let $B(\mathcal{S})$ be the space of *uniformly bounded functions* with domain $\mathcal{S}$:
\begin{equation}
B(\mathcal{S})=\\{v:\mathcal{S}\to\mathbb{R}:\Vert v\Vert_\infty<+\infty\\}
\end{equation}
We will view $B(\mathcal{S})$ as a normed vector space with the norm $\Vert\cdot\Vert_\infty$.

It is easily seen that $(B(\mathcal{S}),\Vert\cdot\Vert_\infty)$ is complete: If $(v_n;n\geq0)$ is a Cauchy sequence in it then for any $s\in\mathcal{S}$, $(v_n(s);n\geq0)$ is also a Cauchy sequence over the reals. Denoting by $v(s)$ the limit of $(v_n(s))$, we can show that $\Vert v_n-v\Vert_\infty\to0$. Vaguely speaking, this holds because $(v_n;n\geq0)$ is a Cauchy sequence in the norm $\Vert\cdot\Vert_\infty$, so the rate of convergence of $v_n(s)$ to $v(s)$ is independent of $s$. 

Let $\pi$ be some stationary policy. We have that $\mathcal{T}^\pi$ is *well-defined* since: if $u\in B(\mathcal{S})$, then also $\mathcal{T}^\pi u\in B(S)$.

From equation \eqref{3}, we have that $v_\pi$ is a fixed point to $\mathcal{T}^\pi$.

We also have that $\mathcal{T}^\pi$ is a $\gamma$-contraction in $\Vert\cdot\Vert_\infty$ since for any $u, v\in B(\mathcal{S})$,
\begin{align}
\Vert\mathcal{T}^\pi u-\mathcal{T}^\pi v\Vert_\infty&=\gamma\max_{s\in\mathcal{S}}\left|\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}\left(u(s')-v(s')\right)\right| \\\\ &\leq\gamma\max_{s\in\mathcal{S}}\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}\big|u(s')-v(s')\big| \\\\ &\leq\gamma\max_{s\in\mathcal{S}}\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}\big\Vert u-v\big\Vert_\infty \\\\ &=\gamma\Vert u-v\Vert_\infty,
\end{align}
where the last line follows from $\sum_{s'\in\mathcal{S}}\mathcal{P}^\pi_{ss'}=1$.

It follows that in order to find $v_\pi$, we can construct the sequence $v_0,\mathcal{T}^\pi v_0,(\mathcal{T}^\pi)^2 v_0,\dots$, which, by Banach's fixed-point theorem will converge to $v_\pi$ at a geometric rate.

From the definition \eqref{5} of $\mathcal{T}^\*$, we have that $\mathcal{T}^\*$ is well-defined.

Using the fact that $\left|\max_{a\in\mathcal{A}}f(a)-\max_{a\in\mathcal{A}}g(a)\right|\leq\max_{a\in\mathcal{A}}\left|f(a)-g(a)\right|$, similarly, we have:
\begin{align}
\Vert\mathcal{T}^\*u-\mathcal{T}^\*v\Vert_\infty&\leq\gamma\max_{(s,a)\in\mathcal{S}\times\mathcal{A}}\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}\left|u(s')-v(s')\right| \\\\ &\leq\gamma\max_{(s,a)\in\mathcal{S}\times\mathcal{A}}\sum_{s'\in\mathcal{S}}\mathcal{P}^a_{ss'}\Vert u-v\Vert_\infty \\\\ &=\gamma\Vert u-v\Vert_\infty,
\end{align}
which tells us that $\mathcal{T}^\*$ is a $\gamma$-contraction in $\Vert\cdot\Vert_\infty$.
<br/>

**Theorem**  
Let $v$ be the fixed point of $\mathcal{T}^\*$ and assume that there is policy $\pi$ which is greedy w.r.t $v:\mathcal{T}^\pi v=\mathcal{T}^\* v$. Then $v=v_\*$ and $\pi$ is an optimal policy.

***Proof***  
Pick any stationary policy $\pi$. Then $\mathcal{T}^\pi\leq\mathcal{T}^\*$ in the sense that for any function $v\in B(\mathcal{S})$, $\mathcal{T}^\pi v\leq\mathcal{T}^\* v$ holds ($u\leq v$ means that $u(s)\leq v(s),\forall s\in\mathcal{S}$).

Hence, for all $n\geq0$,
\begin{equation}
v_\pi=\mathcal{T}^\pi v_\pi\leq\mathcal{T}^\*v_\pi\leq(\mathcal{T}^\*)^2 v_\pi\leq\dots\leq(\mathcal{T}^\*)^n v_\pi
\end{equation}
or
\begin{equation}
v_\pi\leq(\mathcal{T}^\*)^n v_\pi
\end{equation}
Since $\mathcal{T}^\*$ is a contraction, the right-hand side converges to $v$, the unique fixed point of $\mathcal{T}^\*$. Thus, $v_\pi\leq v$. And since $\pi$ was arbitrary, we obtain that $v_\*\leq v$.

Pick a policy $\pi$ such that $\mathcal{T}^\pi v=\mathcal{T}^\*v$, then $v$ is also a fixed point of $\mathcal{V}^\pi$. Since $v_\pi$ is the unique fixed point of $\mathcal{T}^\pi$, we have that $v=v_\pi$, which shows that $v_\*=v$ and that $\pi$ is an optimal policy.

## References
[1] Csaba Szepesvári. [Algorithms for Reinforcement Learning](https://www.amazon.com/Algorithms-Reinforcement-Synthesis-Artificial-Intelligence/dp/1608454924).  

[2] A. Lazaric. [Markov Decision Processes and Dynamic Programming](http://researchers.lille.inria.fr/~lazaric/Webpage/MVA-RL_Course14_files/slides-lecture-02-handout.pdf).  

[3] [What is the Bellman operator in reinforcement learning?](https://ai.stackexchange.com/a/11133). AI.StackExchange. 

[4] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition). MIT press, 2018.  

[5] [Normed vector space](https://en.wikipedia.org/wiki/Normed_vector_space). Wikipedia.

## Footnotes
[^1]: A function is called *uniformly bounded* exactly when $\Vert f\Vert_\infty<+\infty$.
[^2]: A *normed vector space* is a vector space over the real or complex number, on which a norm is defined.
[^3]: The details of *sequences* are mentioned in another [note]({% post_url 2021-09-06-infinite-series-of-constants %}#convergent-sequences).
[^4]: ***Proof***  
	Pick any $v_0\in\mathcal{V}$ and define $v_n$ as in the statement of the theorem. a. We first demonstrate that $(v_n)$ converges to some vector. b. Then we will show that this vector is a fixed point to $\mathcal{T}$. c. Finally, we show that $\mathcal{T}$ has a single fixed point. Assume that $\mathcal{T}$ is a $\gamma$-contraction.  
	a. To show that $(v_n)$ converges, it suffices  to show that $(v_n)$ is a Cauchy sequence. We have:
	\begin{align}
	\Vert v_{n+1}-v_n\Vert&=\Vert\mathcal{T}v_{n}-\mathcal{T}v_{n-1}\Vert \\\\ &\leq\gamma\Vert v_{n}-v_{n-1}\Vert \\\\ &\quad\vdots \\\\ &\leq\gamma^n\Vert v_1-v_0\Vert
	\end{align}
	From the properties of norms, we have:
	\begin{align}
	\Vert v_{n+k}-v_n\Vert&\leq\Vert v_{n+1}-v_n\Vert+\dots+\Vert v_{n+k}-v_{n+k-1}\Vert \\\\ &\leq\left(\gamma^n+\dots+\gamma^{n+k-1}\right)\Vert v_1-v_0\Vert \\\\ &=\gamma^n\dfrac{1-\gamma^{k}}{1-\gamma}\Vert v_1-v_0\Vert
	\end{align}
	and so
	\begin{equation}
	\lim_{n\to\infty}\sup_{k\geq0}\Vert v_{n+k}-v_n\Vert=0,
	\end{equation}
	shows us that $(v_n;n\geq0)$ is indeed a Cauchy sequence. Let $v$ be its limit.  
	b. Recall that the definition of the sequence $(v_n;n\geq0)$
	\begin{equation}
	v_{n+1}=\mathcal{T}v_n
	\end{equation}
	Taking the limes as $n\to\infty$ of both sides, one the one hand, we get that $v_{n+1}\to _{\Vert\cdot\Vert}v$. On the other hand, $\mathcal{T}v_n\to _{\Vert\cdot\Vert}\mathcal{T}v$, since $\mathcal{T}$ is a contraction, hence it is continuous. Therefore, we must have $v=\mathcal{T}v$, which tells us that $v$ is a fixed point of $\mathcal{T}$.  
	c. Let us assume that $v,v'$ are both fixed points of $\mathcal{T}$. Then,
	\begin{align}
	\Vert v-v'\Vert&=\Vert\mathcal{T}v-\mathcal{v'}\Vert \\\\ &\leq\gamma\Vert v-v'\Vert \\\\ \text{or}\quad(1-\gamma)\Vert v-v'\Vert&\leq0
	\end{align}
	Thus, we must have that $\Vert v-v'\Vert=0$. Therefore, $v-v'=0$ or $v=v'$.  
	And finally,
	\begin{align}
	\Vert v_n-v\Vert&=\Vert\mathcal{T}v\_{n-1}-\mathcal{T}v\Vert \\\\ &\leq\gamma\Vert v\_{n-1}-v\Vert \\\\ &\quad\vdots \\\\ &\leq\gamma^n\Vert v_0-v\Vert
	\end{align}