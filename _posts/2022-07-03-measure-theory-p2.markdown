---
layout: post
title:  "Measure theory - II: Lebesgue measure"
date:   2022-07-03 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory lebesgue-measure
description: Note on measure theory part 2
comments: true
eqn-number: true
---
> Part II of the measure theory series. Materials are mostly taken from [Tao's book]({% post_url 2022-07-03-measure-theory-p2 %}#taos-book), except for some needed notations extracted from [Stein's book]({% post_url 2022-07-03-measure-theory-p2 %}#steins-book).
<!-- excerpt-end -->

- [Lebesgue measure](#lebesgue-measure)
	- [Properties of Lebesgue outer measure](#lebesgue-outer-measure-properties)
		- [Finite additivity for separated sets](#fnt-add-spt-sets)
		- [Outer measure of elementary sets](#outer-measure-elem-sets)
		- [Finite additivity for almost disjoint boxes](#fnt-add-alm-dsjnt-boxes)
		- [Outer measure of countable unions of almost disjoint boxes](#outer-msr-cntbl-uni-alm-dsjnt-boxes)
		- [Open sets as countable unions of almost disjoint boxes](#open-sets-cntbl-uni-alm-dsjnt-boxes)
		- [Outer measure of open sets](#outer-msr-open-sets)
		- [Outer measure of arbitrary sets - Outer regularity](#outer-msr-arb-sets)
	- [Lebesgue measurability](#lebesgue-measurability)
		- [Existence of Lebesgue measurable sets](#exist-lebesgue-msr-sets)
		- [Criteria for measurability](#crt-msrb)
		- [The measure axioms](#msr-axiom)
		- [Monotone convergence theorem for measurable sets](#mnt-cvg-theorem-msr-sets)
		- [Dominated convergence theorem for measurable sets](#dmnt-cvg-theorem-msr-sets)
		- [Inner regularity](#inn-rglr)
		- [Criteria for finite measure](#crt-fnt-msr)
		- [Carathéodory criterion, one direction](#caratheodory-crt)
		- [Inner measure](#inn-msr)
		- [Translation invariance](#trans-inv)
		- [Change of variables](#change-vars)
		- [Uniqueness of Lebesgue measure](#uniq-lebesgue-msr)
	- [Non-measurable sets](#non-measurable-sets)
- [References](#references)
- [Footnotes](#footnotes)

## Lebesgue measure
{: #lebesgue-measure}
Recall that the Jordan outer measure of a set $E\subset\mathbb{R}^d$ has been defined as
\begin{equation}
m^{\*,(J)}(E)\doteq\inf_{B\supset E;B\text{ elementary}}m(B)
\end{equation}
From the finite additivity and subadditivity of elementary measure, we can also write the Jordan outer measure as
\begin{equation}
m^{\*,(J)}(E)\doteq\inf_{B_1\cup\dots\cup B_k\supset E;B_1,\dots,B_k\text{ boxes}}\vert B_1\vert+\dots+\vert B_k\vert,
\end{equation}
which means the Jordan outer measure is the infimal cost required to cover $E$ by a finite union of boxes. By replacing the finite union of boxes by a countable union of boxes, we obtain the **Lebesgue outer measure** $m^{\*}(E)$ of $E$:
\begin{equation}
m^{\*}(E)\doteq\inf_{\bigcup_{n=1}^{\infty}B_n\supset E;B_1,B_2,\dots\text{ boxes}}\sum_{n=1}^{\infty}\vert B_n\vert,
\end{equation}
which is be seen as the infimal cost required to cover $E$ by a countable union of boxes.

A set $E\subset\mathbb{R}^d$ is said to be **Lebesgue measurable** if, for every $\varepsilon>0$, there exists an open set $U\subset\mathbb{R}^d$ containing $E$ such that $m^{\*}(U\backslash E)\leq\varepsilon$. If $E$ is Lebesgue measurable, we refer to
\begin{equation}
m(E)\doteq m^{\*}(E)
\end{equation}
as the **Lebesgue measure** of $E$.

### Properties of Lebesgue outer measure
{: #lebesgue-outer-measure-properties}
**Remark 1**. (**The outer measure axioms**)
<ul id='roman-list'>
	<li><b>Empty set</b>. $m^*(\emptyset)=0$.</li>
	<li><b>Monotonicity</b>. If $E\subset F\subset\mathbb{R}^d$, then $m^*(E)\leq m^*(F)$.</li>
	<li><b>Countable subadditivity</b>. If $E_1,E_2,\ldots\subset\mathbb{R}^d$ is a countable sequence of sets, then $m^*\left(\bigcup_{n=1}^{\infty}E_n\right)\leq\sum_{n=1}^{\infty}m^*(E_n)$.</li>
</ul>

**Proof**  
<ul id='roman-list'>
	<li>This follows from the definition of Lebesgue outer measure.</li>
	<li>
		Since $E\subset F\subset\mathbb{R}^d$, then any set containing $F$ also includes $E$, but not every set having $E$ contains $F$. That means
		\begin{equation}
		\left\{\sum_{n=1}^{\infty}\vert B_n\vert:E\subset\bigcup_{n=1}^{\infty}B_n;B_n\text{ boxes}\right\}\supset\left\{\sum_{n=1}^{\infty}\vert B_n\vert:F\subset\bigcup_{n=1}^{\infty}B_n;B_n\text{ boxes}\right\}
		\end{equation}
		Thus,
		\begin{equation}
		\inf\left\{\sum_{n=1}^{\infty}\vert B_n\vert:E\subset\bigcup_{n=1}^{\infty}B_n;B_n\text{ boxes}\right\}\leq\inf\left\{\sum_{n=1}^{\infty}\vert B_n\vert:F\subset\bigcup_{n=1}^{\infty}B_n;B_n\text{ boxes}\right\}
		\end{equation}
		or
		\begin{equation}
		m^*(E)< m^*(F)
		\end{equation}
	</li>
	<li>
		By the definition of Lebesgue outer measure, for any positive integer $i$, we have
		\begin{equation}
		m^*(E_i)=\inf_{\bigcup_{n=1}^{\infty}B_n\supset E_i;B_1,B_2,\ldots\text{ boxes}}\sum_{n=1}^{\infty}\vert B_n\vert
		\end{equation}
		Thus, by definition of infimum and by <span markdown=1>[axiom of countable choice]({% post_url 2022-06-16-measure-theory-p1 %}#countable-choice-axiom)</span>, for each $E_i$ in the sequence $(E_n)_{n\in\mathbb{N}}$, there exists a family of boxes $B_{i,1},B_{i,2},\ldots$ in the doubly sequence $(B_{i,j})_{(i,j)\in\mathbb{N}^2}$ covering $E_i$ such that
		\begin{equation}
		\sum_{j=1}^{\infty}\vert B_{i,j}\vert\lt m^*(E_i)+\frac{\varepsilon}{i},
		\end{equation}
		for any $\varepsilon>0$, and for $i=1,2,\ldots$. Plus, we also have
		\begin{equation}
		\bigcup_{n=1}^{\infty}E_n\subset\bigcup_{i=1}^{\infty}\bigcup_{j=1}^{\infty}B_{i,j}
		\end{equation}
		Moreover, by the <span markdown=1>[Tonelli's theorem for series]({% post_url 2022-06-16-measure-theory-p1 %}#tonelli-theorem)</span>, we have
		\begin{equation}
		\bigcup_{i=1}^{\infty}\bigcup_{j=1}^{\infty}B_{i,j}=\bigcup_{(i,j)\in\mathbb{N}^2}B_{i,j}
		\end{equation}
		Therefore once again, by definition of outer measure and definition of infimum, we obtain
		\begin{align}
		m^*\left(\bigcup_{n=1}^{\infty}E_n\right)&=\inf_{\bigcup_{(i,j)\in\mathbb{N}^2}B_{i,j}}\sum_{i=1}^{\infty}\sum_{j=1}^{\infty}\vert B_{i,j}\vert\leq\sum_{i=1}^{\infty}\sum_{j=1}^{\infty}\vert B_{i,j}\vert \\\\ &\lt\sum_{i=1}^{\infty}m^*(E_i)+\frac{\varepsilon}{2^i}=\sum_{i=1}^{\infty}m^*(E_i)+\varepsilon
		\end{align}
		And since $\varepsilon>0$ was arbitrary, we can conclude that
		\begin{equation}
		m^*\left(\bigcup_{n=1}^{\infty}E_n\right)\leq\sum_{i=n}^{\infty}m^*(E_n)
		\end{equation}
	</li>
</ul>

**Corollary 2**  
Combining empty set with countable subadditivity axiom gives us the **finite subadditivity** property
\begin{equation}
m^{\*}\left(E_1\cup\ldots\cup E_k\right)\leq m^{\*}(E_1)+\ldots+m^{\*}(E_k),\hspace{1cm}\forall k\geq 0
\end{equation}

#### Finite additivity for separated sets
{: #fnt-add-spt-sets}
**Lemma 3**    
*Let $E,F\subset\mathbb{R}^d$ be such that $\text{dist}(E,F)>0$, where
\begin{equation}
\text{dist}(E,F)\doteq\inf\left\\{\vert x-y\vert:x\in E,y\in F\right\\}
\end{equation}
is the distance between $E$ and $F$. Then $m^\*(E\cup F)=m^\*(E)+m^\*(F)$.*

**Proof**  
From subadditivity property, we have $m^\*(E\cup F)\leq m^\*(E)+m^\*(F)$. Then it suffices to prove the inverse, that
\begin{equation}
m^\*(E\cup F)\geq m^\*(E)+m^\*(F)
\end{equation}
Let $\varepsilon>0$. By definition of Lebesgue outer measure, we can cover $E\cup F$ by a countable family $B_1,B_2,\ldots$ of boxes such that
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert\leq m^\*(E\cup F)+\varepsilon
\end{equation}
Suppose it was the case that each box intersected at most one of $E$ and $F$. Then we could divide this family into two subfamilies $B_1',B_2',\ldots$ and $B_1\'\',B_2\'\',B_3\'\',\ldots$, the first of which covered $E$, while the second of which covered $F$. From definition of Lebesgue outer measure, we have
\begin{equation}
m^\*(E)\leq\sum_{n=1}^{\infty}\vert B_n'\vert
\end{equation}
and
\begin{equation}
m^\*(F)\leq\sum_{n=1}^{\infty}\vert B_n\'\'\vert
\end{equation}
Summing up these two equation, we obtain
\begin{equation}
m^\*(E)+m^\*(F)\leq\sum_{n=1}^{\infty}\vert B_n\vert
\end{equation}
and thus
\begin{equation}
m^\*(E)+m^\*(F)\leq m^\*(E\cup F)+\varepsilon
\end{equation}
Since $\varepsilon$ was arbitrary, this gives $m^\*(E)+m^\*(F)\leq m^\*(E\cup F)$ as required.

Now we consider the case that some of the boxes $B_n$ intersect both $E$ and $F$. 

Since given any $r>0$, we can always partition a box $B_n$ into a finite number of smaller boxes, each of which has diameter[^1] at most $r$, with the total volume of these sub-boxes equal to the volume of the original box $B_n$. Therefore, given any $r>0$, we may assume without loss of generality that the boxes $B_1,B_2,\ldots$ covering $E\cup F$ have diameter at most $r$. Or in particular, we may assume that all such boxes have diameter strictly less than $\text{dist}(E,f)$.

Once we do this, then it is no longer possible for any box to intersect both $E$ and $F$, which allows the previous argument be applicable.

**Example 1**  
Let $E,F\subset\mathbb{R}^d$ be disjoint closed sets, with at least one of $E,F$ being compact. Then $\text{dist}(E,F)>0$.

**Proof**  

#### Outer measure of elementary sets
{: #outer-measure-elem-sets}
**Lemma 4**    
*Let $E$ be an elementary set. Then the Lebesgue outer measure of $E$ is equal to the elementary measure of $E$:*
\begin{equation}
m^\*(E)=m(E)
\end{equation}

**Proof**  
Since
\begin{equation}
m^\*(E)\leq m^{\*,(J)}(E)=m(E),
\end{equation}
then it suffices to show that
\begin{equation}
m(E)\leq m^\*(E)
\end{equation}
We first consider the case that $E$ is closed. Since $E$ is elementary, $E$ is also bounded, which implies that $E$ is compact.

Let $\varepsilon>0$ be arbitrary, then we can find a countable family $B_1,B_2,\ldots$ of boxes that cover $E$
\begin{equation}
E\subset\bigcup_{n=1}^{\infty}B_n,
\end{equation}
such that
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert\leq m^\*(E)+\varepsilon
\end{equation}
We have that for each box $B_n$, we can find an open box $B_n'$ containing $B_n$ such that
\begin{equation}
\vert B_n'\vert\leq\vert B_n\vert+\frac{\varepsilon}{2^n}
\end{equation}
The $B_n'$ still cover $E$ and we have
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n'\vert\leq\sum_{n=1}^{\infty}\left(\vert B_n\vert+\frac{\varepsilon}{2^n}\right)=\left(\sum_{n=1}^{\infty}\vert B_n\vert\right)+\varepsilon\leq m^\*(E)+2\varepsilon\label{eq:lemma5.1}
\end{equation}
As the $B_n'$ are open, apply the <span markdown=1>[**Heine-Borel theorem**]({% post_url 2022-06-16-measure-theory-p1 %}#heine-borel-theorem), we obtain
\begin{equation}
E\subset\bigcup_{n=1}^{N}B_n',
\end{equation}
for some finite $N$. Thus, using the finite subadditivity property of elementary measure, combined with the result \eqref{eq:lemma5.1}, we obtain
\begin{equation}
m(E)\leq\sum_{n=1}^{N}m(B_n')\leq m^\*(E)+2\varepsilon
\end{equation}
And since $\varepsilon>0$ was arbitrary, we can conclude that
\begin{equation}
m(E)\leq m^\*(E)
\end{equation}
Now we turn to considering the case that $E$ is not closed. Then we can write $E$ as the finite union of disjoint boxes
\begin{equation}
E=Q_1\cup\ldots\cup Q_k,
\end{equation}
which need not be closed.

Analogy to before, we have that for every $\varepsilon>0$ and every $1\leq j\leq k$, we can find a closed sub-box $Q_j'$ of $Q_j$ such that
\begin{equation}
\vert Q_j'\vert\geq\vert Q_j\vert-\frac{\varepsilon}{k}
\end{equation}
Then $E$ now contains the finite union of $Q_1'\cup\ldots\cup Q_k'$ disjoint closed boxes, which is a closed elementary set. By the finite additivity property of elementary measure, the monotonicity property of Lebesgue measure, combined with the result we have proved in the first case, we have
\begin{align}
m^\*(E)&\geq m^\*(Q_1'\cup\ldots\cup Q_k') \\\\ &=m(Q_1'\cup\ldots\cup Q_k') \\\\ &=m(Q_1')+\ldots+m(Q_k') \\\\ &\geq m(Q_1)+\ldots+m(Q_k)-\varepsilon \\\\ &= m(E)-\varepsilon,
\end{align}
for every $\varepsilon>0$. And since $\varepsilon>0$ was arbitrary, our claim has been proved.

**Corollary 6**  
From the lemma above and the monotonicity property, 
for every $E\in\mathbb{R}^d$, we have
\begin{equation}
m_{\*,(J)}(E)\leq m^{\*}(E)\leq m^{\*,(J)}(E)\label{eq:cor6.1}
\end{equation}

**Corollary 7**  
Not every bounded open set or compact set (bounded closed) is Jordan measurable.

**Proof**  
Consider the countable set $\mathbf{Q}\cap[0,1]$, which we enumerate as $\\{q_1,q_2,\ldots\\}$. Let $\varepsilon>0$ be a small number, and consider that
\begin{equation}
U\doteq\bigcup_{n=1}^{\infty}(q_n-\varepsilon/2^n,q_n+\varepsilon/2^n),
\end{equation}
which is a union of open sets and thus is open. On the other hand, by countable subadditivity property of Lebesgue outer measure, we have
\begin{align}
m^{\*}(U)&=m^{\*}\left(\sum_{n=1}^{\infty}\left(q_n-\frac{\varepsilon}{2^n},q_n+\frac{\varepsilon}{2^n}\right)\right) \\\\ &\leq\sum_{n=1}^{\infty}m^{\*}\left(q_n-\frac{\varepsilon}{2^n},q_n+\frac{\varepsilon}{2^n}\right) \\\\ &=\sum_{n=1}^{\infty}\frac{2\varepsilon}{2^n}=2\varepsilon
\end{align}
As $U$ dense in $[0,1]$ (i.e.,$\overline{U}$ contains $[0,1]$), we have
\begin{equation}
m^{\*}(U)=m^{\*,(J)}(\overline{U})\geq m^{\*,(J)}([0,1])=1
\end{equation}
Then for $\varepsilon\lt 1$, we have that
\begin{equation}
m^{\*}(U)\lt 1\leq m^{\*,(J)}(U)
\end{equation}
Combining with \eqref{eq:cor6.1}, we obtain that the bounded open set $U$ is not Jordan measurable.

#### Finite additivity for almost disjoint boxes
{: #fnt-add-alm-dsjnt-boxes}
Two boxes are **almost disjoint** if their interiors are disjoint, e.g., $[0,1]$ and $[1,2]$ are almost disjoint. If a box has the same elementary as its interior, we see that the finite additivity property
\begin{equation}
m(B_1\cup\ldots\cup B_n)=\vert B_1\vert+\ldots+\vert B_n\vert\label{eq:faadb.1}
\end{equation}
also holds for almost disjoint boxes $B_1,\ldots,B_n$.

#### Outer measure of countable unions of almost disjoint boxes
{: #outer-msr-cntbl-uni-alm-dsjnt-boxes}
**Lemma 8**  
*Let $E=\bigcup_{n=1}^{\infty}B_n$ be a countable union of almost disjoint boxes $B_1,B_2,\ldots$. Then*
\begin{equation}
m^\*(E)=\sum_{n=1}^{\infty}\vert B_n\vert
\end{equation}
Thus, for example, $\mathbb{R}^d$ has an infinite outer measure.

**Proof**  
From countable subadditivity property of Lebesgue measure and **Lemma 5**, we have
\begin{equation}
m^\*(E)\leq\sum_{n=1}^{\infty}m^\*(B_n)=\sum_{n=1}^{\infty}\vert B_n\vert,
\end{equation}
so it suffices to show that
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert\leq m^\*(E)
\end{equation}
Since for each integer $N$, $E$ contains the elementary set $B_1\cup\ldots\cup B_N$, then by monotonicity property and **Lemma 5**
\begin{align}
m^\*(E)&\geq m^\*(B_1\cup\ldots\cup B_N)=m(B_1\cup\ldots\cup B_N)
\end{align}
And thus by \eqref{eq:faadb.1}, we have
\begin{equation}
\sum_{n=1}^{N}\vert B_n\vert\leq m^\*(E)
\end{equation}
Letting $N\to\infty$ we obtain the claim.

**Corollary 9**  
If $E=\bigcup_{n=1}^{\infty}B_n=\bigcup_{n=1}^{\infty}B_n'$ can be decomposed in two different ways as the countable union of almost disjoint boxes, then
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert=\sum_{n=1}^{\infty}\vert B_n'\vert
\end{equation}

**Example 2**  
If a set $E\subset\mathbb{R}^{d}$ is expressible as the countable union of almost disjoint boxes, then
\begin{equation}
m^{\*}(E)=m_{\*,(J)}(E)
\end{equation}

**Proof**  
For $B_n$'s are disjoint boxes, we begin by express $E$ as 
\begin{equation}
E=\bigcup_{n=1}^{\infty}B_n\label{eq:eg2.1}
\end{equation}
Hence, by **Lemma 8**, we have
\begin{equation}
m^{\*}(E)=\sum_{n=1}^{\infty}\vert B_n\vert\label{eq:eg2.2}
\end{equation}
Moreover, \eqref{eq:eg2.1} can be continued to derive as
\begin{equation}
E=\bigcup_{n=1}^{\infty}B_n=\left(\bigcup_{n=1}^{N}B_n\right)\cup\left(\bigcup_{n=N+1}^{\infty}B_n\right)=\left(\bigcup_{n=1}^{N}B_n\right)\cup B,
\end{equation}
where we have defined $B=\bigcup_{n=N+1}^{\infty}B_n$. And thus, we also have that $B_1,\ldots,B_N,B$ are almost disjoint boxes, which claims that $E$ is an elementary set. Therefore, $E$ is also Jordan measurable. Using finite additivity property of Jordan measurability yields
\begin{equation}
m_{\*,(J)}(E)=m(E)=\left(\sum_{n=1}^{N}\vert B_n\vert\right)+\vert B\vert=\sum_{n=1}^{\infty}\vert B_n\vert\label{eq:eg2.3}
\end{equation}
Combining \eqref{eq:eg2.2} and \eqref{eq:eg2.3} together gives us
\begin{equation}
m^{\*}(E)=m_{\*,(J)}(E)
\end{equation}

#### Open sets as countable unions of almost disjoint boxes
{: #open-sets-cntbl-uni-alm-dsjnt-boxes}
**Lemma 10**  
*Let $E\subset\mathbb{R}^d$ be an open set. Then $E$ can be expressed as the countable union of almost disjoint boxes (and, in fact, as the countable union of almost disjoint closed cubes)*.

**Proof**  
We begin by defining a **closed dyadic cube** to be a cube $Q$ of the form
\begin{equation}
Q=\left[\frac{i_1}{2^n},\frac{i_1+1}{2^n}\right]\times\ldots\times\left[\frac{i_d}{2^n},\frac{i_d+1}{2^n}\right],
\end{equation}
for some integers $n,i_1,\ldots,i_d;n\geq 0$.

We have that such closed dyadic cubes of a fixed sidelength $2^{-n}$ are almost disjoint and cover all of $\mathbb{R}^d$. And also, each dyadic cube of sidelength $2^{-n}$ is contained in exactly one "parent" of sidelength $2^{-n+1}$ (which, conversely, has $2^d$ "children" of sidelength $2^{-n}$), giving the dyadic cubes a structure analogous to that of a binary tree. 

As a consequence of these facts, we also obtain the **dyadic nesting property**: given any two closed dyadic cubes (not necessarily same sidelength), then either they are almost disjoint, or one of them is contained in the other.

If $E$ is open, and $x\in E$, then by definition there is an open ball centered at $x$ that is contained in $E$. Also, it is easily seen that there is also a closed dyadic cube containing $x$ that is contained in $E$. Hence, if we let $\mathcal{Q}$ be the collection of all the dyadic cubes $Q$ that are contained in $E$, we see that
\begin{equation}
E=\bigcup_{Q\in\mathcal{Q}}Q
\end{equation}
Let $\mathcal{Q}^\*$ denote cubes in $\mathcal{Q}$ such that they are not contained in any other cube in $\mathcal{Q}$. From the nesting property, we see that every cube in $\mathcal{Q}$ is contained in exactly one maximal cube in $\mathcal{Q}^\*$, and that any two such maximal cubes in $\mathcal{Q}^\*$ are almost disjoint. Thus, we have that
\begin{equation}
E=\bigcup_{Q\in\mathcal{Q}^\*}Q,
\end{equation}
which is union of almost disjoint cubes. As $\mathcal{Q}^\*$ is at most countable, the claim follows.

#### Outer measure of open sets
{: #outer-msr-open-sets}
**Corollary 11**  
The Lebesgue outer measure of any open set is equal to the Jordan inner measure of that set, or of the total volume of any partitioning of that set into almost disjoint boxes.

#### Outer measure of arbitrary sets - Outer regularity
{: #outer-msr-arb-sets}
**Lemma 12**.  
*Let $E\subset\mathbb{R}^d$ be an arbitrary set. Then we have*
\begin{equation}
m^\*(E)=\inf_{E\subset U,U\text{ open}}m^\*(U)
\end{equation}

**Proof**  
From monotonicity property, we have
\begin{equation}
m^\*(E)\leq\inf_{E\subset U,U\text{ open}}m^\*(U)
\end{equation}
Then, it suffices to show that
\begin{equation}
m^\*(E)\geq\inf_{E\subset U,U\text{ open}}m^\*(U),
\end{equation}
which is obvious in the case that $m^\*(E)$ is infinite. Thus, we now assume that $m^\*(E)$ is finite.

Let $\varepsilon>0$. By the definition of Lebesgue outer measure, there exists a countable family $B_1,B_2,\ldots$ of boxes covering $E$ such that
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert\leq m^\*(E)+\varepsilon
\end{equation}
We can enlarge each of these boxes $B_n$ to an open box $B_n'$ such that
\begin{equation}
\vert B_n'\vert\leq\vert B_n\vert+\frac{\varepsilon}{2^n},
\end{equation}
for any $\varepsilon>0$. Then the set $\bigcup_{n=1}^{\infty}B_n'$, being a union of open sets, is itself open, and contains $E$, and
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n'\vert\leq m^\*(E)+\varepsilon+\sum_{n=1}^{\infty}\frac{\varepsilon}{2^n}=m^\*(E)+2\varepsilon
\end{equation}
By countable subadditivity property, it implies that
\begin{equation}
m^\*\left(\bigcup_{n=1}^{\infty}B_n'\right)\leq m^\*(E)+2\varepsilon
\end{equation}
and thus
\begin{equation}
\inf_{E\subset U,U\text{ open}}m^\*(U)\leq m^\*(E)+2\varepsilon
\end{equation}
And since $\varepsilon>0$ was arbitrary, the claim follows.

### Lebesgue measurability
{: #lebesgue-measurability}

#### Existence of Lebesgue measurable sets
{: #exist-lebesgue-msr-sets}
**Lemma 13**.   
<ul id='roman-list' style='font-style: italic;'>
	<li>Every open set is Lebesgue measurable.</li>
	<li>Every closed set is Lebesgue measurable.</li>
	<li>Every set of Lebesgue outer measure zero is measurable. (Such sets are called <b>null sets</b>.)</li>
	<li>The empty set $\emptyset$ is Lebesgue measurable.</li>
	<li>If $E\subset\mathbb{R}^d$ is Lebesgue measurable, then so its complement $\mathbb{R}^d\backslash E$.</li>
	<li>If $E_1,E_2,\ldots\subset\mathbb{R}^d$ are a sequence of Lebesgue measurable sets, then the union $\bigcup_{n=1}^{\infty}E_n$ is Lebesgue measurable.</li>
	<li>If $E_1,E_2,\ldots\subset\mathbb{R}^d$ are a sequence of Lebesgue measurable sets, then the intersection $\bigcap_{n=1}^{\infty}E_n$ is Lebesgue measurable.</li>
</ul>

**Proof**
<ul id='roman-list'>
	<li>This follows from definition.</li>
	<li>
		We have that every closed set is a the countable union of closed and bounded set, so by (vi), if suffices to verify the claim when $E$ is bounded and closed.<br>
		Let $U\supset E$ be an open set, we thus have that $U\backslash E$ is also open due to the compactness of $E$. By <b>lemma 10</b>, we can represent the open set $U\backslash E$ as a countable union of almost disjoint boxes as
		\begin{equation}
		U\backslash E=\bigcup_{n=1}^{\infty}B_n
		\end{equation}
		The problem remains to prove that for any $\varepsilon>0$
		\begin{equation}
		\sum_{n=1}^{\infty}\vert B_n\vert<\varepsilon
		\end{equation}
	</li>
	<li>This follows from definition.</li>
	<li>This follows from definition.</li>
	<li>
		Given $E$ is Lebesgue measurable, for each positive integer $n$, we can find an open set $U_n$ containing $E$ such that
		\begin{equation}
		m^*(U_n\backslash E)\leq\frac{1}{n}
		\end{equation}
		Let $F_n=U_n^c=\mathbb{R}^d\backslash U_n$. Thus, we have $F_n\subset\mathbb{R}^d\backslash E$ and
		\begin{equation}
		m^*\big((\mathbb{R}^d\backslash E)\backslash F_n\big)=m^*\big((\mathbb{R}^d\backslash E)\backslash(\mathbb{R}^d\backslash U_n)\big)=m^*(U_n\backslash E)\leq\frac{1}{n}\label{eq:lemma13.1}
		\end{equation}
		In addition, since $F_n\subset\mathbb{R}^d\backslash E$, the countable union of them, denoted as $F$, is also a subset of $\mathbb{R}^d\backslash E$
		\begin{equation}
		F=\bigcup_{n=1}^{\infty}F_n\subset\mathbb{R}^d\backslash E
		\end{equation}
		Moreover, from \eqref{eq:lemma13.2}, we have
		\begin{equation}
		m^*\left((\mathbb{R}^d\backslash E)\backslash\bigcup_{n=1}^{N}F_n\right)=m^*\left(\bigcap_{n=1}^{N}(\mathbb{R}^d\backslash E)\backslash F_n\right)\leq\frac{1}{N}
		\end{equation}
		Let $N$ approaches $\infty$, we have
		\begin{equation}
		m^*\left((\mathbb{R}^d\backslash E)\backslash F\right)=m^*\left((\mathbb{R}^d\backslash E)\backslash\bigcup_{n=1}^{\infty}F_n\right)\leq 0
		\end{equation}
		By non-negativity property, we then have
		\begin{equation}
		m^*\left((\mathbb{R}^d\backslash E)\backslash F\right)=0,
		\end{equation}
		Hence, $\mathbb{R}^d\backslash E$ is a union of $F$ with a set of Lebesgue outer measure of zero. The set $F$, in the other hand, is a countable union of closed set $F_n$'s (since each $U_n$ is an open set). Therefore, by (ii), (iii) and (vi), we have that $\mathbb{R}^d\backslash E$ is also Lebesgue measurable.
	</li>
	<li>
		For each Lebesgue measurable set $E_n$, for any $\varepsilon>0$ and for $U_n$ is an open set containing $E_n$ we have 
		\begin{equation}
		m^{*}(U_n\backslash E_n)\leq\frac{\varepsilon}{2^n}\label{eq:lemma13.2}
		\end{equation}
		Moreover, since $E_n\subset U_n$, then
		\begin{equation}
		\bigcup_{n=1}^{\infty}E_n\subset\bigcup_{n=1}^{\infty}U_n,
		\end{equation}
		which is also an open set. Therefore, from \eqref{eq:lemma13.2} and by countable subadditivity, we have
		\begin{equation}
		m^*\left(\left(\bigcup_{n=1}^{\infty}U_n\right)\backslash\left(\bigcup_{n=1}^{\infty}E_n\right)\right)\leq\sum_{n=1}^{\infty}m^*(U_n\backslash E_n)\leq\sum_{n=1}^{\infty}\frac{\varepsilon}{2^n}=\varepsilon,
		\end{equation}
		which proves that $\bigcup_{n=1}^{\infty}E_n$ is Lebesgue measurable.
	</li>
	<li>
		Given $E_1,E_2,E_3,\ldots\subset\mathbb{R}^d$ are Lebesgue measurable, by (v), the complement of them,
		\begin{equation}
		E_1^c,E_2^c,E_3^c,\ldots\subset\mathbb{R}^d,
		\end{equation}
		are also Lebesgue measurable. By <b>De Morgan's laws</b>, we have
		\begin{equation}
		\left(\bigcap_{n=1}^{\infty}E_n\right)^c=\bigcup_{n=1}^{\infty}E_n^c,
		\end{equation}
		which is Lebesgue measurable by (vi). Thus, $\left(\bigcap_{n=1}^{\infty}E_n\right)^c$ is also Lebesgue measurable. This means, using (v) once again, we obtain that $\bigcap_{n=1}^{\infty}E_n$ is Lebesgue measurable.
	</li>
</ul>

#### Criteria for measurability
{: #crt-msrb}
Let $E\subset\mathbb{R}^d$, then the following are equivalent:
<ul id='roman-list'>
	<li>$E$ is Lebesgue measurable.</li>
	<li><b>Outer approximation by open</b>. For every $\varepsilon>0$, $E$ can be contained in an open set $U$ with $m^*(U\backslash E)\leq\varepsilon$.</li>
	<li><b>Almost open</b>. For every $\varepsilon>0$, we can find an open set $U$ such that $m^*(U\Delta E)\leq\varepsilon$. ($E$ differs from an open set by a set of outer measure at most $\varepsilon$.)</li>
	<li><b>Inner approximation by closed</b>. For every $\varepsilon>0$, we can find a closed set $F$ contained in $E$ with $m^*(E\backslash F)\leq\varepsilon$.</li>
	<li><b>Almost closed</b>. For every $\varepsilon>0$, we can find a closed set $F$ such that $m^*(F\Delta E)\leq\varepsilon$. ($E$ differs from a closed set by a set of outer measure at most $\varepsilon$.)</li>
	<li><b>Almost measurable</b>. For every $\varepsilon>0$, we can find a Lebesgue measurable set $E_\varepsilon$ such that $m^*(E_\varepsilon\Delta E)\leq\varepsilon$. ($E$ differs from a measurable set by a set of outer measure at most $\varepsilon$.)</li>
</ul>

**Proof**
<ul id='number-list'>
	<li>
		(i) $\Rightarrow$ (ii)<br>
		This follows from definition
	</li>
	<li>
		(i) $\Rightarrow$ (iii)<br>
		Given $E$ is Lebesgue measurable, for any $\varepsilon>0$, we can find an open set $U$ containing $E$ such that
		\begin{equation}
		m^*(U\backslash E)\leq\varepsilon
		\end{equation}
		And since $E\subset U$, we have that
		\begin{equation}
		m^(E\backslash U)=m^*(\emptyset)=0,
		\end{equation}
		which implies that for any $\varepsilon>0$
		\begin{equation}
		m^*(U\Delta E)=m^*(U\backslash E)+m^*(E\backslash U)\leq\varepsilon
		\end{equation}
	</li>
	<li>
		(i) $\Rightarrow$ (iv)<br>
		By the claim (v) in <b>lemma 13</b>, given Lebesgue measurable set $E\subset\mathbb{R}^d$, we have that its complement $\mathbb{R}^d\backslash E$ is also Lebesgue measurable. Therefore, there exists an open set $U$ containing $\mathbb{R}^d\backslash E$ such that for any $\varepsilon>0$ we have
		\begin{equation}
		m^*\left(U\backslash(\mathbb{R}^d\backslash E)\right)\leq\varepsilon\label{eq:cm.1}
		\end{equation}
		Let $F$ denote the complement of $U$, $F=\mathbb{R}\backslash U$, thus $F$ is a closed set contained in $E$. Moreover, from \eqref{eq:cm.1} we also have for any $\varepsilon>0$
		\begin{equation}
		m^*(E\backslash F)=m^*\left(E\backslash(\mathbb{R}^d\backslash U)\right)=m^*\left(U\backslash(\mathbb{R}^d\backslash E)\right)\leq\varepsilon
		\end{equation}
	</li>
	<li>
		(i) $\Rightarrow$ (v)<br>
		Given Lebesgue measurable set $E\subset\mathbb{R}^d$, using the claim (v) in <b>lemma 13</b> gives us that its complement $\mathbb{R}^d\backslash E$ is also Lebesgue measurable.<br>
		From claim (iii), for any $\varepsilon>0$, we can find an open set $U$ such that
		\begin{equation}
		m^*\left(U\Delta(\mathbb{R}^d\backslash E)\right)\leq\varepsilon\label{eq:cm.2}
		\end{equation}
		Let $F$ denote the complement of $U$, $F=\mathbb{R}^d\backslash$. We then have that $F$ is a closed set. In addition, $U\Delta(\mathbb{R}^d\backslash E)$ can be rewritten by
		\begin{align}
		U\Delta(\mathbb{R}^d\backslash E)&=\left(U\backslash(\mathbb{R}^d\backslash E)\right)\cup\left((\mathbb{R}^d\backslash E)\backslash U\right) \\ &=\left(E\backslash(\mathbb{R}^d\backslash U)\right)\cup\left((\mathbb{R}^d\backslash U)\backslash E\right) \\ &=(\mathbb{R}^d\backslash U)\backslash E \\ &=F\Delta E,
		\end{align}
		which lets \eqref{eq:cm.2} can be written as, for any $\varepsilon>0$
		\begin{equation}
		m^*(F\Delta E)\leq\varepsilon
		\end{equation}
	</li>
	<li>
		(i) $\Rightarrow$ (vi)<br>
		Given $E$ is Lebesgue measurable, by claim (v), for any $\varepsilon>0$ we can find a closed set $E_\varepsilon$ such that
		\begin{equation}
		m^*(E_\varepsilon\Delta E)\leq\varepsilon
		\end{equation}
		While by property (ii) of <b>lemma 13</b>, we have that $E_\varepsilon$ is Lebesgue measurable, which proves our claim.
	</li>
	<li>
		(vi) $\Rightarrow$ (i)<br>
		Given (vi), for any $\varepsilon>0$, we can find a Lebesgue measurable set $E_\varepsilon^{(n)}$ such that
		\begin{equation}
		m^*\left(E_\varepsilon^{(n)}\Delta E\right)\leq\frac{\varepsilon}{2^n}
		\end{equation}
		Therefore, by countable subadditivity property of Lebesgue outer measurability
		\begin{equation}
		m^*\left(\bigcup_{n=1}^{\infty}E_\varepsilon^{(n)}\Delta E\right)\leq\sum_{n=1}^{\infty}m^*\left(E_\varepsilon^{(n)}\Delta E\right)\leq\sum_{n=1}^{\infty}\frac{\varepsilon}{2^n}=\varepsilon
		\end{equation}
	</li>
</ul>

**Remark 14**    
Every Jordan measurable set is Lebesgue measurable.

**Proof**  
This follows directly from **corollary 6**.

**Remark 15**    
The [**Cantor set**]({% post_url 2022-06-16-measure-theory-p1 %}#cantor-set) is compact, uncountable, and a null set.

**Proof**  
- Since $\mathcal{C}\subseteq[0,1]$ is closed and bounded, by the [Heine-Borel theorem]({% post_url 2022-06-16-measure-theory-p1 %}#heine-borel-theorem), $\mathcal{C}$ is then compact.
- 


#### The measure axioms
{: #msr-axiom}
**Lemma 16**  
<ul id='roman-list' style='font-style: italic;'>
	<li><b>Empty set</b>. $m(\emptyset)=0$.</li>
	<li><b>Countable additivity</b>. If $E_1,E_2,\ldots\subset\mathbb{R}^d$ is a countable sequence of disjoint Lebesgue measurable sets, then</li>
	\begin{equation}
	m\left(\bigcup_{n=1}^{\infty}E_n\right)=\sum_{n=1}^{\infty}m(E_n)
	\end{equation}
</ul>

**Proof**
<ul id='roman-list'>
	<li>
		<b>Empty set</b><br>
		We have that empty set $\emptyset$ is Lebesgue measurable since for every $\varepsilon>0$, there exists an open set $U\subset\mathbb{R}^d$ containing $\emptyset$ such that $m^*(U\backslash\emptyset)\leq\varepsilon$. Thus,
		\begin{equation}
		m(\emptyset)=m^*(\emptyset)=0
		\end{equation}
	</li>
	<li>
		<b>Countable additivity</b><br>
		We begin by considering the case that $E_n$ are all compact sets.
		<br>
		By repeated use of <b>Lemma 12</b> and <b>Example ?</b>, we have
		\begin{equation}
		m\left(\bigcup_{n=1}^{N}E_n\right)=\sum_{n=1}^{N}m(E_n)
		\end{equation}
		Thus, using monotonicity property, we have
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)\geq\sum_{n=1}^{N}m(E_n)
		\end{equation}
		Let $N\to\infty$, we obtain
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)\geq\sum_{n=1}^{\infty}m(E_n)
		\end{equation}
		On the other hand, by countable subadditivity, we also have
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)\leq\sum_{n=1}^{N}m(E_n)
		\end{equation}
		Therefore, we can conclude that
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)=\sum_{n=1}^{N}m(E_n)
		\end{equation}
		Next, we consider the case that $E_n$ are bounded but not necessarily compact.
		<br>
		Let $\varepsilon>0$. By criteria for measurability, we know that each $E_n$ is the union of a compact set $K_n$ and a set of outer measure at most $\varepsilon/2^n$. Thus
		\begin{equation}
		m(E_n)\leq m(K_n)+\frac{\varepsilon}{2^n}
		\end{equation}
		And hence
		\begin{equation}
		\sum_{n=1}^{\infty}m(E_n)\leq\left(\sum_{n=1}^{\infty}m(K_n)\right)+\varepsilon
		\end{equation}
		From the first case, we know that
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}K_n\right)=\sum_{n=1}^{\infty}m(K_n)
		\end{equation}
		while from monotonicity property of Lebesgue measure
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}K_n\right)\leq m\left(\bigcup_{n=1}^{\infty}E_n\right)
		\end{equation}
		Putting these results together we obtain
		\begin{equation}
		\sum_{n=1}^{\infty}m(E_n)\leq m\left(\bigcup_{n=1}^{\infty}E_n\right)+\varepsilon,
		\end{equation}
		for every $\varepsilon>0$. And since $\varepsilon$ was arbitrary, we have
		\begin{equation}
		\sum_{n=1}^{\infty}m(E_n)\leq m\left(\bigcup_{n=1}^{\infty}E_n\right)
		\end{equation}
		while from countable subadditivity property we have
		\begin{equation}
		\sum_{n=1}^{\infty}m(E_n)\geq m\left(\bigcup_{n=1}^{\infty}E_n\right)
		\end{equation}
		Therefore, the claim follows.
		<br>
		Finally, we consider the case that $E_n$ are not bounded or closed with the idea of decomposing each $E_n$ as a countable disjoint union of bounded Lebesgue measurable sets.
		<br>
	</li>
</ul>

**Remark 17**  
The countable additivity also implies the **finite additivity** property of Lebesgue  measure
\begin{equation}
m\left(\bigcup_{n=1}^{N}E_n\right)=\sum_{n=1}^{N}m(E_n),
\end{equation}
where $E_1,\ldots,E_N$ are Lebesgue measurable.

#### Monotone convergence theorem for measurable sets
{: #mnt-cvg-theorem-msr-sets}  
<ul id='roman-list'>
	<li>
		<b>Upward monotone convergence</b>. Let $E_1\subset E_2\subset\ldots\subset\mathbb{R}^d$ be a countable non-decreasing sequence of Lebesgue measurable sets. Then
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)=\lim_{n\to\infty}m(E_n)
		\end{equation}
	</li>
	<li>
		<b>Downward monotone convergence</b>. Let $\mathbb{R}^d\supset E_1\supset E_2\supset\ldots$ be a countable non-increasing sequence of Lebesgue measurable sets. If at least one of the $m(E_n)$ is finite, then
		\begin{equation}
		m\left(\bigcap_{n=1}^{\infty}E_n\right)=\lim_{n\to\infty}m(E_n)
		\end{equation}
	</li>
	<li>
		The hypothesis that at least one of the $m(E_n)$ is finite in the downward monotone convergence theorem cannot be dropped.
	</li>
</ul>

**Proof**
<ul id='roman-list'>
	<li>
		<b>Upward monotone convergence</b><br>
		Since $E_1\subset E_2\subset\ldots\subset\mathbb{R}^d$ is a countable non-decreasing sequence of Lebesgue measurable sets, by countable additivity, we have
		\begin{align}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)&=m\left(\bigcup_{n=1}^{\infty}E_n\backslash\bigcup_{n'=1}^{n-1}E_{n'}\right) \\ &=m\left(\bigcup_{n=1}^{\infty}E_n\backslash E_{n-1}\right) \\ &=\left(\sum_{n=2}^{\infty}m(E_n)-m(E_{n-1})\right)+m(E_1) \\ &=\lim_{n\to\infty}m(E_n)
		\end{align}
	</li>
	<li>
		<b>Downward monotone convergence</b><br>
		Since $\mathbb{R}^d\supset E_1\supset E_2\supset\ldots$ is a countable non-increasing sequence of Lebesgue measurable sets, the sequence of their complement $E_1^c\subset E_2^c\subset\ldots\subset\mathbb{R}^d$ is therefore a countable non-decreasing sequence of Lebesgue measurable sets. Using the claim (i) and by De Morgan's laws, we have
		\begin{align}
		m\left(\bigcap_{n=1}^{\infty}E_n\right)&=m\left(\mathbb{R}^d\backslash\bigcup_{n=1}^{\infty}E_n^c\right) \\ &=m(\mathbb{R}^d)-m\left(\bigcup_{n=1}^{\infty}E_n^c\right) \\ &=m(\mathbb{R}^d)-\lim_{n\to\infty}m(E_n^c) \\ &=m(\mathbb{R}^d)-m(\mathbb{R}^d)+\lim_{n\to\infty}m(E_n) \\ &=\lim_{n\to\infty}m(E_n)
		\end{align}
	</li>
	<li>
		Consider sequence $\mathbb{R}^d\supset E_1\supset E_2\supset\ldots$ of non-increasing Lebesgue measurable sets where each $E_n$ is given by
		\begin{equation}
		E_n\doteq[n,+\infty)
		\end{equation}
		Therefore, by De Morgan's laws, the Lebesgue measure of their countable intersection is
		\begin{align}
		m\left(\bigcap_{n=1}^{\infty}E_n\right)&=m\left(\mathbb{R}^d\backslash\bigcup_{n=1}^{\infty}E_n^c\right) \\ &=m\left(\mathbb{R}^d\backslash\bigcup_{n=1}^{\infty}(-\infty,n)\right) \\ &=m(\mathbb{R}^d\backslash\mathbb{R}^d) \\ &=m(\emptyset)=0,
		\end{align}
		while for every $n$, we have
		\begin{equation}
		m(E_n)=m\left([n,+\infty)\right)=\infty
		\end{equation}
	</li>
</ul>

#### Dominated convergence theorem for measurable sets
{: #dmnt-cvg-theorem-msr-sets} 
We say that a sequence $E_n$ of sets in $\mathbb{R}^d$ **converges pointwise** to another set $E$ in $\mathbb{R}^d$ if the indicator function $1_{E_n}$ converges pointwise to $1_E$.
<ul id='roman-list'>
	<li>
		If the $E_n$ are all Lebesgue measurable, and converge pointwise to $E$, then $E$ is Lebesgue measurable also.
	</li>
	<li>
		<b>Dominated convergence theorem</b>. Suppose that the $E_n$ are all contained in another Lebesgue measurable set $F$ of finite measure. Then $m(E_n)$ converges to $m(E)$.
	</li>
	<li>
		The dominated convergence theorem fails if the $E_n$'s are not contained in a set of finite measure, even if we assume that the $m(E_n)$ are all uniformly bounded.
	</li>
</ul>

**Proof**
<ul id='roman-list'>
	<li>
		We have
	</li>
</ul>

**Remark 18**  
Let $E\subset\mathbb{R}^d$, then $E$ is contained in a Lebesgue measurable set of measure exactly equal to $m^\*(E)$.

**Proof**  


#### Inner regularity
{: #inn-rglr}
Let $E\subset\mathbb{R}^d$ be Lebesgue measurable. Then
\begin{equation}
m(E)=\sup_{K\subset E,K\text{ compact}}m(K)
\end{equation}

**Proof**  
By monotonic we have that
\begin{equation}
m(E)\geq\sup_{K\subset E,K\text{ compact}}m(K),
\end{equation}
thus it suffices to show that
\begin{equation}
m(E)\leq\sup_{K\subset E,K\text{ compact}}m(K)
\end{equation}
Consider the case that $E$ is bounded. By the **criteria for Lebesgue measurability**, we have that for any $\varepsilon>0$, there exist a bounded and closed, and thus compact by the Heine-Borel theorem, set $K'$ contained in $E$ such that
\begin{equation}
m(E\backslash K')\leq\varepsilon
\end{equation}
Moreover, by claim (ii) of **lemma 13**, we have that $K'$ is Lebesgue measurable. Using finite additivity property of Lebesgue measure gives us
\begin{equation}
\varepsilon\geq m(E\backslash K')=m(E)-m(K'),
\end{equation}
which means
\begin{equation}
m(E)\leq m(K')\leq\sup_{K\subset E,K\text{ compact}}m(K)
\end{equation}
Now consider {the case that $E$ is an unbounded set. Let $(K_r)\_{r=1,2,\ldots}$ be the sequence sets in which each $K_r$ is defined as
\begin{equation}
K_r\doteq E\cap B_r(\mathbf{0}),\label{eq:ir.1}
\end{equation}
where $B_r(\mathbf{0})$ is a closed ball centered at $\mathbf{0}\in\mathbb{R}^d$ with radius $r$
\begin{equation}
B_r(\mathbf{0})=\\{\mathbf{x}:\vert\mathbf{x}\vert\leq r\\}
\end{equation}
which means $K_1\subset K_2\subset\ldots\subset E$ is an increasing sequence of compact set (since \eqref{eq:ir.1} also implies that $K_r\subset B_r(\mathbf{0})$, and hence bounded and closed, then using the Heine-Borel theorem to obtain the compactness of $K_r$). By the **monotone convergence theorem**, we have
\begin{equation}
m\left(\bigcup_{r=1}^{\infty}K_r\right)=\lim_{r\to\infty}m(K_r)
\end{equation}
On the other hand, the countable union of $K_r$ can be written as
\begin{equation}
\bigcup_{r=1}^{\infty}K_r=\bigcup_{r=1}^{\infty}E\cap B_r(\mathbf{0})=E\cap\bigcup_{r=1}^{\infty}B_r(\mathbf{0})=E\cap\mathbb{R}^d=E,
\end{equation}
which therefore gives us
\begin{equation}
m(E)=\lim_{r\to\infty}m(K_r)\label{eq:ir.2}
\end{equation}
Moreover, by monotonicity property $m(E)\geq m(K_r),\forall r$. Hence, \eqref{eq:ir.2} implies that for any $\varepsilon>0$, there exists $r'$ such that for all $r\geq r'$
\begin{equation}
\varepsilon>\vert m(E)-m(K_{r'})\vert=m(E)-m(K_{r'})
\end{equation}
This means that
\begin{equation}
m(A)\leq\sup_{K\subset E,K\text{ compact}}m(K)
\end{equation}
Our claim then follows.

#### Criteria for finite measure
{: #crt-fnt-msr}
Let $E\subset\mathbb{R}^d$, then the following are equivalent:
<ul id='roman-list'>
	<li>
		$E$ is Lebesgue measurable with finite measure.
	</li>
	<li>
		<b>Outer approximation by open</b>. For every $\varepsilon>0$, we can contain $E$ in an open set $U$ of finite measure with $m^*(U\backslash E)\leq\varepsilon$.
	</li>
	<li>
		<b>Almost open bounded</b>. For every $\varepsilon>0$, there exists a bounded open set $U$ such that $m^*(E\Delta U)\leq\varepsilon$. (In other words, $E$ differs from a bounded set by a set of arbitrarily small Lebesgue outer measure.)
	</li>
	<li>
		<b>Inner approximation by compact</b>. For every $\varepsilon>0$, we can find a compact set $F$ contained in $E$ with $m^*(E\backslash F)\leq\varepsilon$.
	</li>
	<li>
		<b>Almost compact</b>. $E$ differs from a compact set by a set of arbitrarily small Lebesgue outer measure.
	</li>
	<li>
		<b>Almost bounded measurable</b>. $E$ differs from a bounded Lebesgue measurable set by a set of arbitrarily small Lebesgue outer measure.
	</li>
	<li>
		<b>Almost finite measure</b>. $E$ differs from a Lebesgue measurable set with finite measure by a set of arbitrarily small Lebesgue outer measure.
	</li>
	<li>
		<b>Almost elementary</b>. $E$ differs from an elementary set by a set of arbitrarily small Lebesgue outer measure.
	</li>
	<li>
		<b>Almost dyadically elementary</b>. For every $\varepsilon>0$, there exists an integer $n$ and a finite union $F$ of closed dyadic cubes of sidelength $2^{-n}$ such that $m^*(E\Delta F)\leq\varepsilon$.
	</li>
</ul>

**Proof**
<ul id='roman-list'>
	<li>
		(i) $\Rightarrow$ (ii)<br>
		Given $E$ is Lebesgue measurable with finite measure, by definition, for any $\varepsilon>0$, there exists an open set $U$ o such that
		\begin{equation}
		m^*(U\backslash E)\leq\varepsilon
		\end{equation}
		Then, by finite subadditivity property of Lebesgue outer measure
		\begin{equation}
		m^*(U)\leq m^*(E)+\varepsilon,
		\end{equation}
		which implies that $m^*(U)$ finite due to finiteness of $m^*(E)$ and $\varepsilon$, and hence $U$ has finite measure since $m(U)\leq m^*(U)$.
	</li>
	<li>
		(i) $\Rightarrow$ (iii)<br>
	</li>
</ul>


#### Carathéodory criterion, one direction
{: #caratheodory-crt}
Let $E\subset\mathbb{R}^d$, the following are then equivalent:
<ul id='roman-list'>
	<li>
		$E$ is Lebesgue measurable.
	</li>
	<li>
		For every elementary set $A$
		\begin{equation}
		m(A)=m^*(A\cap E)+m^*(A\backslash E)
		\end{equation}
	</li>
	<li>
		For every box $B$, we have
		\begin{equation}
		\vert B\vert=m^*(B\cap E)+m^*(B\backslash E)
		\end{equation}
	</li>
</ul>

**Proof**
<ul id='number-list'>
	<li>
		(i) $\Rightarrow$ (ii)<br>
		We begin with an observation that, by finite additivity property of Lebesgue measure
		\begin{equation}
		m(A)=m(A\cap E)+m(A\backslash E)\leq m^*(A\cap E)+m^*(A\backslash E)\label{eq:cc.1}
		\end{equation}
		Given $A$ is elementary, by <span markdown=1>[<b>lemma 10</b>]({% post_url 2022-06-16-measure-theory-p1 %}#measure-elementary-set)</span>, we can express $A$ as a finite union of disjoint boxes
		\begin{equation}
		A=\bigcup_{n=1}^{N}B_n
		\end{equation}
		Continuing using finite subadditivity of Lebesgue outer measure and finite additivity of Lebesgue measure, \eqref{eq:cc.1} then can be continued to derive as
		\begin{align}
		m(A)&\leq m^*(A\cap E)+m^*(A\backslash E) \\ &=m^*\left(\left(\bigcup_{n=1}^{N}B_n\right)\cap E\right)+m^*\left(\left(\bigcup_{n=1}^{N}B_n\right)\backslash E\right) \\ &=m^*\left(\bigcup_{n=1}^{N}B_n\cap E\right)+m^*\left(\bigcup_{n=1}^{N}B_n\backslash E\right) \\ &\leq\sum_{n=1}^{N}m^*(B_n\cap E)+m^*(B_n\backslash E) \\ &=\sum_{n=1}^{N}m^*(B_n)=\sum_{n=1}^{N}m(B_n)=m\left(\bigcup_{n=1}^{N}B_n\right)=m(A),
		\end{align}
		which implies that
		\begin{equation}
		m(A)=m^*(A\cap E)+m^*(A\backslash E)
		\end{equation}
	</li>
	<li>
		(i) $\Rightarrow$ (iii)<br>
		Since every box $B$ is Lebesgue measurable, then given $E$ is also Lebesgue measurable, by <b>lemma 13</b>, their difference and intersection are also Lebesgue measurable, which means by additivity property of Lebesgue measure we have
		\begin{equation}
		\vert B\vert=m(B)=m(B\cap E)+m(B\backslash E)=m^*(B\cap E)+m^*(B\backslash E)
		\end{equation}
	</li>
	<li>
		(ii) $\Rightarrow$ (i)<br>
	</li>
</ul>

#### Inner measure
{: #inn-msr}
Let $E\subset\mathbb{R}^d$ be a bounded set. The **Lebesgue inner measure** $m_\*(E)$ of $E$ is defined by
\begin{equation}
m_\*(E)\doteq m(A)-m^\*(A\backslash E),
\end{equation}
for any elementary set $A$ containing $E$. Then
<ul id='roman-list'>
	<li>
		If $A,A'$ are two elementary sets containing $E$, then
		\begin{equation}
		m(A)-m^*(A\backslash E)=m(A')-m^*(A'\backslash E)
		\end{equation}
	</li>
	<li>
		We have that $m_*(E)\leq m^*(E)$, and that equality holds iff $E$ is Lebesgue measurable.
	</li>
</ul>

**Proof**  


**Example 3**  
Let $E\subset \mathbb{R}^d$, and define a $G_\delta$ *set* to be a countable intersection $\bigcap_{n=1}^{\infty}U_n$ of open sets, and define an $F_\delta$ *set* to be a countable union $\bigcup_{n=1}^{\infty}F_n$ of closed sets. The following are then equivalent:
<ul id='roman-list'>
	<li>
		$E$ is Lebesgue measurable.
	</li>
	<li>
		$E$ is a $G_\delta$ set with a null set removed.
	</li>
	<li>
		$E$ is the union of an $F_\delta$ set and a null set.
	</li>
</ul>

**Proof**  




#### Translation invariance
{: #trans-inv}
Let $E\subset\mathbb{R}^d$ be Lebesgue measurable, then $E+x$ is also Lebesgue measurable for any $x\in\mathbb{R}^d$, and $m(E+x)=m(E)$.

**Proof**  


#### Change of variables
{: #change-vars}
Let $E\subset\mathbb{R}^d$ be Lebesgue measurable, and $T:\mathbb{R}^d\to\mathbb{R}^d$ be a linear transformation, then $T(E)$ is Lebesgue measurable, and $m(T(E))=\vert\text{det}(T)\vert m(E)$.

**Note**  
If $T:\mathbb{R}^d\to\mathbb{R}^{d'}$ is a linear map to a space $\mathbb{R}^{d'}$ of strictly smaller dimension than $\mathbb{R}^d$, then $T(E)$ need not be Lebesgue measurable.

**Proof**  


**Remark 19**  
Let $d,d'\geq 1$ be natural numbers
<ul id='roman-list'>
	<li>
		If $E\subset\mathbb{R}^d$ and $F\subset\mathbb{R}^{d'}$, then
		\begin{equation}
		(m^{d+d'})^*(E\times F)\leq(m^d)^*(E)(m^{d'})^*(F)
		\end{equation}
	</li>
	<li>
		Let $E\subset\mathbb{R}^d,F\subset\mathbb{R}^{d'}$ be Lebesgue measurable sets. Then $E\times F\subset\mathbb{R}^{d+d'}$ is Lebesgue measurable, with \begin{equation}
		m^{d+d'}(E\times F)=m^d(E).m^{d'}(F)
		\end{equation}
	</li>
</ul>

**Proof**  


#### Uniqueness of Lebesgue measure
{: #uniq-lebesgue-msr}
Lebesgue measure $E\mapsto m(E)$ is the only map from Lebesgue measurable sets to $[0,+\infty]$ that obeys the following axioms:
<ul id='roman-list'>
	<li>
		<b>Empty set</b>. $m(\emptyset)=0$.
	</li>
	<li>
		<b>Countable additivity</b>. If $E_1,E_2,\ldots\subset\mathbb{R}^d$ is a countable sequence of disjoint Lebesgue measurable sets, then 
		\begin{equation}
		m\left(\bigcup_{n=1}^{\infty}E_n\right)=\sum_{n=1}^{\infty}m(E_n)
		\end{equation}
	</li>
	<li>
		<b>Translation invariance</b>. If $E$ is Lebesgue measurable and $x\in\mathbb{R}^d$, then $m(E+x)=m(E)$.
	</li>
	<li>
		<b>Normalisation</b>. $m([0,1]^d)=1$.
	</li>
</ul>

**Proof**  


### Non-measurable sets
{: #non-measurable-sets}
**Remark 20**  
There exists a subset $E\subset[0,1]$ which is not Lebesgue measurable.

**Remark 21** (Outer measure is not finitely additive)  
There exists disjoint bounded subsets $E,F\subset\mathbb{R}$ such that
\begin{equation}
m^\*(E\cap F)\neq m^\*(E)+m^\*(F)
\end{equation}

**Remark 22**  
Let $\pi:\mathbb{R}^2\to\mathbb{R}$ be the coordinate projection $\pi(x,y)\doteq x$. Then there exists a measurable $E\subset\mathbb{R}^2$ such that $\pi(E)$ is not measurable.

## References
{: #references}
[1] <span id='taos-book'>Terence Tao. [An introduction to measure theory](https://terrytao.wordpress.com/books/an-introduction-to-measure-theory/). Graduate Studies in Mathematics, vol. 126.</span>

[2] <span id='steins-book'>Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](#http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). </span>

## Footnotes
{: #footnotes}

[^1]: The **diameter** of a set $B$ is defined as
	\begin{equation\*}
	\text{dia}(B)\doteq\sup\\{\vert x-y\vert:x,y\in B\\}
	\end{equation\*}
