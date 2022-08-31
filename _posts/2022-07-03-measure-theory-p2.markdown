---
layout: post
title:  "Measure theory - II"
date:   2022-07-03 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory random-stuffs
description: Note on measure theory part 2
comments: true
---
> A note on measure theory (continued from [part I]({% post_url 2022-06-16-measure-theory %})): materials were mostly taken from [Tao's book]({% post_url 2022-07-03-measure-theory-p2 %}#taos-book), except for some notations needed from [Stein's book]({% post_url 2022-07-03-measure-theory-p2 %}#steins-book).
<!-- excerpt-end -->

- [Lebesgue measure](#lebesgue-measure)
	- [Properties of Lebesgue outer measure](#lebesgue-outer-measure-properties)
	- [Lebesgue measurability](#lebesgue-measurability)
		- [Criteria for measurability](#criteria-measurability)
	- [Non-measurable sets](#non-measurable-sets)
- [Lebesgue integral](#lebesgue-int)
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

**Remark**. we always have $m^\*(E)\leq m^{\*,(J)}(E)$.

A set $E\subset\mathbb{R}^d$ is said to be **Lebesgue measurable** if, for every $\varepsilon>0$, there exists an open set $U\subset\mathbb{R}^d$ containing $E$ such that $m^{\*}(U\backslash E)\leq\varepsilon$. If $E$ is Lebesgue measurable, we refer to
\begin{equation}
m(E)\doteq m^{\*}(E)
\end{equation}
as the **Lebesgue measure** of $E$.

### Properties of Lebesgue outer measure
{: #lebesgue-outer-measure-properties}
**Remark**. (**The outer measure axioms**)
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
		Thus, by definition of infimum and by axiom of countable choice (<b>Axiom 8</b>), for each $E_i$ in the sequence $(E_n)_{n\in\mathbb{N}}$, there exists a family of boxes $B_{i,1},B_{i,2},\ldots$ in the doubly sequence $(B_{i,j})_{(i,j)\in\mathbb{N}^2}$ covering $E_i$ such that
		\begin{equation}
		\sum_{j=1}^{\infty}\vert B_{i,j}\vert\lt m^*(E_i)+\frac{\varepsilon}{i},
		\end{equation}
		for any $\varepsilon>0$, and for $i=1,2,\ldots$. Plus, we also have
		\begin{equation}
		\bigcup_{n=1}^{\infty}E_n\subset\bigcup_{i=1}^{\infty}\bigcup_{j=1}^{\infty}B_{i,j}
		\end{equation}
		Moreover, by the Tonelli's theorem for series (<b>Theorem 6</b>), we have
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

**Corollary 11**  
Combining empty set with countable subadditivity axiom gives us the finite subadditivity property
\begin{equation}
m^{\*}\left(E_1\cup\ldots\cup E_k\right)\leq m^{\*}(E_1)+\ldots+m^{\*}(E_k),\hspace{1cm}\forall k\geq 0
\end{equation}

**Lemma 12**. (**Finite additivity for separated sets**)  
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

**Example**  
Let $E,F\subset\mathbb{R}^d$ be disjoint closed sets, with at least one of $E,F$ being compact. Then $\text{dist}(E,F)>0$.

**Lemma 13**. (**Outer measure of elementary sets**)  
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
\sum_{n=1}^{\infty}\vert B_n'\vert\leq\sum_{n=1}^{\infty}\left(\vert B_n\vert+\frac{\varepsilon}{2^n}\right)=\left(\sum_{n=1}^{\infty}\vert B_n\vert\right)+\varepsilon\leq m^\*(E)+2\varepsilon\tag{11}\label{11}
\end{equation}
As the $B_n'$ are open, apply the **Heine-Borel theorem** (**Theorem 5**), we obtain
\begin{equation}
E\subset\bigcup_{n=1}^{N}B_n',
\end{equation}
for some finite $N$. Thus, using the finite subadditivity property of elementary measure, combined with the result \eqref{11}, we obtain
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

**Corollary 14**  
From the lemma above and the monotonicity property, 
for every $E\in\mathbb{R}^d$, we have
\begin{equation}
m_{\*,(J)}(E)\leq m^{\*}(E)\leq m^{\*,(J)}(E)
\end{equation}

**Remark**  
- Not every bounded open set or compact set (bounded closed) is Jordan measurable.
- Two boxes are **almost disjoint** if their interiors are disjoint. E.g., $[0,1]$ and $[1,2]$ are almost disjoint. If a box has the same elementary as its interior, we see that the finite additivity property
\begin{equation}
m(B_1\cup\ldots\cup B_n)=\vert B_1\vert+\ldots+\vert B_n\vert\tag{12}\label{12}
\end{equation}
also holds for almost disjoint boxes $B_1,\ldots,B_n$.

**Lemma 15**. (**Outer measure of countable unions of almost disjoint boxes**)  
*Let $E=\bigcup_{n=1}^{\infty}B_n$ be a countable union of almost disjoint boxes $B_1,B_2,\ldots$. Then*
\begin{equation}
m^\*(E)=\sum_{n=1}^{\infty}\vert B_n\vert
\end{equation}
Thus, for example, $\mathbb{R}^d$ has an infinite outer measure.

**Proof**  
From countable subadditivity property of Lebesgue measure and **Lemma 13**, we have
\begin{equation}
m^\*(E)\leq\sum_{n=1}^{\infty}m^\*(B_n)=\sum_{n=1}^{\infty}\vert B_n\vert,
\end{equation}
so it suffices to show that
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert\leq m^\*(E)
\end{equation}
Since for each integer $N$, $E$ contains the elementary set $B_1\cup\ldots\cup B_N$, then by monotonicity property and **Lemma 13**
\begin{align}
m^\*(E)&\geq m^\*(B_1\cup\ldots\cup B_N)=m(B_1\cup\ldots\cup B_N)
\end{align}
And thus by \eqref{12}, we have
\begin{equation}
\sum_{n=1}^{N}\vert B_n\vert\leq m^\*(E)
\end{equation}
Letting $N\to\infty$ we obtain the claim.

**Corollary 16**  
If $E=\bigcup_{n=1}^{\infty}B_n=\bigcup_{n=1}^{\infty}B_n'$ can be decomposed in two different ways as the countable union of almost disjoint boxes, then
\begin{equation}
\sum_{n=1}^{\infty}\vert B_n\vert=\sum_{n=1}^{\infty}\vert B_n'\vert
\end{equation}

**Lemma 17**  
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

**Corollary 18**  
The Lebesgue outer measure of any open set is equal to the Jordan inner measure of that set, or of the total volume of any partitioning of that set into almost disjoint boxes.

**Lemma 19**. (**Outer regularity**)  
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

**Lemma 20**. (**Existence of Lebesgue measurable sets**)  
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
</ul>

**Remark**. (**Criteria for measurability**)
{: #criteria-measurability}
Let $E\subset\mathbb{R}^d$. The following are equivalent
<ul id='roman-list'>
	<li>$E$ is Lebesgue measurable.</li>
	<li><b>Outer approximation by open</b>. For every $\varepsilon>0$, $E$ can be contained in an open set $U$ with $m^*(U\backslash E)\leq\varepsilon$.</li>
	<li><b>Almost open</b>. For every $\varepsilon>0$, we can find an open set $U$ such that $m^*(U\Delta E)\leq\varepsilon$. ($E$ differs from an open set by a set of outer measure at most $\varepsilon$.)</li>
	<li><b>Inner approximation by closed</b>. For every $\varepsilon>0$, we can find a closed set $F$ contained in $E$ with $m^*(E\backslash F)\leq\varepsilon$.</li>
	<li><b>Almost closed</b>. For every $\varepsilon>0$, we can find a closed set $F$ such that $m^*(F\Delta E)\leq\varepsilon$. ($E$ differs from a closed set by a set of outer measure at most $\varepsilon$.)</li>
	<li><b>Almost measurable</b>. For every $\varepsilon>0$, we can find a Lebesgue measurable set $E_\varepsilon$ such that $m^*(E_\varepsilon\Delta E)\leq\varepsilon$. ($E$ differs from a measurable set by a set of outer measure at most $\varepsilon$.)</li>
</ul>

**Proof**  

**Lemma 21**. (**The measure axioms**)
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
		We have that empty set $\emptyset$ is Lebesgue measurable since for every $\varepsilon>0$, there exists an open set $U\subset\mathbb{R}^d$ containing $\emptyset$ such that $m^*(U\backslash\emptyset)\leq\varepsilon$. Thus,
		\begin{equation}
		m(\emptyset)=m^*(\emptyset)=0
		\end{equation}
	</li>
	<li>
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

**Example** (**Monotone convergence theorem for measurable sets**)  
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
</ul>

**Example**  
We say that a sequence $E_n$ of sets in $\mathbb{R}^d$ **converges pointwise** to another set $E$ in $\mathbb{R}^d$ if the indicator function $1_{E_n}$ converges pointwise to $1_E$.
<ul id='roman-list'>
	<li>If the $E_n$ are all Lebesgue measurable, and converge pointwise to $E$, then $E_n$ is Lebesgue measurable also.</li>
	<li><b>Dominated convergence theorem</b>. Suppose that the $E_n$ are all contained in another Lebesgue measurable set $F$ of finite measure. Then $m(E_n)$ converges to $m(E)$.</li>
</ul>


### Non-measurable sets
{: #non-measurable-sets}


## Lebesgue integral
{: #lebesgue-integral}


## References
{: #references}
[1] <span id='taos-book'>Terence Tao. [An introduction to measure theory](https://terrytao.wordpress.com/books/an-introduction-to-measure-theory/). Graduate Studies in Mathematics, vol. 126.</span>

[2] <span id='steins-book'>Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](#http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). </span>

## Footnotes
{: #footnotes}

[^1]: The **diameter** of a set $B$ is defined as
	\begin{equation}
	\text{dia}(B)\doteq\sup\\{\vert x-y\vert:x,y\in B\\}
	\end{equation}
