---
layout: post
title:  "Measure theory"
date:   2022-08-20 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory random-stuffs
description: Note on measure theory
comments: true
---
> A note on measure theory 
<!-- excerpt-end -->


- [Elementary measure](#elementary-measure)
	- [Intervals, boxes, elementary sets](#intervals-boxes-elementary-sets)
	- [Measure of an elementary set](#measure-elementary-set)
- [Jordan measure](#jordan-measure)
	- [Characteristic](#jordan-m-characteristic)
	- [Connection with the Riemann integral](#connect-riemann-int)
- [References](#references)
- [Footnotes](#footnotes)

### Elementary measure
{: #elementary-measure}

#### Intervals, boxes, elementary sets
{: #intervals-boxes-elementary-sets} 
An **interval** is a subset of $\mathbb{R}$ having one of the forms
\begin{align}
[a,b]&\doteq\\{x\in\mathbb{R}:a\leq x\leq b\\}, \\\\ [a,b)&\doteq\\{x\in\mathbb{R}:a\leq x\lt b\\}, \\\\ (a,b]&\doteq\\{x\in\mathbb{R}:a\lt x\lt b\\}, \\\\ (a,b)&\doteq\\{x\in\mathbb{R}:a\lt x\lt b\\},
\end{align}
where $a\leq b$ are real numbers.  
The **length** of an interval $I=[a,b],[a,b),(a,b],(a,b)$ is denoted as $\vert I\vert$ and is defined by
\begin{equation}
\vert I\vert\doteq b-a
\end{equation}
A **box** in $\mathbb{R}^d$ is a Cartesian product $B\doteq I_1\times\dots\times I_d$ of $d$ intervals $I_1,\dots,I_d$ (not necessarily the same length). The **volume** $\vert B\vert$ of such a box $B$ is defined as
\begin{equation}
\vert B\vert\doteq \vert I_1\vert\times\dots\times\vert I_d\vert
\end{equation}
An **elementary set** is any subset of $\mathbb{R}^d$ which is the union of a finite number of boxes.

**Example 1** (**Boolean closure**) 
If $E,F\subset\mathbb{R}^d$ are elementary sets, then the union $E\cup F$, the intersection $E\cap F$, the set theoretic difference $E\backslash F\doteq\\{x\in E:x\notin F\\}$, and the symmetric difference $E\Delta F\doteq(E\backslash F)\cup(F\backslash E)$ are also elementary. If $x\in\mathbb{R}^d$, then the translate $E+x\doteq\\{y+x:y\in E\\}$ is also an elementary set.

**Solution**  
With their definitions as elementary sets, we can assume that
\begin{align}
E&=B_1\cup\dots\cup B_k, \\\\ F&=B_1'\cup\dots\cup B_{k'}',
\end{align}
where each $B_i$ and $B_i'$ is a d-dimensional box. By set theory, we have that
- The union of $E$ and $F$ can be written as
\begin{equation}
E\cup F=B_1\cup\dots\cup B_k\cup B_1'\cup\dots\cup B_{k'}',
\end{equation}
which is an elementary set.
- The intersection of $E$ and $F$ can be written as
\begin{align}
E\cap F&=\left(B_1\cup\dots\cup B_k\right)\cup\left(B_1'\cup\dots\cup B_{k'}'\right) \\\\ &=\bigcup_{i=1,j=1}^{k,k'}\left(B_i\cap B_j'\right),
\end{align}
which is also an elementary set.
- The set theoretic difference of $E$ and $F$ can be written as
\begin{align}
E\backslash F&=\left(B_1\cup\dots\cup B_k\right)\backslash\left(B_1'\cup\dots\cup B_{k'}'\right) \\\\ &=\bigcup_{i=1,j=1}^{k,k'}\left(B_i\backslash B_j'\right),
\end{align}
which is, once again, an elementary set.
- With this display, the symmetric difference of $E$ and $F$ can be written as
\begin{align}
E\Delta F&=\left(E\backslash F\right)\cup\left(F\backslash E\right) \\\\ &=\Bigg[\bigcup_{i=1,j=1}^{k,k'}\left(B_i\backslash B_j'\right)\Bigg]\cup\Bigg[\bigcup_{i=1,j=1}^{k,k'}\left(B_j'\backslash B_i\right)\Bigg],
\end{align}
which satisfies conditions of an elementary set.

#### Measure of an elementary set
{: #measure-elementary-set}
**Lemma 1**  
Let $E\subset\mathbb{R}^d$ be an elementary set  
<ul id="roman-list">
	<li>$E$ can be expressed as the finite union of disjoint boxes</li>
	<li>If $E$ is partitioned as the finite union $B_1\cup\dots\cup B_k$ of disjoint boxes, then the quantity $m(E)\doteq\vert B_1\vert+\dots+\vert B_k\vert$ is independent of the partition. In other words, given any other partition $B_1'\cup\dots\cup B_{k'}'$ of $E$, we have</li>
	\begin{equation}
	\vert B_1\vert+\dots+\vert B_k\vert=\vert B_1'\vert+\dots+\vert B_{k'}'\vert
	\end{equation}
</ul>

We refer to $m(E)$ as the **elementary measure** of $E$.

**Proof**  
<ul id='roman-list'>
	<li>Consider the one-dimensional case, with these $k$ intervals, we can put their $2k$ endpoints into an increasing-order list (discarding repetitions). By looking at the open intervals between these end points, together with the endpoints themselves (viewed as intervals of length zero), we see that there exists a finite collection of disjoint intervals $J_1,\dots,J_{k'}$, such that each of the $I_1,\dots,I_k$ are union of some collection of the $J_1,\dots,J_{k'}$. And since each interval is a one-dimensional box, our statement has been proved with $d=1$.<br>
	In order to prove the multi-dimensional case, we begin by expressing $E$ as
	\begin{equation}
	E=\bigcap_{i=1}^{k}B_i,
	\end{equation}
	where each box $B_i=I_{i,1}\times\dots\times I_{i,d}$. For each $j=1,\dots,d$, since we has proved the one-dimensional case, we can express $I_{1,j},\dots I_{k,j}$ as the union of subcollections of collections $J_{1,j},\dots,J_{k',j}$ of disjoint intervals. Taking Cartesian product, we can express the $B_1,\dots,B_k$ as finite unions of box $J_{i_1,1}\times\dots\times J_{i_d,d}$, where $1\leq i_j\leq k_j'$ for all $1\leq j\leq d$. Moreover such boxes are disjoint, which proved our argument.</li>
	<li> We have that the length for an interval $I$ can be computed as
	\begin{equation}
	\vert I\vert=\lim_{N\to\infty}\frac{1}{N}\#\left(I\cap\frac{1}{N}\mathbb{Z}\right),
	\end{equation}
	where $\#A$ represents the cardinality of a finite set $A$ and 
	\begin{equation}
	\frac{1}{N}\mathbb{Z}\doteq\left\{\frac{x}{N}:x\in\mathbb{Z}\right\}
	\end{equation}
	Thus, volume of the box, say $B$, established from $d$ intervals $I_1,\dots,I_d$ by taking Cartesian product of them can be written as
	\begin{equation}
	\vert B\vert=\lim_{N\to\infty}\frac{1}{N^d}\#\left(B\cap\frac{1}{N}\mathbb{Z}^d\right)
	\end{equation}
	Therefore, with $k$ disjoint boxes $B_1,\dots,B_k$, we have that
	\begin{align}
	\vert B_1\vert+\dots+\vert B_k\vert&=\lim_{N\to\infty}\frac{1}{N^d}\#\left[\left(\bigcup_{i=1}^{k}B_i\right)\cap\frac{1}{N}\mathbb{Z}^d\right] \\\\ &=\lim_{N\to\infty}\frac{1}{N^d}\#\left(E\cap\frac{1}{N}\mathbb{Z}^d\right) \\\\ &=\lim_{N\to\infty}\frac{1}{N^d}\#\left[\left(\bigcup_{i=1}^{k'}B_i'\right)\cap\frac{1}{N}\mathbb{Z}^d\right] \\\\ &=\vert B_1'\vert+\dots+\vert B_{k'}'\vert
	\end{align}
	</li>
</ul>

**Remark**: From the definition of elementary measure, it is easily seen that, for any elementary sets $E$ and $F$ (not necessarily disjoint),
- $m(E)$ is a nonnegative real number (**non-negativity**), and has **finite additivity property**:
\begin{equation}
m(E\cup F)=m(E)+m(F)
\end{equation}
And by induction, it also implies that
\begin{equation}
m(E_1\cup\dots\cup E_k)=m(E_1)+\dots+m(E_k),
\end{equation}
whenever $E_1,\dots,E_k$ are disjoint elementary sets.
- $m(\emptyset)=0$.
- $m(B)=\vert B\vert$ for all box $B$.
- From non-negativity, finite additivity and **Example 1**, we conclude the **monotonicity** property, i.e., $E\subset F$ implies that
\begin{equation}
m(E)\leq m(F)
\end{equation}
- From the above and finite additivity, we also obtain the **finite subadditivity** property
\begin{equation}
m(E\cup F)\leq m(E)+m(F)
\end{equation}
- And by induction, we then have
\begin{equation}
m(E_1\cup\dots\cup E_k)\leq m(E_1)+\dots+m(E_k),
\end{equation}
whenever $E_1,\dots,E_k$ are elementary sets (not necessarily disjoint).
- we also have the **translation invariance** property
\begin{equation}
m(E+x)=m(E),\hspace{1cm}\forall x\in\mathbb{R}^d
\end{equation}

**Example 2**. (**Uniqueness of elementary measure**)  
Let $d\geq 1$ and let $m':\mathcal{E}(\mathbb{R}^d)\to\mathbb{R}^+$ be a map from the collection $\mathcal{E}(\mathbb{R}^d)$ of elementary subsets of $\mathbb{R}^d$ to the nonnegative reals that obeys the non-negativity, finite additivity, and translation invariance properties. Then there exists a constant $c\in\mathbb{R}^+$ such that
\begin{equation}
m'(E)=cm(E),
\end{equation}
for all elementary sets $E$. In particular, if we impose the additional normalization $m'([0,1)^d)=1$, then $m'\equiv m$.

**Solution**  


### Jordan measure
{: #jordan-measure}
Let $E\subset\mathbb{R}^d$ be a bounded set.
- The **Jordan inner measure** $m_{\*,(J)}(E)$ of $E$ is defined as
\begin{equation}
m_{\*,(J)}(E)\doteq\sup_{A\subset E,A\text{ elementary}}m(A)
\end{equation}
- The **Jordan outer measure** $m^{\*,(J)}(E)$ of $E$ is defined as
\begin{equation}
m^{\*,(J)}(E)\doteq\inf_{B\supset E,B\text{ elementary}}m(B)
\end{equation}
- If $m_{\*,(J)}(E)=m^{\*,(J)}(E)$, then we say that $E$ is **Jordan measurable**, and call
\begin{equation}
m(E)\doteq m_{\*,(J)}(E)=m^{\*,(J)}(E)
\end{equation}
the **Jordan measure** of $E$.

#### Characteristic
{: #jordan-m-characteristic}
Let $E\subset\mathbb{R}^d$ be bounded. Then these following statement are equivalence
<ul id='number-list'>
	<li>$E$ is Jordan measurable.</li>
	<li>For every $\epsilon>0$, there exists elementary sets $A\subset E\subset B$ such that $m(B\backslash A)\leq\epsilon$.</li>
	<li>For every $\epsilon>0$, there exists an elementary set $A$ such that $m^{*,(J)}(A\Delta E)\leq\epsilon$.</li>
</ul>

**Proof**  


**Corollary**  
Every elementary set $E$ is Jordan measurable.

#### Connection with the Riemann integral
{: #connect-riemann-int}





## References
{: #references}
[1] Terence Tao. [An introduction to measure theory](https://terrytao.wordpress.com/books/an-introduction-to-measure-theory/). Graduate Studies in Mathematics, vol. 126.

## Footnotes
{: #footnotes}

