---
layout: post
title:  "Measure Theory (in probability perspective)"
date:   2022-08-20 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory probability-statistics random-stuffs
description: Note on Measure Theory
comments: true
---
> Measure Theory (in probability perspective)
<!-- excerpt-end -->

- [Probability Spaces](#prob-spaces)
- [References](#references)
- [Footnotes](#footnotes)

## Probabilities Spaces
{: #prob-spaces}
A **probability space** is a triple $(\Omega,\mathcal{F},P)$ where 
- $\Omega$ is a set of "outcomes";
- $\mathcal{F}$ is a set of "events";
- $P:\mathcal{F}\to[0,1]$ is a function that assigns probabilities to events.

We assume that $\mathcal{F}$ is a **$\sigma$-field** (or **$\sigma$-algebra**), i.e., a (nonempty) collection of subsets of $\Omega$, that satisfy  
(i) If $A\in\mathcal{F}$ then $A^c\in\mathcal{F}$[^1].  
(ii) If $A_i\in\mathcal{F}$ is a countable sequence of sets then $\cup_i A_i\in\mathcal{F}$.  
Since $\cap_i A_i=\left(\cup_i A_i\right)^c$, it follows that a $\sigma$-field is closed[^2] under countable intersection.

Without $P$, $(\Omega,\mathcal{F})$ is called a **measurable space**, i.e., a space on which we can put on a measure. A **measure** is a nonnegative countably additive set function; that is, a function $\mu:\mathcal{F}\to\mathbb{R}$ with  
(i) $\mu(A)\geq\mu(\emptyset)=0$ for all $A\in\mathcal{F}$.
(ii) If $A_i\in\mathcal{F}$ is a countable sequence of disjoint sets, then
\begin{equation}
\mu\left(\bigcup_i A_i\right)=\sum_i\mu(A_i)
\end{equation}

If $\mu(\Omega)=1$, then $\mu$ is called a **probability measure**.

**Theorem 1**  
Let $\mu$ be a measure on $(\Omega,\mathcal{F})$. Then $\mu$ has these following properties:  
*(i)* **monotonicity**. If $A\subset B$ then $\mu(A)\leq\mu(B)$.  
*(ii)* **subadditivity**. If $A\subset\bigcup_{i=1}^{\infty}A_i$ then $\mu(A)\leq\sum_{i=1}^{\infty}\mu(A_i)$.  
*(iii)* **continuity from below**. If $A_i\uparrow A$ (i.e., $A_1\subset A_2\subset\dots$ and $\bigcup_i A_i=A$) then $\mu(A_i)\uparrow\mu(A)$.  
*(iv)* **continuity from above**. If $A_i\downarrow A$ (i.e., $A_1\supset A_2\supset\dots$ and $\bigcap_i A_i=A$) with $\mu(A_i)<\infty$ then $\mu(A_i)\downarrow\mu(A)$.

**Proof**  
(i) Using $+$ to denote disjoint union and $-$ to represent **difference** of the two sets[^3], since $B=A+(B-A)$ so
\begin{equation}
\mu(B)=\mu(A)+\mu(B-A)\geq\mu(A)
\end{equation}
(ii) Let $A_i'\doteq A\cup A_i$; let $B_i\doteq A_i'-\bigcup_{j=1}^{i=1}(A_j')^c$ and $B_1\doteq A_1'$. Thus, we have that $B_i$'s are disjoint and $A=\bigcup_i B_i$.  
Using (i) in the definition of measure, $B_i\subset A_i$, and (i) of this theorem, we have
\begin{equation}
\mu(A)\doteq\sum_{i=1}^{\infty}\mu(B_i)\leq\sum_{i=1}^{\infty}\mu(A_i)
\end{equation}
(iii) Let $B_i\doteq A_i-A_{i-1}$. Therefore, $B_i$'s are disjoint and $\bigcup_{i=1}^{\infty}B_i=A$ and $\bigcup_{i=1}^{n}B_i=A_n$, thus,
\begin{equation}
\mu(A)=\sum_{i=1}^{\infty}\mu(B_i)=\lim_{n\to\infty}\sum_{i=1}^{n}\mu(B_i)=\lim_{n\to\infty}\mu(A_n)
\end{equation}
(iv) Since $A_i\downarrow A$, we have that $A_1-A_i\uparrow A_1-A$.  
Using (iii) and $A_1\supset A$, we obtain that $\mu(A_1-A_i)\uparrow\mu(A_1-A)$. And since $A_1\supset A_i$, we also have that $\mu(A_1-A_i)=\mu(A_1)-\mu(A_i)$.

## References
{: #references}
[1] Rick Durrett. [Probability: Theory and Examples](https://www.amazon.com/Probability-Cambridge-Statistical-Probabilistic-Mathematics/dp/0521765390). Cambridge University Press; 4th edition (August 30, 2010). 

[2] Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](#http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf).

## Footnotes
{: #footnotes}

[^1]: The **complement** of a set $A\in\mathbb{R}^d$, denoted as $A^c$, is defined by
	\begin{equation}
	A^c=\\{x\in\mathbb{R}^d:\,x\notin A\\}
	\end{equation}

[^2]: The **open ball** in $\mathbb{R}^d$ centered at $x$ and of radius $r$ is defined by
	\begin{equation}
	B_r(x)=\\{y\in\mathbb{R}^d:\vert y-x\vert< r\\}
	\end{equation}
	A subset $E\subset\mathbb{R}^d$ is **open** if for every $x\in E$ there exists $r>0$ with $B_r(x)\subset E$.  
	And a set is **closed** if its complement is open.  
	Any (not necessarily countable) union of open sets is open, while in general, the intersection of only finitely many open sets is open.  
	A similar statement holds for the class of closed sets, if we interchange the roles of unions and intersections.

[^3]: **Difference** of two sets $A$ and $B$, denoted as $B-A$, is defined as
	\begin{equation}
	B-A=B\cap A^c
	\end{equation}
