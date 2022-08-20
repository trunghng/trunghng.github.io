---
layout: post
title:  "Measure Theory (more formal)"
date:   2022-08-20 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory random-stuffs
description: Note on Measure Theory
comments: true
---
> 
<!-- excerpt-end -->

- [Preliminaries](#preliminaries)
	- [Points, sets](#pts-sets)
	- [Open, closed and compact sets](#open-closed-compact-sets)
	- [Rectangles and cubes](#rect-cube)
	- [The Cantor set](#cantor-set)
- [The exterior measure](#exterior-measure)
	- [Definition](#ex-measure-def)
	- [Properties](#ex-measure-properties)
- [Measurable sets, the Lebesgue measure](#measurable-sets-lebesgue-measure)
	- [Invariance properties of Lebesgue measure](#inv-properties-lebesgue-measure)
	- [\\(\sigma\\)-algebras, Borel sets](#sigma-algebras-borel-sets)
	- [Non-measurable set construction](#non-measurable-set-const)
- [Measurable functions](#measurable-funcs)
	- [Definition](#measurable-func-def)
	- [Properties](#measurable-func-properties)
	- [Approximation by simple functions or step functions](#measurable-func-approx)
	- [Littlewood's three principles](#littlewoods-principles)
- [Refereces](#references)
- [Footnotes](#footnotes)

## Preliminaries
Before diving into details, we need some elementary concepts.

### Points, sets
{: #pts-sets}
A **point** $x\in\mathbb{R}^d$ consists of a $d$-tuple of real numbers
\begin{equation}
x=\left(x_1,x_2,\dots,x_d\right),\hspace{1cm}x_i\in\mathbb{R}, i=1,\dots,d
\end{equation}
Addition between points and multiplication of a point by a real scalar is elementwise.

The **norm** of $x$ is denoted by $\vert x\vert$ and is defined to be the standard **Euclidean norm** given by
\begin{equation}
\vert x\vert=\left(x_1^2+\dots+x_d^2\right)^{1/2}
\end{equation}
We can then calculate the **distance** between two points $x$ and $y$, which is \begin{equation}
d(x,y)=\vert x-y\vert
\end{equation}
The **complement** of a set $E$ in $\mathbb{R}^d$ is denoted as $E^c$, and defined by
\begin{equation}
E^c=\\{x\in\mathbb{R}^d:x\notin E\\}
\end{equation}
If $E$ and $F$ are two subsets of $\mathbb{R}^d$, we denote the complement of $F$ in $E$ by
\begin{equation}
E-F=\\{x\in\mathbb{R}^d:x\in E;\,x\notin F\\}
\end{equation}
The **distance** between two sets $E$ and $F$ is defined by
\begin{equation}
d(E,F)=\inf_{x\in E,\,y\in F}\vert x-y\vert
\end{equation}

### Open, closed and compact sets
{: #open-closed-compact-sets}
The **open ball** in $\mathbb{R}^d$ centered at $x$ and of radius $r$ is defined by
\begin{equation}
B_r(x)=\\{y\in\mathbb{R}^d:\vert y-x\vert< r\\}
\end{equation}
A subset $\mbox{for}$ for

### Rectangles and cubes
{: #rect-cube}

### The Cantor sets
{: #cantor-sets}

## References
{: #references}
[1] Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](#http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). 

## Footnotes
{: #footnotes}