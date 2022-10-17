---
layout: post
title:  "Convex sets, convex functions"
date:   2021-12-02 13:03:00 +0700
categories: mathematics optimization
tags: mathematics convex-optimization
description: convex sets, convex functions
comments: true
eqn-number: true
---
> A note on convex sets, convex functions

<!-- excerpt-end -->

- [Convex sets](#cvx-sets)
	- [Affine & convex sets](#aff-cvx-sets)
		- [Affine sets](#aff-sets)
		- [Affine dimension, relative interior](#aff-dim-rel-int)
		- [Convex sets](#cvx-sets-def)
- [Convex functions](#cvx-funcs)
- [References](#references)
- [Footnotes](#footnotes)

## Convex sets
{: #cvx-sets}

### Affine & convex sets
{: #aff-cvx-sets}

#### Affine sets
{: #aff-sets}
A set $C\subset\mathbb{R}^n$ is **affine** if the line through any two distinct points in $C$ lies in $C$, i.e. for any $x_1,x_2\in C$ and for any $\theta\in\mathbb{R}$ we have
\begin{equation}
\theta x_1+(1-\theta)x_2\in C
\end{equation}
A point of the form $\theta_1 x_1+\ldots+\theta_k x_k$, where $\theta_1+\ldots+\theta_k=1$ is known as an **affine combination** of the points $x_1,\ldots,x_k$.

Hence, if $C$ is an affine set, and $x_1,\ldots,x_k\in C$, and $\theta_1+\ldots+\theta_k=1$, then the point
\begin{equation}
\theta_1 x_1+\ldots+\theta_k x_k\in C
\end{equation}
If $C$ is an affine set and $x_0\in C$, then the set
\begin{equation}
V=C-x_0\\{x-x_0:x\in C\\}
\end{equation}
is a subspace.

The set of all affine combinations of points in some set $C\subset\mathbb{R}^n$ is referred as **affine hull** of $C$, denoted as $\text{aff}\,C$:
\begin{equation}
\text{aff}\,C=\\{\theta_1 x_1+\ldots+\theta_k x_k:x_1,\ldots,x_k\in C;\theta_1+\ldots+\theta_k=1\\}
\end{equation}
The affine hull is the *smallest* affine set containing $C$.

#### Affine dimension, relative interior
{: #aff-dim-rel-int}
The **affine dimension** of a set $C$ is defined as the dimension of $\text{aff}\,C$.

If the affine dimension of $C\subset\mathbb{R}^n$ is less than $n$, then the set lies in $\text{aff}\,C\neq\mathbb{R}^n$

#### Convex sets
{: #cvx-sets-def}

## Convex functions
{: #cvx-funcs}

## References
{: #references}
[1] Stephen Boyd & Lieven Vandenberghe. [Convex Optimization](http://www.stanford.edu/∼boyd/cvxbook/). Cambridge UP, 2004.

## Footnotes
{: #footnotes}