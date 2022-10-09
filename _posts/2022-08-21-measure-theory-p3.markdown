---
layout: post
title:  "Measure theory - III: the Lebesgue integral"
date:   2022-08-21 13:00:00 +0700
categories: mathematics measure-theory
tags: mathematics measure-theory lebesgue-integral
description: Note on measure theory part 3
comments: true
eqn-number: true
---
> Part III of the measure theory series. Materials are mostly taken from [Tao's book]({% post_url 2022-08-21-measure-theory-p3 %}#taos-book), except for some needed notations extracted from [Stein's book]({% post_url 2022-08-21-measure-theory-p3 %}#steins-book).
<!-- excerpt-end -->

- [Integration of simple functions](#int-simp-funcs)
	- [Simple function](#simp-func)
	- [Integral of unsigned simple functions](#int-unsgn-simp-func)
	- [Well-definedness of simple integral](#well-dfn-simp-int)
	- [Almost everywhere and support](#alm-evwhr-spt)
	- [Basic properties of the simple unsigned integral](#bsc-prop-simp-unsgn-int)
	- [Absolutely convergence simple integral](#abs-cvg-simp-int)
	- [Basic properties of the complex-valued simple integral](#bsc-prop-cmplx-simp-int)
- [Measurable functions](#msr-funcs)
	- [Unsigned measurable functions](#unsgn-msr-funcs)
	- [Equivalent notions of measurability](#equiv-ntn-msrb)
- [Unsigned Lebesgue integrals](#unsgn-lebesgue-int)
- [Absolute integrability](#abs-intb)
- [Littlewood's three principles](#littlewoods-prncpl)
- [References](#references)
- [Footnotes](#footnotes)

## Integration of simple functions
{: #int-simp-funcs}
Analogy to how the [**Riemann integral**]({% post_url 2022-06-16-measure-theory-p1 %}#riemann-integrability) was established by using the integral for [**piecewise constant functions**]({% post_url 2022-06-16-measure-theory-p1 %}#pc-func), the **Lebesgue integral** is set up using the integral for **simple functions**.

### Simple function
{: #simp-func}
A (complex-valued) **simple function** $f:\mathbb{R}^d\to\mathbb{C}$ is a finite linear combination
\begin{equation}
f=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k},\label{eq:sf.1}
\end{equation}
of indicator functions $1_{E_i}$ of Lebesgue measurable sets $E_i\subset\mathbb{R}^d$ for $i=1,\ldots,k$, for natural number $k\geq 0$ and where $c_1,\ldots,c_k\in\mathbb{C}$ are complex numbers.

An **unsigned simple function** $f:\mathbb{R}^d\to[0,+\infty]$ is given as \eqref{eq:sf.1} but with the $c_i$ taking values in $[0,+\infty]$ rather than $\mathbb{C}$.

### Integral of a unsigned simple function
{: #int-unsgn-simp-func}
If $f=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}$ is an unsigned simple function, the integral $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx$ is defined by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx\doteq c_1 m(E_1)+\ldots+c_k m(E_k),
\end{equation}
which means $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx\in[0,+\infty]$.

### Well-definedness of simple integral
{: #well-dfn-simp-int}
**Lemma 1**  
*Let $k,k'\geq 0$ be natural, $c_1,\ldots,c_k,c_1',\dots,c_k'\in[0,+\infty]$ and $E_1,\ldots,E_k,E_1',\ldots,E_k'\subset\mathbb{R}^d$ be Lebesgue measurable sets such that the identity
\begin{equation}
c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}=c_1' 1\_{E_1'}+\ldots+c_k' 1\_{E_k'}
\end{equation}
hold identically on $\mathbb{R}^d$. Then we have*
\begin{equation}
c_1 m(E_1)+\ldots+c_k m(E_k)=c_1' m(E_1')+\ldots+c_k' m(E_k')
\end{equation}

**Proof**  
The $k+k'$ sets $E_1,\ldots,E_k,E_1',\ldots,E_k'$ partition $\mathbb{R}^d$ into $2^{k+k'}$ disjoint sets, each of which is an intersection of some of the $E_1,\ldots,E_k,E_1',\ldots,E_k'$ and their complements.


### Almost everywhere and support
{: #alm-evwhr-spt}
A property $P(x)$ of a point $x\in\mathbb{R}^d$ is said to hold **(Lebesgue) almost everywhere** in $\mathbb{R}^d$ or for **(Lebesgue) almost every point** $x\in\mathbb{R}^d$, if the set of $x\in\mathbb{R}^d$ for which $P(x)$ fails has Lebesgue measure of zero (i.e. $P$ is true outside of a null set).

Two functions $f,g:\mathbb{R}^d\to Z$ into an arbitrary range $Z$ are referred to **agree almost everywhere** if we have $f(x)=g(x)$ almost every $x\in\mathbb{R}^d$.

The **support** of a function $f:\mathbb{R}^d\to\mathbb{C}$ or $f:\mathbb{R}^d\to[0,+\infty]$ is defined to be the set $\\{x\in\mathbb{R}^d:f(x)\neq 0\\}$ where $f$ is non-zero.

**Remark 2**  
- If $P(x)$ holds for almost every $x$, and $P(x)$ implies $Q(x)$, then $Q(x)$ holds for almost every $x$.
- If $P_1(x),P_2(x),\ldots$ are an at most countable family of properties, each of which individually holds for almost every $x$, then they will simultaneously holds for almost every $x$, since the countable union of null sets is still a null set.

### Basic properties of the simple unsigned integral
{: #bsc-prop-simp-unsgn-int}
Let $f,g:\mathbb{R}^d\to[0,+\infty]$ be simple unsigned functions.
<ul id='roman-list'>
	<li>
		<b>Unsigned linearity</b>. We have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\,dx=\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{equation}
		and
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\,dx=c\,\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx,
		\end{equation}
		for all $c\in[0,+\infty]$.
	</li>
	<li>
		<b>Finiteness</b>. We have $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx<\infty$ iff $f$ is finite almost everywhere, and its support has finite measure.
	</li>
	<li>
		<b>Vanishing</b>. We have $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx=0$ iff $f$ is zero almost everywhere.
	</li>
	<li>
		<b>Equivalence</b>. If $f$ and $g$ agree almost everywhere, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx=\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{equation}
	</li>
	<li>
		<b>Monotonicity</b>. If $f(x)\leq g(x)$ for almost every $x\in\mathbb{R}^d$, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx\leq\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{equation}
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b>. For any Lebesgue measurable $E$, we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}1_E(x)\,dx=m(E)
		\end{equation}
	</li>
</ul>

**Proof**  
Since $f,g:\mathbb{R}^d\to[0,+\infty]$ are simple unsigned functions, we can assume that
\begin{align}
f&=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}, \\\\ g&=c_1 1\_{E_1}+\ldots+c_k' 1\_{E_k'},
\end{align}
where $c_1,\ldots,c_k,c_1',\ldots,c_k'\in[0,+\infty]$.
<ul id='roman-list'>
	<li>
		<b>Unsigned linearity</b><br>
		We have
		\begin{align}
		\hspace{-1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\,dx&=c_1 m(E_1)+\ldots+c_k m(E_k)+c_1' m(E_1')+\ldots+c_k' m(E_k') \\ &=\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{align}
		For any $c\in[0,+\infty]$, we have
		\begin{align}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\,dx&=c\left(c_1 m(E_1)+\ldots+c_k m(E_k)\right) \\ &=c\,\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx
		\end{align}
	</li>
	<li>
		<b>Finiteness</b><br>
		Given $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx<\infty$, then for every $i=1,\ldots,k$ we have that
		\begin{equation}
		c_i m(E_i)<\infty\label{eq:bpsui.1}
		\end{equation}
		Suppose that $f$ is not finite almost everywhere, which means that there exists $1\leq i\leq k$ such that $E_i$ is a non-null set and $c_i=\infty$, or
		\begin{equation}
		c_i m(E_i)=\infty,
		\end{equation}
		which is in contrast with \eqref{eq:bpsui.1}.<br>
		Suppose that the support of $f$ has infinite measure, or in other word
		\begin{equation}
		c_i\neq 0,\hspace{1cm}i=1,\ldots,k\label{eq:bpsui.2}
		\end{equation}
		and
		\begin{equation}
		m\left(\bigcup_{n=1}^{k}E_n\right)=\infty,
		\end{equation}
		Since any $k$ subsets $E_1,\ldots,E_k$ of $\mathbb{R}^d$ partition $\mathbb{R}^d$ into $2^k$ disjoint sets, say $F_1,\ldots,F_{2^k}$. Hence, by finite additivity property of Lebesgue measure, we have
		\begin{equation}
		\sum_{n=1}^{2^k}m(F_n)=\infty,
		\end{equation}
		which implies that there exists $1\leq n\leq 2^k$ such that $m(F_n)=\infty$. And therefore, for a particular $1\leq i\leq k$ such that $F_n\subset E_i$, by monotonicity property of Lebesgue measure
		\begin{equation}
		m(E_i)\geq m(F_n)=\infty
		\end{equation}
		Thus, combining with \eqref{eq:bpsui.2} gives us
		\begin{equation}
		c_i m(E_1)=\infty,
		\end{equation}
		which again contradicts to \eqref{eq:bpsui.1}.<br>
		Given $f$ is finite almost everywhere and its support has finite measure, suppose that its integral is infinite, or
		\begin{equation}
		c_1 m(E_1)+\ldots+c_k m(E_k)=\infty,
		\end{equation}
		which implies that there exists $1\leq i\leq k$ such that either<br>
		(1) $c_i=\infty$ and $E_i$ is a non-null set, or<br>
		(2) $c_i\neq 0$ and $m(E)=\infty$.<br>
		If (1) happens, we then have that
		\begin{equation}
		f\geq c_i 1_{E_i}=\infty,
		\end{equation}
		which contradicts to our hypothesis.<br>
		If (2) happens, by monotonicity of Lebesgue measure, the support of $f$ then has infinite measure, which also contradicts to our hypothesis.
	</li>
	<li>
		<b>Vanishing</b><br>
		Given $\text{Simp}\int_{\mathbf{R^d}}f(x)\,dx=0$, we then have
		\begin{equation}
		c_1 m(E_1)+\ldots+c_k m(E_k)=0,
		\end{equation}
		which implies that for every $1\leq i\leq k$, we have that $c_i=0$ or $E_i$ is a null set.
		Therefore, $f$ is zero almost everywhere because in this case $f$ takes the value of non-zero iff $x$ is in a particular null set $E_j$.<br>
		Given $f$ is zero almost everywhere, for every $i=1,\ldots,k$, we have that either<br>
		(1) $c_i=0$, or<br>
		(2) $c_i\neq 0$ and $x\notin E_i$ with $E_i$ is a null set.<br>
		Therefore, the integral of $f$
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx=c_1 m(E_1)+\ldots+c_k m(E_k)=0
		\end{equation}
	</li>
	<li>
		<b>Equivalence</b><br>
	</li>
	<li>
		<b>Monotonicity</b><br>
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b><br>
	</li>
</ul>


### Absolutely convergence simple integral
{: #abs-cvg-simp-int}
A complex valued simple function $f:\mathbb{R}^d\to\mathbb{C}$ is known as **absolutely integrable** if
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}\vert f(x)\vert\,dx<\infty
\end{equation}
If $f$ is absolutely integrable, the integral $\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx$ is defined for real signed $f$ by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx\doteq\text{Simp}\int_{\mathbb{R}^d}f_+(x)\,dx+\text{Simp}\int_{\mathbb{R}^d}f_-(x)\,dx,
\end{equation}
where
\begin{align}
f_+(x)&\doteq\max\left(f(x),0\right), \\\\ f_-(x)&\doteq\max\left(-f(x),0\right),
\end{align}
and for complex-valued $f$ by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx\doteq\text{Simp}\int_{\mathbb{R}^d}\text{Re}\,f(x)\,dx+i\,\text{Simp}\int_{\mathbb{R}^d}\text{Im}\,f(x)\,dx
\end{equation}

### Basic properties of the complex-valued simple integral
{: #bsc-prop-cmplx-simp-int}
Let $f,g:\mathbb{R}^d\to\mathbb{C}$ be absolutely integrable simple functions
<ul id='roman-list'>
	<li>
		<b>*-linearity</b>. We have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\,dx=\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{equation}
		and
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\,dx=c\,\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx,
		\end{equation}
		for all $c\in\mathbb{C}$. Also we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}\overline{f}(x)\,dx=\overline{\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx}
		\end{equation}
	</li>
	<li>
		<b>Equivalence</b>. If $f$ and $g$ agree almost everywhere, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\,dx=\text{Simp}\int_{\mathbb{R}^d}g(x)\,dx
		\end{equation}
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b>. For any Lebesgue measurable $E$, we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}1_E(x)\,dx=m(E)
		\end{equation}
	</li>
</ul>

**Proof**
<ul id='roman-list'>
	<li>
		<b>*-linearity</b><br>
	</li>
	<li>
		<b>Equivalence</b><br>
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b><br>
	</li>
</ul>

## Measurable functions
{: #msr-funcs}
Just as how the piecewise constant integral can be extended to the Riemann integral, the unsigned simple integral can be extended to the unsigned Lebesgue integral, by expanding the class of unsigned simple functions to the broader class of **unsigned Lebesgue measurable functions**.

### Unsigned measurable functions
{: #unsgn-msr-funcs}
An unsigned function $f:\mathbb{R}^d\to[0,+\infty]$ is **unsigned Lebesgue measurable**, or **measurable**, if it is the pointwise limit of unsigned simple functions, i.e. if there exists a sequence $f_1,f_2,\ldots:\mathbb{R}\to[0,+\infty]$ of unsigned simple functions such that $f_n(x)\to f(x)$ for every $x\in\mathbb{R}^d$.

### Equivalent notions of measurability
{: #equiv-ntn-msrb}
**Lemma 3**  
Let $f:\mathbb{R}\to[0,+\infty]$ be an unsigned function. The following are then equivalent:
<ul id='roman-list'>
	<li>
		$f$ is unsigned Lebesgue measurable.
	</li>
	<li>
		$f$ is the pointwise limit of unsigned simple functions $f_n$ (hence $\lim_{n\to\infty}f_n(x)$ exists and is equal to $f(x)$ for all $x\in\mathbb{R}^d$).
	</li>
	<li>
		$f$ is the pointwise almost everywhere limit of unsigned simple function $f_n$ (thus $\lim_{n\to\infty}f_n(x)$ exists and is equal to $f(x)$ for almost every $x\in\mathbb{R}^d$).
	</li>
	<li>
		$f(x)=\sup_n f_n(x)$, where $0\leq f_1\leq f_2\leq\ldots$ is an increasing sequence of unsigned simple functions, each of which are bounded with finite measure support.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)>\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)\geq\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)<\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)\leq\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every interval $I\subset[0,+\infty)$, the set $f^{-1}(I)\doteq\{x\in\mathbb{R}^d:f(x)\in I\}$ is Lebesgue measurable.
	</li>
	<li>
		For every (relatively) open set $U\subset[0,+\infty)$, the set $f^{-1}(U)\doteq\{x\in\mathbb{R}^d:f(x)\in U\}$ is Lebesgue measurable.
	</li>
	<li>
		For every (relatively) closed set $K\subset[0,+\infty)$, the set $f^{-1}(K)\doteq\{x\in\mathbb{R}^d:f(x)\in K\}$ is Lebesgue measurable.
	</li>
</ul>

**Proof**

## Unsigned Lebesgue integrals
{: #unsgn-lebesgue-int}

## Absolute integrability
{: #abs-intb}

## Littlewood's three principles
{: #littlewoods-prncpl}

## References
{: #references}
[1] <span id='taos-book'>Terence Tao. [An introduction to measure theory](https://terrytao.wordpress.com/books/an-introduction-to-measure-theory/). Graduate Studies in Mathematics, vol. 126.</span>

[2] <span id='steins-book'>Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](#http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). </span>

## Footnotes
{: #footnotes}