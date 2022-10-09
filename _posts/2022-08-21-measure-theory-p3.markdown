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
- [Measurable functions](#msr-funcs)
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
Let $k,k'\geq 0$ be natural, $c_1,\ldots,c_k,c_1',\dots,c_k'\in[0,+\infty]$ and $E_1,\ldots,E_k,E_1',\ldots,E_k'\subset\mathbb{R}^d$ be Lebesgue measurable sets such that the identity
\begin{equation}
c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}=c_1' 1\_{E_1'}+\ldots+c_k' 1\_{E_k'}
\end{equation}
hold identically on $\mathbb{R}^d$. Then we have
\begin{equation}
c_1 m(E_1)+\ldots+c_k m(E_k)=c_1' m(E_1')+\ldots+c_k' m(E_k')
\end{equation}

**Proof**  


### Almost everywhere and support
{: #alm-evwhr-spt}

### Basic properties of the simple unsigned integral
{: #bsc-prop-simp-unsgn-int}

### Absolutely convergence simple integral
{: #abs-cvg-simp-int}

## Measurable functions
{: #msr-funcs}

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