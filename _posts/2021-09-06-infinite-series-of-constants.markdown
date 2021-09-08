---
layout: post
title:  "Infinite Series of Constants"
date:   2021-09-06 11:20:00 +0700
categories: mathematics calculus
tags: mathematics calculus series random-stuffs
description: infinite series of constants
comments: true
---
> No idea what to say yet :D

<!-- excerpt-end -->
- [Infinite Series](#infinite-series)
	- [Examples](#examples)
- [Convergent Sequences](#convergent-sequences)
	- [Sequences](#sequences)
	- [Limits of Sequences](#lim-seq)
- [Convergent and Divergent Series](#conv-div-series)
	- [$n$-th term test](#nth-term-test)
- [General Properties of Convergent Series](#gen-props-conv-series)
- [Series of Nonnegative terms. Comparison Tests](#series-nonneg-ct)
- [References](#references)
- [Footnotes](#footnotes)


## Infinite Series
An **infinite series**, or simply a **series**, is an expression of the form
\begin{equation}
a_1+a_2+\dots+a_n+\dots=\sum_{n=1}^{\infty}a_n
\end{equation}

### Examples
1. *Infinite decimal*
\begin{equation}
.a_1a_2\ldots a_n\ldots=\dfrac{a_1}{10}+\dfrac{a_2}{10^2}+\ldots+\dfrac{a_n}{10^n}+\ldots,
\end{equation}
where $a_i\in\\{0,1,\dots,9\\}$.  

2. *Power series expansion*[^1]
- We have
\begin{equation}
\dfrac{1}{1-x}=1+x+x^2+\dots\tag{1}\label{1}
\end{equation}
- If we replace $x$ by $-x$ in \eqref{1} we have
\begin{equation}
\dfrac{1}{1+x}=1-x+x^2-x^3+\dots\tag{2}\label{2}
\end{equation}
And if we replace $x$ by $x^2$ in \eqref{2} we obtain
\begin{equation}
\dfrac{1}{1+x^2}=1-x^2+x^4-x^6+\dots\tag{3}\label{3}
\end{equation}
Moreover, if we take the integral of the left side of \eqref{3} we get
\begin{equation}
\int\dfrac{dx}{1+x^2}=\tan^{-1}x,
\end{equation}
which leads to the result that
\begin{equation}
\tan^{-1}x=x-\dfrac{x^3}{3}+\dfrac{x^5}{5}-\dfrac{x^7}{7}+\dots\tag{4}\label{4}
\end{equation}
Let $x=1$, \eqref{4} gives us an interesting result
\begin{equation}
\dfrac{\pi}{4}=1-\dfrac{1}{3}+\dfrac{1}{5}-\dfrac{1}{7}+\dots
\end{equation}

## Convergent Sequences

### Sequences
If to each positive integer $n$ there corresponds a definite number $x_n$, then the $x_n$'s are said to form a **sequence** (denoted as $\\{x_n\\}$)
\begin{equation}
x_1,x_2,\dots,x_n,\dots
\end{equation}
We call the numbers constructing a sequence its terms, where $x_n$ is the $n$-th term.  

A sequence $\\{x_n\\}$ is said to be *bounded* if there exists $A, B$ such that $A\leq x_n\leq B, \forall n$. $A, B$ respectively are called *lower bound*, *upper bound* of the sequence. A sequence that is not bounded is said to be *unbounded*.

### Limits of Sequences
{: #lim-seq}
A sequence $\\{x_n\\}$ is said to have a number $L$ as **limit** if for each $\epsilon>0$, there exists a positive integer $n_0$ that
\begin{equation}
\vert x_n-L\vert<\epsilon\hspace{1cm}n\geq n_0
\end{equation}
We say that $x_n$ *converges to* $L$ *as* $n$ *approaches infinite* ($x_n\to L$ as $n\to\infty$) and denote this as
\begin{equation}
\lim_{n\to\infty}x_n=L
\end{equation}
- A sequence is said to **converge** or to be **convergent** if it has a limit.  
- A convergent sequence is bounded, but not all bounded sequences are convergent.
- If $x_n\to L,y_n\to M$, then
\begin{align}
&\lim(x_n+y_n)=L+M \\\\ &\lim(x_n-y_n)=L-M \\\\ &\lim x_n y_n=LM \\\\ &\lim\dfrac{x_n}{y_n}=\dfrac{L}{M}\hspace{1cm}M\neq0
\end{align}
- An *increasing* (or *decreasing*) sequence converges if and only if it is bounded.

## Convergent and Divergent Series
{: #conv-div-series}
Recall from the previous sections that if $a_1,a_2,\dots,a_n,\dots$ is a *sequence* of numbers, then
\begin{equation}
\sum_{n=1}^{\infty}a_n=a_1+a_2+\ldots+a_n+\ldots\tag{5}\label{5}
\end{equation}
is called an *infinite series*. We begin by establishing the sequence of *partial sums*
\begin{align}
s_1&=a_1 \\\\ s_2&=a_1+a_2 \\\\ &\,\vdots \\\\ s_n&=a_1+a_2+\dots+a_n \\\\ &\,\vdots
\end{align}
The series \eqref{5} is said to be **convergent** if the sequences $\\{s_n\\}$ converges. And if $\lim s_n=s$, then we say that \eqref{5} converges to $s$, or that $s$ is the sum of the series.
\begin{equation}
\sum_{n=1}^{\infty}a_n=s
\end{equation}
If the series does not converge, we say that it **diverges** or is **divergent**, and no sum is assigned to it.

**Examples** (*harmonic series*)  
Let's consider the convergence of *harmonic series*
\begin{equation}
\sum_{n=1}^{\infty}\frac{1}{n}=1+\frac{1}{2}+\frac{1}{3}+\ldots\tag{6}\label{6}
\end{equation}
Let $m$ be a positive integer and choose $n>2^{m+1}$. We have
\begin{align}
s_n&>1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+\dots+\frac{1}{2^{m+1}} \\\\ &=\left(1+\frac{1}{2}\right)+\left(\frac{1}{3}+\frac{1}{4}\right)+\left(\frac{1}{5}+\ldots+\frac{1}{8}\right)+\ldots+\left(\frac{1}{2^m+1}+\ldots+\frac{1}{2^{m+1}}\right) \\\\ &>\frac{1}{2}+2.\frac{1}{4}+4.\frac{1}{8}+\ldots+2^m.\frac{1}{2^{m+1}} \\\\ &=(m+1)\frac{1}{2}
\end{align}
This proves that $s_n$ can be made larger than the sum of any number of $\frac{1}{2}$'s and therefore as large as we please, by taking $n$ large enough, so the $\\{s_n\\}$ are unbounded, which leads to that \eqref{6} is a divergent series.
\begin{equation}
\sum_{n=1}^{\infty}\frac{1}{n}=1+\frac{1}{2}+\frac{1}{3}+\ldots=\infty
\end{equation}


The simplest general principle that is useful to study the convergence of a series is the **$\mathbf{n}$-th term test**.

### $\mathbf{n}$-th term test
{: #nth-term-test}
If the series $\\{a_n\\}$ converges, then $a_n\to0$ as $n\to\infty$; or equivalently, if $\neg(a_n\to0)$ as $n\to\infty$, then the series must necessarily diverge.  

**Proof**  
When $\\{a_n\\}$ converges, as $n\to\infty$ we have
\begin{equation}
a_n=s_n-s_{n-1}\to s-s=0
\end{equation}
This result shows that $a_n\to0$ is a necessary condition for convergence. However, it is not a sufficient condition; i.e., it does not imply the convergence of the series when $a_n\to0$ as $n\to\infty$.

## General Properties of Convergent Series
{: #gen-props-conv-series}
- Any finite number of 0's can be inserted or removed anywhere in a series without affecting its convergence behavior or its sum (in case it converges).
- When two convergent series are added term by term, the resulting series converges to the expected sum; i.e., if $\sum_{n=1}^{\infty}a_n=s$ and $\sum_{n=1}^{\infty}b_n=t$, then
\begin{equation}
\sum_{n=1}^{\infty}(a_n+b_n)=s+t
\end{equation}
	- **Proof**  
	Let $\\{s_n\\}$ and $\\{t_n\\}$ respectively be the sequences of partial sums of $\sum_{n=1}^{\infty}a_n$ and $\sum_{n=1}^{\infty}b_n$. As $n\to\infty$ we have
	\begin{align}
	(a_1+b_1)+(a_2+b_2)+\dots+(a_n+b_n)&=\sum_{i=1}^{n}a_i+\sum_{i=1}^{n}b_i \\\\ &=s_n+t_n\to s+t
	\end{align}
- Similarly, $\sum_{n=1}^{\infty}(a_n-b_n)=s-t$ and $\sum_{n=1}^{\infty}ca_n=cs$ for any constant $c$.
- Any finite number of terms can be added or subtracted at the beginning of a convergent series without disturbing its convergence, and the sum of various series are related in the expected way.
	- **Proof**  
	If $\sum_{n=1}^{\infty}a_n=s$, then
	\begin{equation}
	\lim_{n\to\infty}(a_0+a_1+a_2+\dots+a_n)=\lim_{n\to\infty} a_0+\lim_{n\to\infty}(a_1+a_2+\dots+a_n)=a_0+s
	\end{equation}

## Series of Nonnegative terms. Comparison Tests
{: #series-nonneg-ct}





## References
[1] George F.Simmons. [Calculus With Analytic Geometry - 2nd Edition](https://www.amazon.com/Calculus-Analytic-Geometry-George-Simmons/dp/0070576424)  

[2] MIT 18.01. [Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)

## Footnotes
[^1]: More information about power series are gonna be written in another post.
