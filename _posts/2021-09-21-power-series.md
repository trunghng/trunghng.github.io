---
layout: post
title:  "Power Series"
date:   2021-09-21 15:40:00 +0700
categories: mathematics calculus
tags: mathematics calculus series power-series taylor-series random-stuffs
description: power series
comments: true
---
> In the previous post, [Infinite Series of Constants]({% post_url 2021-09-06-infinite-series-of-constants %}), we mentioned a type of series called **power series** a lot. In the content of this post, we will be diving deeper into details of that series.

<!-- excerpt-end -->
- [Power Series](#power-series)
- [The Interval of Convergence](#int-conv)
- [Differentiation and Integration of Power Series](#dif-int-power-series)
- [References](#references)
- [Footnotes](#footnotes)

## Power Series
A **power series** is a series of the form
\begin{equation}
\sum_{n=0}^{\infty}a_nx^n=a_0+a_1x+a_2x^2+\ldots+a_nx^n+\ldots,
\end{equation}
where the coefficient $a_n$ are constants and $x$ is a variable.  

It is easily seen that the sum of the series $\sum_{n=0}^{\infty}a_nx^n$  is a function of $x$ since the sum depends only on $x$ for any value of $x$. Hence, we can denote this as
\begin{equation}
f(x)=\sum_{n=0}^{\infty}a_nx^n=a_0+a_1x+a_2x^2+\ldots+a_nx^n+\ldots
\end{equation}

## The Interval of Convergence
{: #int-conv}
Similar to what we have done in the post of [infinite series of constants]({% post_url 2021-09-06-infinite-series-of-constants %}), we begin studying properties of power series by considering their convergence behavior.  

**Lemma 1**  
*If a power series $\sum a_nx^n$ converges at $x_1$, with $x_1\neq 0$, then it converges [absolutely]({% post_url 2021-09-06-infinite-series-of-constants %}#abs-conv) at all $x$ with $\vert x\vert<\vert x_1\vert$; and if it diverges at $x_1$, then it diverges at all $x$ with $\vert x\vert>\vert x_1\vert$.*  

**Proof**  
By the [$n$-th term test]({% post_url 2021-09-06-infinite-series-of-constants %}#nth-term-test), we have that if $\sum a_nx^n$ converges, then $a_nx^n\to 0$. In particular, if $n$ is sufficiently large, then $\vert a_n{x_1}^n\vert<1$, and therefore
\begin{equation}
\vert a_nx^n\vert=\vert a_n{x_1}^n\vert\left\vert\dfrac{x}{x_1}\right\vert^n\<r^n,\tag{1}\label{1}
\end{equation}
where $r=\vert\frac{x}{x_1}\vert$. Suppose that $\vert x\vert<\vert x_1\vert$, we have
\begin{equation}
r=\left\vert\dfrac{x}{x_1}\right\vert<1,
\end{equation}
which leads to the result that geometric series $\sum r^n$ converges (with the sum $\frac{1}{1-r}$). And hence, from \eqref{1} and by the [comparison test]({% post_url 2021-09-06-infinite-series-of-constants %}#comparison-test), the series $\sum\vert a_nx^n\vert$ also converges.  

Moreover, if $\sum a_n{x_1}^n$ diverges, then $\sum\vert a_n{x_1}^n\vert$ also diverges. By the [comparison test]({% post_url 2021-09-06-infinite-series-of-constants %}#comparison-test), for any $x$ such that $\vert x\vert>\vert x_1\vert$, we also have that $\sum\vert a_nx^n\vert$ diverges. This leads to the divergence of $\sum a_nx^n$, because if the series $\sum a_nx^n$ converges, so does $\sum\vert a_nx^n\vert$, which contradicts to our result.  

These are some main facts about the convergence behavior of an arbitrary power series and some properties of its:
- Given a power series $\sum a_nx^n$, precisely one of the following is true:
	- The series converges only for $x=0$.
	- The series is absolutely convergent for all $x$.
	- There exists a positive real number $R$ such that the series is absolutely convergent for $\vert x\vert\<R$ and divergent for $\vert x\vert>R$.
- The positive real number $R$ is called **radius of convergence** of the power series: the series converges absolutely at every point of the open interval $(-R,R)$, and diverges outside the closed interval $[-R,R]$.
- The set of all $x$'s for which a power series converges is called its **interval of convergence**.
- When the series converges only for $x=0$, we define $R=0$; and we define $R=\infty$ when the series converges for all $x$.
- Every power series $\sum a_nx^n$ has a radius of convergence $R$, where $0\leq R\leq\infty$, with the property that the series converges absolutely if $\vert x\vert\<R$ and diverges if $\vert x\vert>R$.  

**Example**  
Find the interval of convergence of the series
\begin{equation}
\sum_{n=0}^{\infty}\dfrac{x^n}{n+1}=1+\dfrac{x}{2}+\dfrac{x^2}{3}+\ldots
\end{equation}

**Solution**  
In order to find the interval of convergence of a series, we begin by identifying its radius of convergence.  

Consider a power series $\sum a_nx^n$. Suppose that this limit exists, and has $\infty$ as an allowed value, we have
\begin{equation}
\lim_{n\to\infty}\dfrac{\vert a_{n+1}x^{n+1}\vert}{a_nx^n}=\lim_{n\to\infty}\left\vert\dfrac{a_{n+1}}{a_n}\right\vert.\vert x\vert=\dfrac{\vert x\vert}{\lim_{n\to\infty}\left\vert\frac{a_n}{a_{n+1}}\right\vert}=L
\end{equation}
By the [ratio test]({% post_url 2021-09-06-infinite-series-of-constants %}#ratio-test), we have $\sum a_nx^n$ converges absolutely if $L<1$ and diverges in case of $L>1$. Or in other words, the series converges absolutely if
\begin{equation}
\vert x\vert<\lim_{n\to\infty}\left\vert\dfrac{a_n}{a_{n+1}}\right\vert,
\end{equation}
or diverges if
\begin{equation}
\vert x\vert>\lim_{n\to\infty}\left\vert\dfrac{a_n}{a_{n+1}}\right\vert
\end{equation}
From the definition of radius of convergence, we can choose the radius of converge of $\sum a_nx^n$ as
\begin{equation}
R=\lim_{n\to\infty}\left\vert\dfrac{a_n}{a_{n+1}}\right\vert
\end{equation}

Back to our problem, for the series $\sum\frac{x^n}{n+1}$, we have its radius of convergence is
\begin{equation}
R=\lim_{n\to\infty}\left\vert\dfrac{a_n}{a_{n+1}}\right\vert=\lim_{n\to\infty}\dfrac{\frac{1}{n+1}}{\frac{1}{n+2}}=\lim_{n\to\infty}\dfrac{n+2}{n+1}=1
\end{equation}
At $x=1$, the series becomes the *harmonic series* $1+\frac{1}{2}+\frac{1}{3}+\ldots$, which diverges; and at $x=-1$, it is the *alternating harmonic series* $1-\frac{1}{2}+\frac{1}{3}-\ldots$, which converges. Hence, the interval of convergence of the series is $[-1,1)$.

## Differentiation and Integration of Power Series
{: #dif-int-power-series}








## References
[1] George F.Simmons. [Calculus With Analytic Geometry - 2nd Edition](https://www.amazon.com/Calculus-Analytic-Geometry-George-Simmons/dp/0070576424)  

[2] MIT 18.01. [Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)  


## Footnotes
