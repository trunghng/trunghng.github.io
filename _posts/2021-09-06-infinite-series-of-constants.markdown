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
- [Series of Nonnegative terms. Comparison tests](#series-nonneg-ct)
	- [Comparison test](#comparison-test)
	- [Limit comparison test](#limit-comparison-test)
- [Integral test. Euler's constant](#int-test-euler-c)
	- [Integral test](#integral-test)
	- [Euler's constant](#euler-c)
- [Ratio test. Root test](#ratio-root)
	- [Ratio test](#ratio-test)
	- [Root test](#root-test)
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
- Geometric series
\begin{equation}
\dfrac{1}{1-x}=\sum_{n=0}^{\infty}x^n=1+x+x^2+x^3+\dots,\hspace{1cm}\vert x\vert<1
\end{equation}
- Exponential function
\begin{equation}
{\rm e}^x=\sum_{n=0}^{\infty}\dfrac{x^n}{n!}=1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+\ldots
\end{equation}
- Sine and cosine formulas
\begin{align}
\sin x&=\sum_{n=0}^{\infty}\dfrac{(-1)^n x^{2n+1}}{(2n+1)!}=x-\dfrac{x^3}{3!}+\dfrac{x^5}{5!}-\dfrac{x^7}{7!}+\ldots \\\\ \cos x&=\sum_{n=0}^{\infty}\dfrac{(-1)^n x^{2n}}{(2n)!}=1-\dfrac{x^2}{2!}+\dfrac{x^4}{4!}-\dfrac{x^6}{6!}+\ldots
\end{align}

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
\sum_{n=1}^{\infty}a_n=a_1+a_2+\ldots+a_n+\ldots\tag{1}\label{1}
\end{equation}
is called an *infinite series*. We begin by establishing the sequence of *partial sums*
\begin{align}
s_1&=a_1 \\\\ s_2&=a_1+a_2 \\\\ &\,\vdots \\\\ s_n&=a_1+a_2+\dots+a_n \\\\ &\,\vdots
\end{align}
The series \eqref{1} is said to be **convergent** if the sequences $\\{s_n\\}$ converges. And if $\lim s_n=s$, then we say that \eqref{1} converges to $s$, or that $s$ is the sum of the series.
\begin{equation}
\sum_{n=1}^{\infty}a_n=s
\end{equation}
If the series does not converge, we say that it **diverges** or is **divergent**, and no sum is assigned to it.

**Examples** (*harmonic series*)  
Let's consider the convergence of *harmonic series*
\begin{equation}
\sum_{n=1}^{\infty}\frac{1}{n}=1+\frac{1}{2}+\frac{1}{3}+\ldots\tag{2}\label{2}
\end{equation}
Let $m$ be a positive integer and choose $n>2^{m+1}$. We have
\begin{align}
s_n&>1+\frac{1}{2}+\frac{1}{3}+\frac{1}{4}+\dots+\frac{1}{2^{m+1}} \\\\ &=\left(1+\frac{1}{2}\right)+\left(\frac{1}{3}+\frac{1}{4}\right)+\left(\frac{1}{5}+\ldots+\frac{1}{8}\right)+\ldots+\left(\frac{1}{2^m+1}+\ldots+\frac{1}{2^{m+1}}\right) \\\\ &>\frac{1}{2}+2.\frac{1}{4}+4.\frac{1}{8}+\ldots+2^m.\frac{1}{2^{m+1}} \\\\ &=(m+1)\frac{1}{2}
\end{align}
This proves that $s_n$ can be made larger than the sum of any number of $\frac{1}{2}$'s and therefore as large as we please, by taking $n$ large enough, so the $\\{s_n\\}$ are unbounded, which leads to that \eqref{2} is a divergent series.
\begin{equation}
\sum_{n=1}^{\infty}\frac{1}{n}=1+\frac{1}{2}+\frac{1}{3}+\ldots=\infty
\end{equation}


The simplest general principle that is useful to study the convergence of a series is the **$\mathbf{n}$-th term test**.

### $\mathbf{n}$-th term test
{: #nth-term-test}
If the series $\\{a_n\\}$ converges, then $a_n\to 0$ as $n\to\infty$; or equivalently, if $\neg(a_n\to0)$ as $n\to\infty$, then the series must necessarily diverge.  

**Proof**  
When $\\{a_n\\}$ converges, as $n\to\infty$ we have
\begin{equation}
a_n=s_n-s_{n-1}\to s-s=0
\end{equation}
This result shows that $a_n\to 0$ is a necessary condition for convergence. However, it is not a sufficient condition; i.e., it does not imply the convergence of the series when $a_n\to 0$ as $n\to\infty$.

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
The easiest infinite series to work with are those whose terms are all nonnegative numbers. The reason, as we saw in the above [section](#conv-div-series), is that if $a_n\geq0$, then the series $\sum a_n$ converges if and only if its sequence $\\{s_n\\}$ of partial sums is bounded (since $s_{n+1}=s_n+a_{n+1}$).  

Thus, in order to establish the convergence of a series of nonnegative terms, it suffices to show that its terms approach zero fast enough, or at least as fast as the terms of a known convergent series of nonnegative terms to keep the partial sums bounded.

### Comparison test
If $0\leq a_n\leq b_n$, then
- $\sum a_n$ converges if $\sum b_n$ converges.
- $\sum b_n$ diverges if $\sum a_n$ diverges.  

**Proof**  
If $s_n, t_n$ respectively are the partial sums of $\sum a_n,\sum b_n$, then
\begin{equation}
0\leq s_n=\sum_{i=1}^{n}a_i\leq\sum_{i=1}^{n}b_i=t_n
\end{equation}
Then if $\\{t_n\\}$ is bounded, then so is $\\{s_n\\}$; and if $\\{s_n\\}$ is unbounded, then so is $\\{t_n\\}$.  

**Example**  
Consider convergence behavior of two series
\begin{equation}
\sum_{n=1}^{\infty}\frac{1}{2^n+1};\hspace{2cm}\sum_{n=1}^{\infty}\frac{1}{\ln n}
\end{equation}
The first series converges, because
\begin{equation}
\frac{1}{2^n+1}<\frac{1}{2^n}
\end{equation}
and $\sum_{n=1}^{\infty}\frac{1}{2^n}=1$, which is a convergent series. At the same time, the second series diverges, since
\begin{equation}
\frac{1}{n}\leq\frac{1}{\ln n}
\end{equation}
and $\sum_{n=1}^{\infty}\frac{1}{n}$ diverges.  

One thing worth remarking is that the condition $0\leq a_n\leq b_n$ for the comparison test need not hold for all $n$, but only for all $n$ from some point on.  

The comparison test is simple, but in some cases where it is difficult to establish the necessary inequality between the n-th terms of the two series. And since limits are often easier to work with than inequalities, we have the following test.

### Limit comparison test
If $\sum a_n, \sum b_n$ are series with positive terms such that
\begin{equation}
\lim_{n\to\infty}\frac{a_n}{b_n}=1\tag{3}\label{3}
\end{equation}
then either both series converge or both series diverge.  

**Proof**  
we observe that \eqref{3} implies that for all sufficient large $n$, we have
\begin{align}
\frac{1}{2}&\leq\frac{a_n}{b_n}\leq 2 \\\\ \text{or}\hspace{1cm}\frac{1}{2}b_n&\leq a_n\leq 2b_n
\end{align}
which leads to the fact that $\sum a_n$ and $\sum b_n$ have the same convergence behavior.  

The condition \eqref{3} can be generalized by
\begin{equation}
\lim_{n\to\infty}\frac{a_n}{b_n}=L,
\end{equation}
where $0\<L<\infty$.  

If a convergent series of nonnegative terms is rearranged in any manner, then the resulting series also converges and has the same sum.

## The Integral test. Euler's constant
{: #int-test-euler-c}
In this section, we will be going through a more detailed class of infinite series with nonnegative terms which is those whose terms form a decreasing sequence of positive numbers.  

We begin by considering a series
\begin{equation}
\sum_{n=1}^{\infty}a_n=a_1+a_2+\ldots+a_n+\ldots
\end{equation}
whose terms are positive and decreasing. Suppose $a_n=f(n)$, as shown is ***Figure 1***.
<figure>
	<img src="/assets/images/2021-09-06/integral-test.png" alt="integral test" width="500px" height="230px" style="display: block; margin-left: auto; margin-right: auto;"/>
	<figcaption style="text-align: center;font-style: italic;"><b>Figure 1</b></figcaption>
</figure>

On the left in this figure we see that the rectangles of areas $a_1,a_2,\dots,a_n$ have a greater combined area than the area under the curve from $x=1$ to $x=n1$, so
\begin{equation}
a_1+a_2+\dots+a_n\geq\int_{1}^{n+1}f(x)\,dx\geq\int_{1}^{n}f(x)\,dx\tag{4}\label{4}
\end{equation}
On the right side of the figure, the rectangles lie under the curve, which makes
\begin{align}
a_2+a_3+\dots+a_n&\leq\int_{1}^{n}f(x)\,dx \\\\ a_1+a_2+\dots+a_n&\leq a_1+\int_{1}^{n}f(x)\,dx\tag{5}\label{5}
\end{align}
Putting \eqref{4} and \eqref{5} together we have
\begin{equation}
\int_{1}^{n}f(x)\,dx\leq a_1+a_2+\dots+a_n\leq a_1+\int_{1}^{n}f(x)\,dx\tag{6}\label{6}
\end{equation}
The result we obtained in \eqref{6} allows us to establish the **integral test**.

### Integral test

If $f(x)$ is a positive decreasing function for $x\geq1$ such that $f(n)=a_n$ for each positive integer $n$, then the series and integral
\begin{equation}
\sum_{n=1}^{\infty}a_n;\hspace{2cm}\int_{1}^{\infty}f(x)\,dx
\end{equation}
converge or diverge together.  

The integral test holds for any interval of the form $x\geq k$, not just for $x\geq 1$.  

**Example** (*Abel's series*)  
Let's consider the convergence behavior of the series
\begin{equation}
\sum_{n=2}^{\infty}\frac{1}{n\ln n}\tag{7}\label{7}
\end{equation}
By the integral test, we have that \eqref{7} diverges, because
\begin{equation}
\sum_{2}^{\infty}\frac{dx}{x\ln x}=\lim_{b\to\infty}\int_{2}^{b}\frac{dx}{x\ln x}=\lim_{b\to\infty}\left(\ln\ln x\Big|\_{2}^{b}\right)=\lim_{b\to\infty}\left(\ln\ln b-\ln\ln 2\right)=\infty
\end{equation}
More generally, if $p>0$, then
\begin{equation}
\sum_{n=2}^{\infty}\frac{1}{n(\ln n)^p}
\end{equation}
converges if $p>1$ and diverges if $0\<p\leq 1$. For if $p\neq 1$, we have
\begin{align}
\int_{2}^{\infty}\frac{dx}{x(\ln x)^p}&=\lim_{b\to\infty}\int_{2}^{b}\frac{dx}{x(\ln x)^p} \\\\ &=\lim_{b\to\infty}\left[\dfrac{(\ln x)^{1-p}}{1-p}\Bigg|\_2^b\right] \\\\ &=\lim_{b\to\infty}\left[\dfrac{(\ln b)^{1-p}-(\ln 2)^{1-p}}{1-p}\right]
\end{align}
exists if and only if $p>1$.

### Euler's constant
{: #euler-c}
From \eqref{6} we have that
\begin{equation}
0\leq a_1+a_2+\ldots+a_n-\int_{1}^{n}f(x)\,dx\leq a_1
\end{equation}
Denoting $F(n)=a_1+a_2+\ldots+a_n-\int_{1}^{n}f(x)\,dx$, the above expression becomes
\begin{equation}
0\leq F(n)\leq a_1
\end{equation}
Moreover, $\\{F(n)\\}$ is a decreasing sequence, because
\begin{align}
F(n)-F(n+1)&=\left[a_1+a_2+\ldots+a_n-\int_{1}^{n}f(x)\,dx\right]-\left[a_1+a_2+\ldots+a_{n+1}-\int_{1}^{n+1}f(x)\,dx\right] \\\\ &=\int_{n}^{n+1}f(x)\,dx-a_{n+1}\geq 0
\end{align}
where the last step can be seen by observing the right side of ***Figure 1***.  

Since any decreasing sequence of nonnegative numbers converges, we have that
\begin{equation}
L=\lim_{n\to\infty}F(n)=\lim_{n\to\infty}\left[a_1+a_2+\ldots+a_n-\int_{1}^{n}f(x)\,dx\right]\tag{8}\label{8}
\end{equation}
exists and satisfies the inequalities $0\leq L\leq a_1$.  

Let $a_n=\frac{1}{n}$ and $f(x)=\frac{1}{x}$, the last quantity in \eqref{8} becomes
\begin{equation}
\lim_{n\to\infty}\left(1+\dfrac{1}{2}+\ldots+\dfrac{1}{n}-\ln n\right)\tag{9}\label{9}
\end{equation}
since
\begin{equation}
\int_{1}^{n}\dfrac{dx}{x}=\ln x\Big|\_1^n=\ln n
\end{equation}
The value of the limit \eqref{9} is called **Euler's constant** (denoted as $\gamma$).
\begin{equation}
\gamma=\lim_{n\to\infty}\left(1+\dfrac{1}{2}+\ldots+\dfrac{1}{n}-\ln n\right)
\end{equation}

## Ratio test. Root test
{: #ratio-root}

### Ratio test




## References
[1] George F.Simmons. [Calculus With Analytic Geometry - 2nd Edition](https://www.amazon.com/Calculus-Analytic-Geometry-George-Simmons/dp/0070576424)  

[2] MIT 18.01. [Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)

## Footnotes
[^1]: We will be going through power series in more detailed in another post.
