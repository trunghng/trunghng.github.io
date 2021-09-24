---
layout: post
title:  "Power Series"
date:   2021-09-21 15:40:00 +0700
categories: mathematics calculus
tags: mathematics calculus series power-series taylor-series random-stuffs
description: power series
comments: true
---
> Recall that in the previous post, [Infinite Series of Constants]({% post_url 2021-09-06-infinite-series-of-constants %}), we mentioned a type of series called **power series** a lot. In the content of this post, we will be diving deeper into details of that series.

<!-- excerpt-end -->
- [Power Series](#power-series)
- [The Interval of Convergence](#int-conv)
	- [Example](#eg1)
- [Differentiation and Integration of Power Series](#dif-int-power-series)
	- [Differentiation of Power Series](#dif-power-series)
	- [Integration of Power Series](#int-power-series)
	- [Example](#eg2)
- [Taylor Series, Taylor's Formula](#taylor-series-formula)
	- [Taylor Series](#taylor-series)
	- [Taylor's Formula](#taylors-formula)
- [References](#references)
- [Footnotes](#footnotes)

## Power Series
A **power series** is a series of the form
\begin{equation}
\sum_{n=0}^{\infty}a_nx^n=a_0+a_1x+a_2x^2+\ldots+a_nx^n+\ldots,
\end{equation}
where the coefficient $a_n$ are constants and $x$ is a variable.  

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

### Example
{: #eg1} 
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

It is easily seen that the sum of the series $\sum_{n=0}^{\infty}a_nx^n$  is a function of $x$ since the sum depends only on $x$ for any value of $x$. Hence, we can denote this as
\begin{equation}
f(x)=\sum_{n=0}^{\infty}a_nx^n=a_0+a_1x+a_2x^2+\ldots+a_nx^n+\ldots\tag{2}\label{2}
\end{equation}
This relation between the series and the function is also expressed by saying that $\sum a_nx^n$ is a **power series expansion** of $f(x)$.  

These are some crucial facts about that relation.
- (i) The function $f(x)$ defined by \eqref{2} is continuous on the open interval $(-R,R)$.  
- (ii) The function $f(x)$ is differentiable on $(-R,R)$, and its derivative is given by the formula
\begin{equation}
f'(x)=a_1+2a_2x+3a_3x^2+\ldots+na_nx^{n-1}+\ldots\tag{3}\label{3}
\end{equation}
- (iii) If $x$ is any point in $(-R,R)$, then
\begin{equation}
\int_{0}^{x}f(t)\,dt=a_0x+\dfrac{1}{2}a_1x^2+\dfrac{1}{3}a_2x^3+\ldots+\dfrac{1}{n+1}a_nx^{n+1}+\ldots\tag{4}\label{4}
\end{equation}

**Remark**  
We have that series \eqref{3} and \eqref{4} converge on the interval $(-R,R)$.  

**Proof**  
1. We begin by proving the convergence on $(-R,R)$ of \eqref{3}.  
Let $x$ be a point in the interval $(-R,R)$ and choose $\epsilon>0$ so that $\vert x\vert+\epsilon\<R$. Since $\vert x\vert+\epsilon$ is in the interval, $\sum\vert a_n\left(\vert x\vert+\epsilon\right)^n\vert$ converges.  
We continue by proving the inequality
\begin{equation}
\vert nx^{n-1}\vert\leq\left(\vert x\vert+\epsilon\right)^n\hspace{1cm}\forall n\geq n_0,
\end{equation}
where $\epsilon>0$, $n_0$ is a positive integer.  
We have
\begin{align}
\lim_{n\to\infty}n^{1/n}&=\lim_{n\to\infty} \\\\ &=\lim_{n\to\infty}\exp\left(\frac{\ln n}{n}\right) \\\\ &=\exp\left(\lim_{n\to\infty}\frac{\ln n}{n}\right) \\\\ &={\rm e}^0=1,
\end{align}
where in the fourth step, we use the *L’Hospital theorem*[^1]. Therefore, as $n\to\infty$
\begin{equation}
n^{1/n}\vert x\vert^{1-1/n}\to\vert x\vert
\end{equation}
Then for all sufficiently large $n$'s
\begin{align}
n^{1/n}\vert x\vert^{1-1/n}&\leq\vert x\vert+\epsilon \\\\ \vert nx^{n-1}\vert&\leq\left(\vert x\vert+\epsilon\right)^n
\end{align}
This implies that
\begin{equation}
\vert na_nx^{n-1}\vert\leq\vert a_n\left(\vert x\vert+\epsilon\right)^n\vert
\end{equation}
By the [comparison test]({% post_url 2021-09-06-infinite-series-of-constants %}#comparison-test), we have that the series $\sum\vert na_nx^{n-1}\vert$ converges, and so does $\sum na_nx^{n-1}$.  

2. Since $\sum\vert a_nx^n\vert$ converges and
\begin{equation}
\left\vert\dfrac{a_nx^n}{n+1}\right\vert\leq\vert a_nx^n\vert,
\end{equation}
the [comparison test]({% post_url 2021-09-06-infinite-series-of-constants %}#comparison-test) implies that $\sum\left\vert\frac{a_nx^n}{n+1}\right\vert$ converges, and therefore
\begin{equation}
x\sum\frac{a_nx^n}{n+1}=\sum\frac{1}{n+1}a_nx^{n+1}
\end{equation}
also converges.





### Differentiation of Power Series
{: #dif-power-series}

If we instead apply (ii) to the function $f'(x)$ in \eqref{3}, then it follows that $f'(x)$ is also differentiable. Doing the exact same process to $f'\'(x)$, we also have that $f'\'(x)$ is differentiable, and so on. Hence, the original $f(x)$ has derivatives of all orders, as expressed in the following statement:  

*In the interior of its interval of convergence, a power series defines an finitely differentiable function whose derivatives can be calculated by differentiating the series term by term*.
\begin{equation}
\dfrac{d}{dx}\left(\sum a_nx^n\right)=\sum\dfrac{d}{dx}(a_nx^n)
\end{equation}

### Integration of Power Series
{: #int-power-series}

Similarly, from (iii), the term-by-term integration of power series can be emphasized by writing \eqref{4} as
\begin{equation}
\int\left(\sum a_nx^n\right)\,dx=\sum\left(\int a_nx^n\,dx\right)
\end{equation}

### Example
{: #eg2}

Find a power series expansion of ${\rm e}^x$.  

**Solution**  
Since ${\rm e}^x$ is the only function that equals its own derivatives[^2] and has the value $1$ at $x=0$. To construct a power series equal to its own derivative, we use the fact that when such a series is differentiated, the degree of each term drops by $1$. We therefore want each term to be the derivative of the one that follows it.  

Starting with $1$ as the constant term, the next should be $x$, then $\frac{1}{2}x^2$, then $\frac{1}{2.3}x^3$, and so on. This produces the series
\begin{equation}
1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+\ldots+\dfrac{x^n}{n!}+\ldots,\tag{5}\label{5}
\end{equation}
which converges for all $x$ because
\begin{equation}
R=\lim_{n\to\infty}\dfrac{\frac{1}{n!}}{\frac{1}{(n+1)!}}=\lim_{n\to\infty}(n+1)=\infty
\end{equation}
We have constructed the series \eqref{5} so that its sum is unchanged by differentiated and has the value $1$ at $x=0$. Therefore, for all $x$,
\begin{equation}
{\rm e}^x=1+x+\dfrac{x^2}{2!}+\dfrac{x^3}{3!}+\ldots+\dfrac{x^n}{n!}+\ldots
\end{equation}

## Taylor Series, Taylor's Formula
{: #taylor-series-formula}

### Taylor Series
Assume that $f(x)$ is the sum of a power series with positive radius of convergence
\begin{equation}
f(x)=\sum_{n=0}^{\infty}a_nx^n=a_0+a_1x+a_2x^2+\ldots,\hspace{1cm}R>0\tag{6}\label{6}
\end{equation}
By the results obtained from previous section, differentiating \eqref{6} term by term we have
\begin{align}
f^{(1)}(x)&=a_1+2a_2x+3a_3x^2+\ldots \\\\ f^{(2)}(x)&=1.2a_2+2.3a_3x+3.4a_4x^2+\ldots \\\\ f^{(3)}(x)&=1.2.3a_3+2.3.4a_4x+3.4.5a_5x^2+\ldots
\end{align}
and in general,
\begin{equation}
f^{(n)}(x)=n!a_n+A(x),\tag{7}\label{7}
\end{equation}
where $A(x)$ contains $x$ as a factor.  

Since these series expansions of the derivatives are valid on the open interval $(-R,R)$, putting $x=0$ in \eqref{7} we obtain
\begin{equation}
f^{(n)}(0)=n!a_n
\end{equation}
so
\begin{equation}
a_n=\dfrac{f^{(n)}(0)}{n!}
\end{equation}
Putting this result in \eqref{6}, our series becomes
\begin{equation}
f(x)=f(0)+f^{(1)}(0)x+\dfrac{f^{(2)}(0)}{2!}x^2+\ldots+\dfrac{f^{(n)}(0)}{n!}x^n+\ldots\tag{8}\label{8}
\end{equation}
This power series is called **Taylor series** of $f(x)$ [at $x=0$], which is named after the person who introduced it, Brook Taylor.  

If we use the convention that $0!=1$, then \eqref{8} can be written as
\begin{equation}
f(x)=\sum_{n=0}^{\infty}\dfrac{f^{(n)}(0)}{n!}x^n
\end{equation}
The numbers $a_n=\frac{f^{(n)}(0)}{n!}$ are called the **Taylor coefficients** of $f(x)$.  

**Remark**  
Given a function $f(x)$ that is infinitely differentiable in some interval containing the point $x=0$, we have already examined the possibility of expanding this function as a power series in $x$. More generally, if $f(x)$ is infinitely differentiable in some interval containing the point $x=a$, is there any possibility for the power series expansion of $f(x)$ in $x-a$ instead of $x$?  
\begin{equation}
f(x)=\sum_{n=0}^{\infty}a_n(x-a)^n=a_0+a_1(x-a)+a_2(x-a)^2+\ldots
\end{equation}
Let $w=x-a$, and $g(w)=f(x)$, we have that $g^{(n)}(0)=f^{(n)}(a)$. Thus, the Taylor series of $f(x)$ in power of $x-a$ (or at $x=a$) is
\begin{align}
f(x)&=\sum_{n=0}^{\infty}\dfrac{f^{(n)}(a)}{n!}(x-a)^n \\\\ &=f(a)+f^{(1)}(a)(x-a)+\dfrac{f^{(2)}(a)}{2!}(x-a)^2+\ldots+\dfrac{f^{(n)}(a)}{n!}(x-a)^n+\ldots\tag{9}\label{9}
\end{align}

### Taylor's Formula
{: #taylors-formula}
If we break off the Taylor series on the right side of \eqref{8} at the term containing $x^n$ and define the *remainder* $R_n(x)$ by the equation
\begin{equation}
f(x)=f(0)+f^{(1)}(0)x+\dfrac{f^{(2)}(0)}{2!}x^2+\ldots+\dfrac{f^{(n)}(0)}{n!}x^n+R_n(x),\tag{10}\label{10}
\end{equation}
then the Taylor series on the right side of \eqref{8} converges to the function $f(x)$ as $n$ tends to infinity precisely when
\begin{equation}
\lim_{n\to\infty}R_n(x)=0
\end{equation}
Since $R_n(x)$ contains $x^{n+1}$ as a factor, we can define a function $S_n(x)$ by writing
\begin{equation}
R_n(x)=S_n(x)x^{n+1}
\end{equation}
for $x\neq 0$. Next, we keep $x$ fixed and define a function $F(t)$ for $0\leq t\leq x$ (or $x\leq t\leq 0$) by writing
\begin{multline}
F(t)=f(x)-f(t)-f^{(1)}(t)(x-t)-\dfrac{f^{(2)}(t)}{2!}(x-t)^2-\ldots \\\\ -\dfrac{f^{(n)}(t)}{n!}(x-t)^n-S_n(x)(x-t)^{n+1}
\end{multline}
It is easily seen that $F(x)=0$. Also, from equation \eqref{10}, we have that $F(0)=0$. Then by the *Mean Value Theorem*[^3], $F'(c)=0$ for some constant $c$ between $0$ and $x$.  

By differentiating $F(t)$ w.r.t $t$, and evaluate it at $t=c$, we have
\begin{equation}
F'(c)=-\dfrac{f^{(n+1)}(c)}{n!}(x-c)^n+S_n(x)(n+1)(x-c)^n=0
\end{equation}
so
\begin{equation}
S_n(x)=\dfrac{f^{(n+1)}(c)}{(n+1)!}
\end{equation}
and
\begin{equation}
R_n(x)=S_n(x)x^{n+1}=\dfrac{f^{(n+1)}(c)}{(n+1)!}x^{n+1}
\end{equation}
which makes \eqref{10} become
\begin{equation}
f(x)=f(0)+f^{(1)}(0)x+\dfrac{f^{(2)}(0)}{2!}x^2+\ldots+\dfrac{f^{(n)}(0)}{n!}x^n+\dfrac{f^{(n+1)}(c)}{(n+1)!}x^{n+1},
\end{equation}
where $c$ is some number between $0$ and $x$. This equation is called **Taylor's formula with derivative remainder**.  

Moreover, with this formula we can rewrite \eqref{9} as
\begin{multline}
f(x)=f(a)+f^{(1)}(a)(x-a)+\dfrac{f^{(2)}(a)}{2!}(x-a)^2+\ldots \\\\ +\dfrac{f^{(n)}(a)}{n!}(x-a)^n+\dfrac{f^{(n+1)}(a)}{(n+1)!}(x-a)^{n+1},\tag{11}\label{11}
\end{multline}
where $c$ is some number between $a$ and $x$.  

The polynomial part of \eqref{11}
\begin{equation}
\sum_{j=0}^{n}\dfrac{f^{(j)}(a)}{j!}(x-a)^j=f(a)+f^{(1)}(a)(x-a)+\dfrac{f^{(2)}(a)}{2!}(x-a)^2+\ldots+\dfrac{f^{(n)}(a)}{n!}(x-a)^n
\end{equation}
is called the **nth-degree Taylor polynomial at** $x=a$.  

On the other hand, the remainder part of \eqref{11}
\begin{equation}
R_n(x)=\dfrac{f^{(n+1)}(a)}{(n+1)!}(x-a)^{n+1}
\end{equation}
is often called **Lagrange's remainder formula**.

## References
[1] George F.Simmons. [Calculus With Analytic Geometry - 2nd Edition](https://www.amazon.com/Calculus-Analytic-Geometry-George-Simmons/dp/0070576424)  

[2] Marian M. [A Concrete Approach to Classical Analysis](https://www.springer.com/gp/book/9780387789323)  

[3] MIT 18.01. [Single Variable Calculus](https://ocw.mit.edu/courses/mathematics/18-01-single-variable-calculus-fall-2006/)  


## Footnotes
[^1]: **Theorem** (*L’Hospital*)  
	*Assume $f$ and $g$ are real and differentiable on $]a,b[$ and $g'(x)\neq 0$ for all $x\in]a,b[$, where $-\infty\leq a<b\leq\infty$. Suppose as $x\to a$,
	\begin{equation}
	\dfrac{f'(x)}{g'(x)}\to A\,(\in[-\infty,\infty])
	\end{equation}
	If as $x\to a$, $f(x)\to 0$ and $g(x)\to 0$ or if $g(x)\to+\infty$ as $x\to a$, then
	\begin{equation}
	\dfrac{f(x)}{g(x)}\to A
	\end{equation}
	as $x\to a$.*  

[^2]: **Proof**  
	Consider the function $f(x)=a^x$.  
	Using the definition of the derivative, we have
	\begin{align}
	\dfrac{d}{dx}f(x)&=\lim_{h\to0}\dfrac{f(x+h)-f(x)}{h} \\\\ &=\lim_{h\to0}\dfrac{a^{x+h}-a^x}{h} \\\\ &=a^x\lim_{h\to0}\dfrac{a^h-1}{h}
	\end{align}
	Therefore,
	\begin{equation}
	\lim_{h\to0}\dfrac{a^h-1}{h}=1
	\end{equation}
	then, let $n=\frac{1}{h}$, we have
	\begin{equation}
	a=\lim_{h\to0}\left(1+\dfrac{1}{h}\right)^{1/h}=\lim_{n\to\infty}\left(1+\dfrac{1}{n}\right)^n={\rm e}
	\end{equation}
	Thus, $f(x)=a^x={\rm e}^x$. Every function $y=c{\rm e}^x$ also satisfies the differential equation $\frac{dy}{dx}=y$, because
	\begin{equation}
	\dfrac{dy}{dx}=\dfrac{d}{dx}c{\rm e}^x=c\dfrac{d}{dx}{\rm e}^x=c{\rm e}^x=y
	\end{equation}  
	The rest of our proof is to prove that these are only functions that are unchanged by differentiation.  
	To prove this, suppose $f(x)$ is any function with that property. By the quotient rule,
	\begin{equation}
	\dfrac{d}{dx}\dfrac{f(x)}{e^x}=\dfrac{f'(x)e^x-e^xf(x)}{e^{2x}}=\dfrac{e^xf(x)-e^xf(x)}{e^{2x}}=0
	\end{equation}
	which implies that
	\begin{equation}
	\dfrac{f(x)}{e^x}=c,
	\end{equation}
	for some constant $c$, and so $f(x)=ce^x$.  

[^3]: **Theorem** (*Mean Value Theorem*)  
	*If a function $f(x)$ is continuous on the closed interval $[a,b]$ and differentiable in the open interval $(a,b)$, then there exists at least one number $c$ between $a$ and $b$ with the property that*
	\begin{equation}
	f'(c)=\frac{f(b)-f(a)}{b-a}
	\end{equation}
