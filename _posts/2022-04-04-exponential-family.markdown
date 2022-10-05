---
layout: post
title:  "The exponential family"
date:   2022-05-04 14:00:00 +0700
categories: mathematics probability-statistics
tags: mathematics probability-statistics exponential-family
description: Note on exponential family
comments: true
eqn-number: true
---
> A note on the exponential family.

<!-- excerpt-end -->

- [The exponential family](#exp-fam)
- [Examples](#examples)
	- [Bernoulli distribution](#bern)
	- [Binomial distribution](#bin)
	- [Multinomial distribution](#mult)
	- [Poisson distribution](#pois)
	- [Gaussian distribution](#gauss)
	- [Multivariate Normal distribution](#mvn)
- [Convexity](#cvxt)
- [Moments of sufficient statistic](mmt-suff-stat)
	- [Means, variances](#mean-var)
	- [Moment generating functions](#mgf)
	- [Cumulant generating functions](#cgf)
	- [Cumulants](#cumulants)
- [Sufficiency](#sufficiency)
- [Maximum likelihood estimates](#mle)
- [Conjugate priors](conj-prior)
- [References](#references)
- [Footnotes](#footnotes)

## The exponential family
{: #exp-fam}
The **exponential family** of distributions is defined as family of distributions of form
\begin{equation}
p(x;\eta)=h(x)\exp\Big[\eta^\text{T}T(x)-A(\eta)\Big],\label{eq:ef.1}
\end{equation}
where
- $\eta$ is known as the **natural parameter**, or **canonical parameter**,
- $T(X)$ is referred to as a **sufficient statistic**,
- $A(\eta)$ is called the **cumulant function**, which can be view as the logarithm of a normalization factor since integrating \eqref{eq:ef.1} w.r.t the measure $\nu$ gives us
\begin{equation}
A(\eta)=\log\int h(x)\exp\left(\eta^\text{T}T(x)\right)\nu(dx),\label{eq:ef.2}
\end{equation}
This also implies that $A(\eta)$ will be determined once we have specified $\nu,T(x)$ and $h(x)$.

The set of parameters $\eta$ for which the integral in \eqref{eq:ef.2} is finite is known as the **natural parameter space**
\begin{equation}
N=\left\\{\eta:\int h(x)\exp\left(\eta^\text{T}T(x)\right)\nu(dx)<\infty\right\\}
\end{equation}
which explains why $\eta$ is also referred as **natural parameter**. If $N$ is an non-empty open set, the exponential families are said to be **regular**.

An exponential family is known as **minimal** if there are no linear constraints among the components of $\eta$ nor are there linear constraints among the components of $T(x)$.

## Examples
{: #examples}
Each particular choice of $\nu$, $T$ and $h$ defines a family (or set) of distributions that is parameterized by $\eta$. As we vary $\eta$, we then get different distributions within this family.

### Bernoulli distribution
{: #bern}
The probability mass function (i.e., the density function w.r.t counting measure) of a Bernoulli random variable $X$, denoted as $X\sim\text{Bern}(\pi)$, is given by
\begin{align}
p(x;\pi)&=\pi^x(1-\pi)^{1-x} \\\\ &=\exp\big[x\log\pi+(1-x)\log(1-\pi)\big] \\\\ &=\exp\left[\log\left(\frac{\pi}{1-\pi}\right)x+\log(1-\pi)\right],
\end{align}
which can be written in the form of an exponential family distribution \eqref{eq:ef.1} with
\begin{align}
\eta&=\frac{\pi}{1-\pi} \\\\ T(x)&=x \\\\ A(\eta)&=-\log(1-\pi)=\log(1+e^{\eta}) \\\\ h(x)&=1
\end{align}
Notice that the relationship between $\eta$ and $\pi$ is invertible since
\begin{equation}
\pi=\frac{1}{1+e^{-\eta}},
\end{equation}
which is the **sigmoid function**.

### Binomial distribution
{: #bin}
The probability mass function of a Binomial random variable $X$, denoted as $X\sim\text{Bin}(N,\pi)$, is defined as
\begin{align}
p(x;N,\pi)&=\left(\begin{matrix}N \\\\ x\end{matrix}\right)\pi^{x}(1-\pi)^{1-x} \\\\ &=\left(\begin{matrix}N \\\\ x\end{matrix}\right)\exp\big[x\log\pi+(1-x)\log(1-\pi)\big] \\\\ &=\left(\begin{matrix}N \\\\ x\end{matrix}\right)\exp\left[\log\left(\frac{\pi}{1-\pi}\right)x+\log(1-\pi)\right],
\end{align}
which is in form of an exponential family distribution \eqref{eq:ef.1} with
\begin{align}
\eta&=\frac{\pi}{1-\pi} \\\\ T(x)&=x \\\\ A(\eta)&=-\log(1-\pi)=\log(1+e^{\eta}) \\\\ h(x)&=\left(\begin{matrix}N \\\\ x\end{matrix}\right)
\end{align}
Similar to the Bernoulli case, we also have the invertible relationship between $\eta$ and $\pi$ as
\begin{equation}
\pi=\frac{1}{1+e^{-\eta}}
\end{equation}

### Poisson distribution
{: #pois}
The probability mass function of a Poisson random variable $X$, denoted as $X\sim\text{Pois}(\lambda)$, is given as
\begin{align}
p(x;\lambda)&=\frac{\lambda^x e^{-\lambda}}{x!} \\\\ &=\frac{1}{x!}\exp\left(x\log\lambda-\lambda\right),
\end{align}
which is also able to be written as an exponential family distribution \eqref{eq:ef.1} with
\begin{align}
\eta&=\log\lambda \\\\ T(x)&=x \\\\ A(\eta)&=\lambda=e^{\eta} \\\\ h(x)&=\frac{1}{x!}
\end{align}
Analogy to Bernoulli distribution, we also have that
\begin{equation}
\lambda=e^{\eta}
\end{equation}

### Gaussian distribution
{: #gauss}
The (univariate) Gaussian density of a random variable $X$, denoted as $X\sim\mathcal{N}(\mu,\sigma^2)$, is given by
\begin{align}
p(x;\mu,\sigma^2)&=\frac{1}{\sqrt{2\pi}\sigma}\exp\left[-\frac{(x-\mu)^2}{2\sigma^2}\right] \\\\ &=\frac{1}{\sqrt{2\pi}}\exp\left[\frac{\mu}{\sigma^2}x-\frac{1}{2\sigma^2}x^2-\frac{1}{2\sigma^2}\mu^2-\log\sigma\right],
\end{align}
which allows us to write it as an instance of the exponential family with
\begin{align}
\eta&=\left[\begin{matrix}\mu/\sigma^2 \\\\ -1/2\sigma^2\end{matrix}\right] \\\\ T(x)&=\left[\begin{matrix}x\\\\ x^2\end{matrix}\right] \\\\ A(\eta)&=\frac{\mu^2}{2\sigma^2}+\log\sigma=-\frac{\eta_1^2}{4\eta_2}-\frac{1}{2}\log(-2\eta_2) \\\\ h(x)&=\frac{1}{\sqrt{2\pi}}
\end{align}

### Multinomial distribution
{: #mult}
Let $\mathbf{X}=(X_1,\ldots,X_K)$ be the collection of $K$ random variable in which $X_k$ denotes the number of times the $k$-th event occurs in a set of $N$ independent trials. And let $\mathbf{\pi}=(\pi_1,\ldots,\pi_K)$ with $\sum_{k=1}^{K}\pi_k=1$ correspondingly represents the probability of occurring of each event within each trials.

Then $\mathbf{X}$ is said to have Multinomial distribution, denoted as $\mathbf{X}\sim\text{Mult}\_K(N,\boldsymbol{\pi})$, if its probability mass function is given as with $\sum_{k=1}^{K}x_k=1$
\begin{align}
p(\mathbf{x};\boldsymbol{\pi},N,K)&=\frac{N!}{x_1!x_2!\ldots x_K!}\pi_1^{x_1}\pi_2^{x_2}\ldots\pi_n^{x_n} \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left(\sum_{k=1}^{K}x_k\log\pi_k\right)\label{eq:m.1}
\end{align}
It is noticeable that the above equation is not minimal, since there exists a linear constraint between the components of $T(\mathbf{x})$, which is
\begin{equation}
\sum_{k=1}^{K}x_k=1
\end{equation}
In order to remove this constraint, we substitute $1-\sum_{k=1}^{K-1}x_k$ to $x_K$ , which lets \eqref{eq:m.1} be written by
\begin{align}
\hspace{-0.5cm}p(\mathbf{x};\boldsymbol{\pi},N,K)&=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left(\sum_{k=1}^{K}x_k\log\pi_k\right) \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left[\sum_{k=1}^{K-1}x_k\log\pi_k+\left(1-\sum_{k=1}^{K-1}x_k\right)\log\left(1-\sum_{k=1}^{K-1}\pi_k\right)\right] \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left[\sum_{i=1}^{K-1}\log\left(\frac{\pi_i}{1-\sum_{k=1}^{K-1}\pi_k}\right)x_i+\log\left(1-\sum_{k=1}^{K-1}\pi_k\right)\right]\label{eq:m.2}
\end{align}
With this representation, and also for convenience, for $i=1,\ldots,K$ we continue by letting
\begin{equation}
\eta_i=\log\left(\frac{\pi_i}{1-\sum_{k=1}^{K-1}\pi_k}\right)=\log\left(\frac{\pi_i}{\pi_K}\right)\label{eq:m.3}
\end{equation}
Take the exponential of both sides and summing over $K$, we have
\begin{equation}
\sum_{i=1}^{K}e^{\eta_i}=\frac{\sum_{i=1}^{K}\pi_i}{\pi_K}=\frac{1}{\pi_K}\label{eq:m.4}
\end{equation}
From this result, we have that the Multinomial distribution \eqref{eq:m.2} is therefore also a member of the exponential family with
\begin{align}
\eta&=\left[\begin{matrix}\log\left(\pi_1/\pi_K\right) \\\\ \vdots \\\\ \log\left(\pi_K/\pi_K\right)\end{matrix}\right] \\\\ T(\mathbf{x})&=\left[\begin{matrix}x_1,\ldots,x_K\end{matrix}\right]^\text{T} \\\\ A(\eta)&=-\log\left(1-\sum_{i=1}^{K-1}\pi_i\right)=-\log(\pi_K)=\log\left(\sum_{k=1}^{K}e^{\eta_k}\right) \\\\ h(\mathbf{x})&=\frac{N!}{x_1!x_2!\ldots x_K!}
\end{align}
Additionally, substituting the result \eqref{eq:m.4} into \eqref{eq:m.3} gives us for $i=1,\ldots,K$
\begin{equation}
\eta_i=\log\left(\pi_i\sum_{k=1}^{K}e^{\eta_k}\right),
\end{equation}
or we can express $\boldsymbol{\pi}$ in terms of $\eta$ by
\begin{equation}
\pi_i=\frac{e^{\eta_i}}{\sum_{k=1}^{K}e^{\eta_k}},
\end{equation}
which is the **softmax function**.

### Multivariate Normal distribution
{: #mvn}

## Convexity
{: #cvxt}
**Theorem**  
The natural space $N$ is a convex set and the cumulant function $A(\eta)$ is a convex function. If the family is minimal, then $A(\eta)$ is strictly convex.

**Proof**  
Let $\eta_1,\eta_2\in N$, thus from \eqref{eq:ef.2}, we have that
\begin{align}
\exp\big(A(\eta_1)\big)&=A_1, \\\\ \exp\big(A(\eta_2)\big)&=A_2
\end{align}
where $A_1,A_2$ are finite. 

To prove that $N$ is convex, we need to show that for any $\eta=\lambda\eta_1+(1-\lambda)\eta_2$ for $0\lt\lambda\lt 1$, we also have $\eta\in N$. From \eqref{eq:ef.2}, and by **Hölder's inequality**[^1], we have
\begin{align}
\exp\big(A(\eta)\big)&=\int h(x)\exp\big(\eta^\text{T}T(x)\big)\nu(dx) \\\\ &=\int h(x)\exp\Big[\big(\lambda\eta_1+(1-\lambda)\eta_2\big)^\text{T}T(x)\Big]\nu(dx) \\\\ &=\int \Big[h(x)\exp\big(\eta_1^\text{T}T(x)\big)\Big]^{\lambda}\Big[h(x)\exp\big(\eta_2^\text{T}T(x)\big)\Big]^{1-\lambda}\nu(dx) \\\\ &\leq\Bigg[\int h(x)\exp\big(\eta_1^\text{T}T(x)\big)\nu(dx)\Bigg]^\lambda\Bigg[\int h(x)\exp\big(\eta_2^\text{T}T(x)\big)\nu(dx)\Bigg]^{1-\lambda} \\\\ &=\Big[\exp\big(A(\eta_1)\big)\Big]^\lambda\Big[\exp\big(A(\eta_2)\big)\Big]^{1-\lambda} \\\\ &=A_1^\lambda A_2^{1-\lambda},\label{eq:c.1}
\end{align}
which proves that $A(\eta)$ is finite, or $\eta\in N$.

Moreover, taking logarithm of both sides of \eqref{eq:c.1} gives us
\begin{equation}
\lambda A(\eta_1)+(1-\lambda)A(\eta_2)\geq A(\eta)=A\big(\lambda\eta_1+(1-\lambda)\eta_2\big),
\end{equation}
which also claims the convexity of $A(\eta)$.

By Hölder's inequality, the equality in \eqref{eq:c.1} holds when
\begin{equation}
\Big[h(x)\exp\big(\eta_2^\text{T}T(x)\big)\Big]^{1-\lambda}=c\Big[h(x)\exp\big(\eta_1^\text{T}T(x)\big)\Big]^{\lambda(1/\lambda-1)}
\end{equation}
or
\begin{equation}
\exp\big(\eta_2^\text{T}T(x)\big)=c\exp\big(\eta_1^\text{T}T(x)\big),
\end{equation}
and therefore
\begin{equation}
(\eta_2-\eta_1)^\text{T}T(x)=\log c,
\end{equation}
which is not minimal since $\eta_1,\eta_2$ are taken arbitrarily.

## Moments of sufficient statistic
{: #mmt-suff-stat}
In this section, we will see how the moments of the sufficient statistic $T(X)$ can be calculated from the cumulant function $A(\eta)$. In more specifically, the first moment (mean) and the second central moment (variance) of $T(X)$ are exactly the first and the second **cumulants**.

### Means, variances
{: #mean-var}
Let us first consider the first derivative of the cumulant function $A(\eta)$. By the **dominated convergence theorem**, we have
\begin{align}
\frac{\partial A(\eta)}{\partial\eta^\text{T}}&=\frac{\partial}{\partial\eta^\text{T}}\log\int\exp\big(\eta^\text{T}T(x)\big)h(x)\nu(dx) \\\\ &=\frac{\int T(x)\exp\big(\eta^\text{T}(x)\big)h(x)\nu(dx)}{\int\exp\big(\eta^\text{T}T(x)\big)h(x)\nu(dx)} \\\\ &=\int T(x)\exp\big(\eta^\text{T}T(x)-A(\eta)\big)h(x)\nu(dx)\label{eq:mv.1} \\\\ &=\int T(x)p(x;\eta)\nu(dx) \\\\ &=\mathbb{E}[T(X)],
\end{align}
which is the mean of the sufficient statistic $T(x)$.

Moreover, taking the second derivative of cumulant function by continuing with the result \eqref{eq:mv.1}, we have
\begin{align}
\frac{\partial^2 A(\eta)}{\partial\eta\partial\eta^\text{T}}&=\frac{\partial}{\partial\eta^\text{T}}\int T(x)\exp\big(\eta^\text{T}T(x)-A(\eta)\big)h(x)\nu(dx) \\\\ &=\int T(x)\left(T(x)-\frac{\partial}{\partial\eta^\text{T}}A(\eta)\right)^\text{T}\exp\big(\eta^\text{T}T(x)-A(\eta)\big)h(x)\nu(dx) \\\\ &=\int T(x)\big(T(x)-E(T(X))\big)^\text{T}\exp\big(\eta^\text{T}T(x)-A(\eta)\big)h(x)\nu(dx) \\\\ &=\mathbb{E}\left[T(X)T(X)^\text{T}\right]-\mathbb{E}[T(X)]\mathbb{E}[T(X)]^\text{T} \\\\ &=\text{Var}[T(X)],
\end{align}
which is the variance (or the covariance matrix in the multivariate case) of the sufficient statistic $T(X)$.

### Moment generating functions
{: #mgf}
The **moment generating function** (or **MGF**) of a random variable $X$, denoted as $M(t)$, is given by
\begin{equation}
M(t)=\mathbb{E}(e^{t^\text{T}X}),
\end{equation}
for all values of $t$ for which the expectation exists.

The MGF of the sufficient statistic $T(X)$ then can be computed as
\begin{align}
M_{T(X)}(t)&=\mathbb{E}(e^{t^\text{T}T(X)}) \\\\ &=\int \exp\big((\eta+t)^\text{T}T(x)-A(\eta)\big)h(x)\nu(dx) \\\\ &=\exp\big(A(\eta+t)-A(\eta)\big)\label{eq:mgf.1}
\end{align}

### Cumulant generating functions
{: #cgf}
The **cumulant generating function** (or **CGF**) of a random variable $X$, denoted by $K(t)$, is given as
\begin{equation}
K(t)=\log M(t)=\log\mathbb{E}(e^{t^\text{T}X}),
\end{equation}
for all values of $t$ for which the expectation exists.

From the MGF of $T(X)$ in \eqref{eq:mgf.1}, the CGF of the sufficient statistic $T(X)$ therefore can be calculated by
\begin{equation}
K_{T(X)}(t)=\log M_{T(X)}(t)=A(\eta+t)-A(\eta)
\end{equation}

### Cumulants
{: #cumulants}
The $k$-th **cumulant** of a random variable $X$ is defined to be the $k$-th derivative of $K_{X}(t)$ at $0$, i.e.,
\begin{equation}
c_k=K^{(k)}(0)
\end{equation}

Thus, the mean of $T(X)$ is exactly the first cumulant, while the variance is the second cumulant of $T(X)$.

## Sufficiency
{: #sufficiency}

## Maximum likelihood estimates
{: #mle}
Consider an i.i.d data set $\mathcal{D}=\\{x_1,\ldots,x_N\\}$, the likelihood function is then given by
\begin{align}
L(\eta)=p(\mathbf{X}\vert\eta)&=\prod_{n=1}^{N}p(x_n\vert\eta) \\\\ &=\prod_{n=1}^{N}h(x_n)\exp\big[\eta^\text{T}T(x_n)-A(\eta)\big] \\\\ &=\left(\prod_{n=1}^{N}h(x_n)\right)\exp\left[\eta^\text{T}\left(\sum_{n=1}^{N}T(x_n)\right)-N A(\eta)\right]\label{eq:mle.1}
\end{align}
Taking the logarithm of both sides gives us the log likelihood as
\begin{equation}
\ell(\eta)=\log L(\eta)=\log\left(\prod_{n=1}^{N}h(x_n)\right)+\eta^\text{T}\left(\sum_{n=1}^{N}T(x_n)\right)-N A(\eta)
\end{equation}
Consider the gradient of the log likelihood w.r.t $\eta$, we have
\begin{align}
\nabla_\eta\ell(\eta)&=\nabla_\eta\left[\log\left(\prod_{n=1}^{N}h(x_n)\right)+\eta^\text{T}\left(\sum_{n=1}^{N}T(x_n)\right)-N A(\eta)\right] \\\\ &=\sum_{n=1}^{N}T(x_n)-N\nabla_\eta A(\eta)
\end{align}
Setting the gradient to zero, we have the value of $\eta$ that maximizes the likelihood, or maximum likelihood estimation for $\eta$, denoted as $\eta_\text{ML}$ satisfies
\begin{equation}
\nabla_{\eta}A(\eta_\text{ML})=\frac{1}{N}\sum_{n=1}^{N}T(x_n)
\end{equation}

## Conjugate priors
{: #conj-prior}
Given a probability distribution $p(x\vert\eta)$, its prior $p(\eta)$ is said to be **conjugate** to the likelihood function if the prior and the posterior has the same functional form. The prior distribution in this case is also referred as **conjugate prior**.

For any member of the exponential family, there exists a conjugate prior that can be written in form
\begin{equation}
p(\eta\vert\mathcal{X},\theta)=f(\mathcal{X},\theta)\exp(\eta^\text{T}\mathcal{X}-\theta A(\eta)),\label{eq:cp.1}
\end{equation}
where $\theta>0$ and $\mathcal{X}$ are hyperparameters.

By Bayes' rule, and with the likelihood function as given in \eqref{eq:mle.1}, the posterior distribution can be computed as
\begin{align}
&\hspace{0.7cm}p(\eta\vert\mathbf{X},\mathcal{X},\theta) \\\\ &\propto p(\eta\vert\mathcal{X},\theta)p(\mathbf{X}\vert\eta) \\\\ &=f(\mathcal{X},\theta)\exp\big(\eta^\text{T}\mathcal{X}-\theta A(\eta)\big)\left(\prod_{n=1}^{N}h(x_n)\right)\exp\left[\eta^\text{T}\left(\sum_{n=1}^{N}T(x_n)\right)-N A(\eta)\right] \\\\ &\propto\exp\left[\eta^\text{T}\left(\mathcal{X}+\sum_{n=1}^{N}T(x_n)\right)-(\theta+N)A(\eta)\right],
\end{align}
which is in the same form as \eqref{eq:cp.1} and therefore claims the conjugacy.

## References
{: #references}
[1] M. Jordan. [The Exponential Family: Basics](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf). 2009.

[2] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573).

[3] Weisstein, Eric W. [Hölder's Inequalities](https://mathworld.wolfram.com/HoeldersInequalities.html) From MathWorld--A Wolfram Web Resource.

## Footnotes
{: #footnotes}

[^1]: Let $p,q>1$ such that
	\begin{equation\*}
	\frac{1}{p}+\frac{1}{q}=1
	\end{equation\*}
	The **Hölder's inequality** for integrals states that
	\begin{equation\*}
	\int_a^b\vert f(x)g(x)\vert\,dx\leq\left(\int_a^b\vert f(x)\vert\,dx\right)^{1/p}\left(\int_a^b\vert g(x)\vert\,dx\right)^{1/q}
	\end{equation\*}
	The equality holds with
	\begin{equation\*}
	\vert g(x)\vert=c\vert f(x)\vert^{p-1}
	\end{equation\*}