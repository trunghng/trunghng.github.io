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
- [Maximum likelihood estimates](#max-llh)
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
which explains why $\eta$ is also referred as **natural parameter**. If $N$ is an unempty open set, the exponential families are said to be **regular**.

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
Let $\mathbf{X}=(X_1,\ldots,X_K)$ be the collection of $K$ random variable in which $X_k$ denotes the number of times the $k$-th event occurs in a set of $N$ independent trials. And let $\mathbf{\pi}=(\pi_1,\ldots,\pi_K)$ with $\sum_{k=1}^{K}\pi_k=1$ correspondingly represents the probability of occuring of each event within each trials.

Then $\mathbf{X}$ is said to have Multinomial distribution, denoted as $\mathbf{X}\sim\text{Mult}\_K(N,\boldsymbol{\pi})$, if its probability mass function is given as with $\sum_{k=1}^{K}x_k=1$
\begin{align}
p(\mathbf{x};\boldsymbol{\pi},N,K)&=\frac{N!}{x_1!x_2!\ldots x_K!}\pi_1^{x_1}\pi_2^{x_2}\ldots\pi_n^{x_n} \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left(\sum_{k=1}^{K}x_k\log\pi_k\right)\label{eq:m.1}
\end{align}
It is noticable that the above equation is not mininal, since there exists a linear constraint between the components of $T(\mathbf{x})$, which is
\begin{equation}
\sum_{k=1}^{K}x_k=1
\end{equation}
In order to remove this contraint, we substitute $1-\sum_{k=1}^{K-1}x_k$ to $x_K$ , which lets \eqref{eq:m.1} be written by
\begin{align}
\hspace{-0.5cm}p(\mathbf{x};\boldsymbol{\pi},N,K)&=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left(\sum_{k=1}^{K}x_k\log\pi_k\right) \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left[\sum_{k=1}^{K-1}x_k\log\pi_k+\left(1-\sum_{k=1}^{K-1}x_k\right)\log\left(1-\sum_{k=1}^{K-1}\pi_k\right)\right] \\\\ &=\frac{N!}{x_1!x_2!\ldots x_K!}\exp\left[\sum_{k=1}^{K-1}\log\left(\frac{\pi_k}{1-\sum_{k=1}^{K-1}\pi_k}\right)x_k+\log\left(1-\sum_{k=1}^{K-1}\pi_k\right)\right]
\end{align}


### Multivariate Normal distribution
{: #mvn}


## References
{: #references}
[1] M. Jordan. [The Exponential Family: Basics](https://people.eecs.berkeley.edu/~jordan/courses/260-spring10/other-readings/chapter8.pdf). 2009.

[2] Joseph K. Blitzstein & Jessica Hwang. [Introduction to Probability](https://www.amazon.com/Introduction-Probability-Chapman-Statistical-Science/dp/1466575573).

## Footnotes
{: #footnotes}