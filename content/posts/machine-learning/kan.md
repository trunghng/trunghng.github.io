---
title: "Kolmogorov–Arnold Networks"
date: 2024-06-10T08:35:54+07:00
tags: [machine-learning, neural-network]
draft: true
math: true
eqn-number: true
hideSummary: true
---

### B-splines, B-spline curves
An order $p+1$ B-spline is a collection of piecewise polynomial functions $B_{i,p}(t)$ of degree $p$ in a variable $t$. The values of $t$ where the pieces of polynomial meet are known as knots, denoted $t_0, t_1,\ldots,t_m$ and sorted into non-decreasing order.

For a given sequences of knots, the associated B-spline basis functions, $B_{i,p}$ are defined as
\begin{equation}
B_{i,0}(t)=\begin{cases}1&\text{if }t_i\leq t\leq t_{i+1} \\\\ 0&\text{otherwise}\end{cases}
\end{equation}
for $p=0$ and
\begin{equation}
B_{i,p}(t)=\frac{t-t_i}{t_{i+p}-t_i}B_{i,p-1}(t)+\frac{t_{i+p+1}-t}{t_{i+p+1}-t_{i+1}}B_{i+1,p-1}(t)
\end{equation}

## Kolmogorov-Arnold representation theorem
Kolmogorov-Arnold representation theorem states that if $f$ is a multivariate continuous function on a bounded domain, then $f$ can be written as a finite composition of continuous functions of a single variable and the binary operation of addition. Specifically, for a smooth $f:[0,1]^n\to\mathbb{R}$
\begin{equation}
f(\mathbf{x})=f(x_1,\ldots,x_n)=\sum_{q=1}^{2n+1}\Phi_q\left(\sum_{p=1}^{n}\phi_{q,p}(x_p)\right),\label{eq:kart.1}
\end{equation}
where $\phi_{q,p}:[0,1]\to\mathbb{R}$ and $\Phi_q:\mathbb{R}\to\mathbb{R}$.

## Kolmogorov-Arnold Network
We have that the Kolmogorov-Arnold representation \eqref{eq:kart.1} can be rewritten in matrix form
\begin{equation}
f(\mathbf{x})=\mathbf{\Phi}\_\text{out}\circ\mathbf{\Phi}\_\text{in}\circ\mathbf{x},
\end{equation}
where
\begin{equation}
\mathbf{\Phi}\_\text{in}=\left[\begin{matrix}\phi_{1,1}(\cdot)&\ldots&\phi_{1,n}(\cdot) \\\\ \vdots&\ddots&\vdots \\\\ \phi_{2n+1,1}(\cdot)&\ldots&\phi_{2n+1,n}(\cdot)\end{matrix}\right]
\end{equation}
and
\begin{equation}
\mathbf{\Phi}\_\text{out}=\big(\Phi_1(\cdot)\dots\Phi_{2n+1}(\cdot)\big)
\end{equation}
Especially, these $\mathbf{\Phi}\_\text{in}$ and $\mathbf{\Phi}\_\text{out}$ are special cases of the following matrix, which will also be defined as a Kolmogorov-Arnold layer
\begin{equation}
\mathbf{\Phi}=\left[\begin{matrix}\phi_{1,1}(\cdot)&\ldots&\phi_{1,n_\text{in}}(\cdot) \\\\ \vdots&\ddots&\vdots \\\\ \phi_{n_\text{out},1}(\cdot)&\ldots&\phi_{n_\text{out},n_\text{in}}(\cdot)\end{matrix}\right],
\end{equation}
where $\mathbf{\Phi}\_\text{in}$ corresponds to $n_\text{in}=n,n_\text{out}=2n+1$ and $\mathbf{\Phi}\_\text{out}$ corresponds to $n_\text{in}=2n+1,n_\text{out}=1$.

We then can stack Kolmogorov-Arnold layers together to form a Kolmogorov-Arnold Network. With a $L$-layer network, in which $l$-th layer $\mathbf{\Phi}^{(l)}$ has shape $(n_{l+1},n_l)$, it is then defined to be
\begin{equation}
\text{KAN}(\mathbf{x})=\mathbf{\Phi}^{(L-1)}\circ\mathbf{\Phi}^{(L-2)}\circ\ldots\circ\mathbf{\Phi}^{(0)}\circ\mathbf{x}
\end{equation}

## Preferences
[1] Ziming Liu, Yixuan Wang, et. al., [KAN: Kolmogorov–Arnold Networks](https://arxiv.org/abs/2404.19756). arXiv preprint, arXiv:2404.19756, 2024.

[2] Tom Lyche, Carla Manni, Hendrik Speleers, [B-Splines and Spline Approximation](https://www.mat.uniroma2.it/~speleers/cime2017/material/notes_lyche.pdf).

## Footnotes
