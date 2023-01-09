---
title: "Convex sets, convex functions"
date: 2021-12-02 13:03:00 +0700
tags: [mathematics, convex-optimization]
math: true
eqn-number: true
draft: true
---
> Notes on convex sets, convex functions.
<!--more-->

## Convex sets{#cvx-sets}

### Affine & convex sets{#aff-cvx-sets}

#### Affine sets{#aff-sets}
A set $C\subset\mathbb{R}^n$ is **affine** if the line through any two distinct points in $C$ lies in $C$, i.e. for any $x_1,x_2\in C$ and for any $\theta\in\mathbb{R}$ we have
\begin{equation}
\theta x_1+(1-\theta)x_2\in C
\end{equation}
A point of the form $\theta_1 x_1+\ldots+\theta_k x_k$, where $\theta_1+\ldots+\theta_k=1$ is known as an **affine combination** of the points $x_1,\ldots,x_k$.

Hence, if $C$ is an affine set, and $x_1,\ldots,x_k\in C$, and $\theta_1+\ldots+\theta_k=1$, then the point
\begin{equation}
\theta_1 x_1+\ldots+\theta_k x_k\in C,
\end{equation}
or in other words, an affine set contains every affine combination of its points.

If $C$ is an affine set and $x_0\in C$, then the set
\begin{equation}
V=C-x_0\\{x-x_0:x\in C\\}
\end{equation}
is a subspace.

The set of all affine combinations of points in some set $C\subset\mathbb{R}^n$ is referred as **affine hull** of $C$, denoted $\text{aff}\hspace{0.1cm}C$:
\begin{equation}
\text{aff}\hspace{0.1cm}C=\\{\theta_1 x_1+\ldots+\theta_k x_k:x_1,\ldots,x_k\in C;\theta_1+\ldots+\theta_k=1\\}
\end{equation}
The affine hull is the *smallest* affine set containing $C$.

#### Affine dimension, relative interior{#aff-dim-rel-int}
The **affine dimension** of a set $C$ is defined as the dimension of $\text{aff}\hspace{0.1cm}C$.

If the affine dimension of $C\subset\mathbb{R}^n$ is less than $n$, then the set lies in $\text{aff}\hspace{0.1cm}C\neq\mathbb{R}^n$. The **relative interior** of the set $C$, denoted as $\text{relint}\hspace{0.1cm}C$, is defined as its interior relative to $\text{aff}\hspace{0.1cm}C$:
\begin{equation}
\text{relint}\hspace{0.1cm}C=\\{x\in C:B(x,r)\cap\text{aff}\hspace{0.1cm}C\in C\text{ for some }r>0\\},
\end{equation}
where $B(x,r)$ is the ball centered at $x$ with radius $r$ in the norm $\Vert\cdot\Vert$ (here $\Vert\cdot\Vert$ could be any norm; all norms define the same relative interior).

The **relative boundary** of $C$ is defined as $\overline{C}\backslash\text{relint}\hspace{0.1cm}C$, where $\overline{C}$ is the closure of $C$.

#### Convex sets{#cvx-sets-def}
A set $C$ is **convex** if the line segment between any points in $C$ also lies in $C$, i.e. for any $x_1,x_2\in C$ and for any $0\leq\theta\leq 1$, we have
\begin{equation}
\theta x_1+(1-\theta)x_2\in C
\end{equation}
It is then easily seen that every affine sets is also convex.

Analogy to affine sets, we also refer a point of the form $\theta_1 x_1+\ldots+\theta_k x_k$, where $\theta_1+\ldots+\theta_k=1$ and $\theta_i\geq 0,\forall i=1,\ldots,k$, a **convex combination** of the points $x_1,\ldots,x_k$. And a set is convex iff it contains every convex combination of its points.

The **convex hull** of $C$, denoted $\text{conv}\hspace{0.1cm}C$, is defined as the set of all convex combinations of points in $C$:
\begin{equation}
\text{conv}\hspace{0.1cm}C=\\{\theta_1 x_1+\ldots+\theta_k x_k:x_1,\ldots,x_k\in C;\theta_1+\ldots+\theta_k=1;\theta_1,\ldots,\theta_k\geq 0\\}
\end{equation}
Thus, $\text{conv}\hspace{0.1cm}C$ is convex and is the smallest convex set containing $C$.

We can generalize the definition of convex combination into: let $x_1,x_2\ldots\in C$ where $C\subset\mathbb{R}^n$ and let $\\{\theta_n\\}\_{n=1,2,\ldots}$ be a countable sequence such that
\begin{equation}
\sum_{i=1}^{\infty}\theta_i=1;\hspace{2cm}\theta_i\geq 0,\hspace{0.5cm}\forall i=1,2,\ldots
\end{equation}
Then the series
\begin{equation}
\sum_{i=1}^{\infty}\theta_i x_i\in C,
\end{equation}
if it converges.

More generally, suppose $p:\mathbb{R}^n\to\mathbb{R}$ satisfies $p(x)\geq 0$ forall $x\in C$ and $\int_C p(x)\hspace{0.1cm}dx=1$, where $C\subset\mathbb{R}^n$ is a convex set. Then the integral
\begin{equation}
\int_C p(x)x\hspace{0.1cm}dx\in C
\end{equation}
if it exists.

In the most general form, suppose $C\subset\mathbb{R}^n$ is convex and $x$ is a random vector with $x\in C$ with probability one. Then we also have that
\begin{equation}
\mathbb{E}x\in C
\end{equation}

#### Cones
A set $C$ is called a **cone**, or **nonnegative homogeneous**, if for every $x\in C$ and for any $\theta\geq 0$, we also have $\theta x\in C$.

A **convex cone** $C$ is both convex and a cone, i.e. for any $x_1,x_2\in C$ and for any $\theta_1,\theta_2\geq 0$, we have
\begin{equation}
\theta_1 x_1+\theta_2 x_2\in C
\end{equation}
since by definition of a cone, we can add a normalization factor $\alpha$ into the point above
\begin{equation}
\alpha\theta_1 x_1+\alpha\theta_2 x_2
\end{equation}
such that $\alpha\theta_1+\alpha\theta_2=1$ (in this particular case, $\alpha=1/(\theta_1+\theta_2)$).

A point of the form $\theta_1 x_1+\ldots+\theta_k x_k$ with $\theta_1,\ldots,\theta_k\geq 0$ is called a **conic combination**. It is easily seen that a set $C$ is a convex cone iff it contains all conic combinations of its points.
Like convex and affine combinations, we can generalize the definition of conic combination into infinite series and integrals.

We define the **conic hull** of a set $C$ as the set of all conic combinations of elements in $C$
\begin{equation}
\\{\theta_1 x_1+\ldots+\theta_k x_k:x_i\in C;\theta_i\geq 0,\forall i=1,\ldots,k\\}
\end{equation}
Also, the conic hull of $C$ is the smallest convex cone containing $C$.

### Examples{#cvx-sets-eg}

#### Hyperplanes, halfspaces{#hyperplane-halfspaces}
A **hyperplane** $P$ is a set of form
\begin{equation}
P=\\{x\in\mathbb{R}^n:a^\text{T}x=b\\},
\end{equation}
where $a\in\mathbb{R}^n$, $a\neq 0$ and $b\in\mathbb{R}$. We have that $P$ is convex.

To prove this, for $x_1,x_2\in P$, and for any $0\leq\theta\leq 1$, we have
\begin{equation}
a^\text{T}\big(\theta x_1+(1-\theta)x_2\big)=\theta a^\text{T}x_1+(1-\theta)a^\text{T}x_2=\theta b+(1-\theta)b=b
\end{equation} 

A hyperplane separates $\mathbb{R}^n$ into two **halfspaces**. A (closed) halfspace is a set of of the form
\begin{equation}
\\{x\in\mathbb{R}^n:a^\text{T}x\leq b\\},
\end{equation}
where $a\in\mathbb{R}^n$, $a\neq 0$ and $b\in\mathbb{R}$. It is also easily seen that halfspaces are also convex.

#### Balls, ellipsoids, norm cones{#balls-ellips-cones}

##### Balls
A (closed) **ball** in $\mathbb{R}^n$ centered at $x_c$ and with radius $r$ and with $\Vert\cdot\Vert$ is any norm in $\mathbb{R}^n$
\begin{equation}
B(x_c,r)=\\{x\in\mathbb{R}^n:\Vert x-x_c\Vert\leq r\\}
\end{equation}
is a convex set.

To see this, for any $x_1,x_2\in B(x_c,r)$ and for any $0\leq\theta\leq 1$, by triangle inequality of norm, we have
\begin{align}
\Vert\theta x_1+(1-\theta)x_2-x_c\Vert&=\Vert\theta(x_1-x_c)+(1-\theta)(x_2-x_c)\Vert \\\\ &\leq\theta\Vert x_1-x_c\Vert+(1-\theta)\Vert x_2-x_c\Vert \\\\ &\leq\theta r+(1-\theta)r \\\\ &=r
\end{align}

##### Ellipsoids{#ellips}
An **ellipsoid** $\mathcal{E}$ in $\mathbb{R}^n$ centered at $x_c\in\mathbb{R}^n$ is defined as
\begin{equation}
\mathcal{E}=\\{x:(x-x_c)^\text{T}P^{-1}(x-x_c)\leq 1\\},
\end{equation}
where $P\in\mathbb{R}^{n\times n}$ is symmetric and positive definite. The matrix $P$ determines how far $\mathcal{E}$ extends in every direction from $x_c$; the lengths of the semi-axes of $\mathcal{E}$ are $\sqrt{\lambda_i}$, where $\lambda_i$ for $i=1,\ldots,n$ are the eigenvalues of $P$. A ball of radius $r$ is an ellipsoid with
\begin{equation}
P=r^2 I
\end{equation}
We then have $\mathcal{E}$ is convex.

To prove this claim, as usual, for $x_1,x_2\in\mathcal{E}$ and for $0\leq\theta\leq 1$, we have
\begin{align}
&\hspace{0.7cm}\big(\theta x_1+(1-\theta)x_2-x_c\big)^\text{T}P^{-1}\big(\theta x_1+(1-\theta)x_2-x_c\big) \\\\ &=\big(\theta x_1-\theta x_c+(1-\theta)x_2-(1-\theta)x_c\big)^\text{T}P^{-1}\big(\theta x_1-\theta x_c+(1-\theta)x_2-(1-\theta)x_c\big) \\\\ &=(a+b)^\text{T}P^{-1}(a+b) \\\\ &=a^\text{T}P^{-1}a+b^\text{T}P^{-1}b+2a^\text{T}P^{-1}b \\\\ &\leq\theta^2+(1-\theta)^2+2\theta(1-\theta) \\\\ &=1
\end{align}
where in the second step, we have let
\begin{equation}
a=\theta x_1-\theta x_c,\hspace{2cm}b=(1-\theta)x_2-(1-\theta)x_c,
\end{equation}
which implies that
\begin{equation}
a^\text{T}P^{-1}a\leq 1,\hspace{2cm}b^\text{T}P^{-1}b\leq 1
\end{equation}
and thus
\begin{equation}
a^\text{T}P^{-1/2}\leq 1,\hspace{2cm}b^\text{T}P^{-1/2}\leq 1
\end{equation}

##### Norm cones
A **norm cone** $C$ associated with the norm $\Vert\cdot\Vert$ is defined as
\begin{equation}
C=\\{(x,t):\Vert x\Vert\leq t\\}\subset\mathbb{R}^{n+1}
\end{equation}
is also convex

#### Polyhedra
A **polyhedron** $\mathcal{P}$ is defined as
\begin{equation}
\mathcal{P}=\\{x:a_i^\text{T}\leq b_i,i=1,\ldots,m;c_j^\text{T}=d_j,j=1,\ldots,p\\}
\end{equation}
Then $\mathcal{P}$ can be seen as the intersection of a finite number of halfspaces and hyperplanes. Another representation of $\mathcal{P}$ is
\begin{equation}
\mathcal{P}=\\{x:Ax\preceq b,Cx=d\\},
\end{equation}
where
\begin{equation}
A=\left[\begin{matrix}a_1^\text{T} \\\\ \vdots \\\\ a_m^\text{T}\end{matrix}\right],\hspace{2cm}C=\left[\begin{matrix}c_1^\text{T} \\\\ \vdots \\\\ c_p^\text{T}\end{matrix}\right]
\end{equation}
And thus, we also have that $\mathcal{P}$ is convex, which can be proved easily since $Ax$ and $Cx$ are both linear functions.

##### Nonnegative orthant{#non-neg-orthant}
The **nonnegative orthant** in $\mathbb{R}^n$, denoted $\mathbb{R}\_+^{n}$, is the set of points with nonnegative components, i.e.
\begin{equation}
\mathbb{R}\_+^n=\\{x\in\mathbb{R}^n:x\succeq 0\\}
\end{equation}
We have that $\mathbb{R}\_+^n$ is both a polyhedron and a cone, or a **polyhedral cone**, and hence is also convex.

##### Simplex
Suppose the $v_0,\ldots,v_k\in\mathbb{R}^n$ are **affinely independent**, i.e. $v_1-v_0,\ldots,v_k-v_0$ are linearly independent. The **simplex** determined by them is given as
\begin{equation}
C=\text{conv}\\{v_0,\ldots,v_k\\}=\\{\theta_0 v_0+\ldots+\theta_k v_k:\theta\succeq 0,\mathbf{1}^\text{T}\theta=1\\}
\end{equation}
As an instance of polyhedra, $C$ is thus convex.

#### Positive semi-definite cone{#psd-cone}
Let $\mathbb{S}^n$ denote the set of symmetric $n\times n$ matrices
\begin{equation}
\mathbb{S}^n=\\{X\in\mathbb{R}^{n\times n}:X=X^\text{T}\\},
\end{equation}
and let $\mathbb{S}\_+^n$ represent the set of symmetric positive semi-definite matrices
\begin{equation}
\mathbb{S}\_+^n=\\{X\in\mathbb{S}^n:X\succeq 0\\},
\end{equation}
and finally, let us assign the set of symmetric positive definite matrices to $\mathbb{S}\_{\+\+}^n$
\begin{equation}
\mathbb{S}\_{\+\+}^n=\\{X\in\mathbb{S}^n:X\succ 0\\}
\end{equation}
We have that $\mathbb{S}\_+^n$ is a convex cone, since for any matrices $A_1,A_2\in\mathbb{S}\_+^n$, for any $\theta_1,\theta_2\geq 0$ and for any $x\in\mathbb{R}^n$, we have
\begin{equation}
x^\text{T}(\theta_1 A_1+\theta_2 A_2)x=\theta_1 x^\text{T}A_1 x+\theta_2 x^\text{T}A_2 x\geq 0
\end{equation}
The same argument can be applied to prove that $\mathbb{S}\_{\+\+}^n$ or even the set of symmetric negative definite matrices and the set of symmetric negative semi-definite matrices are convex.

### Operations that preserve convexity{#operations-sets}

#### Intersection{#intersect}
Let $S_1,S_2$ be convex sets and let $x_1,x_2$ are two points containing in both sets, thus $x_1,x_2\in S_1\cap S_2$.

Since $x_1,x_2\in S_1$ which is convex, for $0\leq\theta\leq 1$, we have the point
\begin{equation}
\theta x_1+(1-\theta)x_2\in S_1
\end{equation}
Analogously, we also have
\begin{equation}
\theta x_1+(1-\theta)x_2\in S_2,
\end{equation}
which implies that
\begin{equation}
\theta x_1+(1-\theta)x_2\in S_1\cap S_2
\end{equation}
Or in other words, $S_1\cap S_2$ is also convex.

By induction, we can extend this property to: if $S_\alpha$ is convex for every $\alpha\in\mathcal{A}$, then their intersection
\begin{equation}
\bigcap_{\alpha\in\mathcal{A}}S_\alpha
\end{equation}
is also convex.

#### Affine functions{#aff-funcs}
A function $f:\mathbb{R}^n\to\mathbb{R}^m$ is **affine** if it is a sum of linear function and a constant, i.e. it can be written as
\begin{equation}
f(x)=Ax+b,
\end{equation}
where $A\in\mathbb{R}^{m\times n}$ and $b\in\mathbb{R}^m$.

Let $S\subset\mathbb{R}^n$ be a convex set and let $f:\mathbb{R}^n\to\mathbb{R}^m$ be an affine function. Then the image of $S$ under $f$
\begin{equation}
f(S)=\\{f(x):x\in S\\}
\end{equation}
is convex.

Analogously, the inverse image of $S$ under an affine function $g:\mathbb{R}^k\to\mathbb{R}^n$
\begin{equation}
g^{-1}(S)=\\{x:g(x)\in S\\}
\end{equation}
is convex.

The **projection** of a convex set $S\subset\mathbb{R}^m\times\mathbb{R}^n$ onto some of its coordinates
\begin{equation}
T=\\{x_1\in\mathbb{R}^m:(x_1,x_2)\in S,\text{ for some }x_2\in\mathbb{R}^n\\}
\end{equation}
is convex.

If $S_1,S_2$ are convex then so is their sum
\begin{equation}
S_1+S_2=\\{x_1+x_2:x_1\in S_1,x_2\in S_2\\}
\end{equation}
This is due to its reverse image under the linear function $f(x_1,x_2)=x_1+x_2$, which is the **Cartesian product**
\begin{equation}
S_1\times S_2=\\{(x_1,x_2):x_1\in S_1,x_2\in S_2\\}
\end{equation}
is convex.

#### Linear-fractional, perspective functions{#lin-frac-persp-funcs}

##### Perspective functions{#persp-funcs}
The **perspective function** $P:\mathbb{R}^{n+1}\to\mathbb{R}^n$, with domain $\text{dom}\hspace{0.1cm}P=\mathbb{R}^n\times\mathbb{R}\_{\+\+}$ is defined as
\begin{equation}
P(z,t)=\frac{z}{t}
\end{equation}
Suppose that $x=(\tilde{x},x_{n+1}),y=(\tilde{y},y_{n+1})\in\mathbb{R}^{n+1}$ with $x_{n+1},y_{n+1}\gt 0$. Then for $0\leq\theta\leq 1$, we have
\begin{equation}
P(\theta x+(1-\theta)y)=\frac{\theta\tilde{x}+(1-\theta)\tilde{y}}{\theta x_{1+1}+(1-\theta)y_{n+1}}=\mu P(x)+(1-\mu)P(y),
\end{equation}
where
\begin{equation}
\mu=\frac{\theta x_{n+1}}{\theta x_{n+1}+(1-\theta)y_{n+1}}\in[0,1],
\end{equation}
which implies that
\begin{equation}
P([x,y])=[P(x),P(y)]\label{eq:pf.1}
\end{equation}
Let $C$ be convex with $C\subset\text{dom}\hspace{0.1cm}P$, and let $x,y\in C$. By \eqref{eq:pf.1}, we have that the line segment $[P(x),P(y)]$ is the image of the line segment $[x,y]$ under $P$, $P([x,y])$, and so lies in $P(C)$, which also claims the convexity of $P(C)$.

The inverse image of a convex set under the perspective function is also convex: if $C\subset\mathbb{R}^n$ is convex, then
\begin{equation}
P^{-1}(C)=\\{(x,t)\in\mathbb{R}^{n+1}:x/t\in C,t>0\\}
\end{equation}
is convex.

To prove this, for any $(x_1,t_1),(x_2,t_2)\in P^{-1}(C)$ and for any $0\leq t\leq 1$, by the result \eqref{eq:pf.1}, we have
\begin{equation}
P^{-1}\big(\theta(x_1,t_1)+(1-\theta)(x_2,t_2)\big)=\frac{\theta x_1+(1-\theta x_2)}{\theta t_1+(1-\theta)t_2}=\mu\frac{x_1}{t_1}+(1-\mu)\frac{x_2}{t_2},
\end{equation}
where
\begin{equation}
\mu=\frac{\theta t_1}{\theta t_1+(1-\theta)t_2}\in[0,1]
\end{equation}

##### Linear-fractional functions{#lin-frac-funcs}
We define the **linear-fractional function** to be the composite function of a perspective function with an affine function. Specifically, let $g:\mathbb{R}^n\to\mathbb{R}^{m+1}$ be affine
\begin{equation}
g(x)=\left[\begin{matrix}A \\\\ c^\text{T}\end{matrix}\right]x+\left[\begin{matrix}b \\\\ d\end{matrix}\right],
\end{equation}
where $A\in\mathbb{R}^{m\times n},b\in\mathbb{R}^m,c\in\mathbb{R}^n$ and $d\in\mathbb{R}$. The function $f:\mathbb{R}^n\to\mathbb{R}^m$ given by
\begin{equation}
f(x)=(P\circ g)(x)=\frac{Ax+b}{c^\text{T}x+d},
\end{equation}
for $\text{dom}\hspace{0.1cm}f\\{x:c^\text{T}x+d>0\\}$, is called a **linear-fractional function**.

It is convenient to represent a linear-fractional function as a matrix
\begin{equation}
Q=\left[\begin{matrix}A&b \\\\ c^\text{T}&d\end{matrix}\right]\in\mathbb{R}^{(m+1)\times(n+1)},
\end{equation}
which lets
\begin{equation}
Q\left[\begin{matrix}x \\\\ 1\end{matrix}\right]=\left[\begin{matrix}A&b \\\\ c^\text{T}&d\end{matrix}\right]\left[\begin{matrix}x \\\\ 1\end{matrix}\right]=\left[\begin{matrix}Ax+b \\\\ c^\text{T}x+d\end{matrix}\right]
\end{equation}

## Convex functions{#cvx-funcs}
A function $f:\mathbb{R}^n\to\mathbb{R}$ is called **convex** if $\text{dom}\hspace{0.1cm}f$ is a convex set and if for all $x,y\in\text{dom}\hspace{0.1cm}f$ and for any $0\leq\theta\leq 1$, we have
\begin{equation}
f\big(\theta x+(1-\theta)y\big)\leq\theta f(x)+(1-\theta)f(y)\label{eq:cf.1}
\end{equation}
Intuitively, we can think of the above inequality as the line segment between $(x,f(x))$ and $(y,f(y))$ lies above the graph of $f$.

We call $f$ a **strictly convex** function if strict inequality hold in \eqref{eq:cf.1} for every $x\neq y$ and $0\lt\theta\lt 1$. And $f$ is referred as **concave** if $-f$ is convex, or is known as **strictly concave** if $-f$ is strictly convex.

A function is convex iff it is convex when restricted to any line that intersects its domain. In other words, $f$ is convex iff for all $x\in\text{dom}\hspace{0.1cm}f$ and for all $v$, the function
\begin{equation}
g(t)=f(x+tv)
\end{equation}
is convex on its domain, $\text{dom}\hspace{0.1cm}g=\\{t:x+tv\in\text{dom}\hspace{0.1cm}f\\}$.

All linear and affine functions are either convex or concave.

### Properties{#props}

#### First-order conditions{#st-order-conds}
Let $f$ be differentible, i.e. $\nabla f$ exists at each point in $\text{dom}\hspace{0.1cm}f$, which is open. Then $f$ is convex iff $\text{dom}\hspace{0.1cm}f$ is convex and
\begin{equation}
f(y)\geq f(x)+\nabla f(x)^\text{T}(y-x)
\end{equation}
holds for all $x,y\in\text{dom}\hspace{0.1cm}f$.

Similarly, we can also have that $f$ is strictly convex iff $\text{dom}\hspace{0.1cm}f$ is convex and for $x,y\in\text{dom}\hspace{0.1cm}f$ such that $x\neq y$, we have
\begin{equation}
f(y)>f(x)+\nabla f(x)^\text{T}(y-x)
\end{equation}
And hence, $f$ is concave iff $\text{dom}\hspace{0.1cm}f$ is convex and for all $x,y\in\text{dom}\hspace{0.1cm}f$, we have
\begin{equation}
f(y)\leq f(x)+\nabla f(x)^\text{T}(y-x)
\end{equation}

**Proof**  
We first consider the case that $n=1$, i.e. $f:\mathbb{R}\to\mathbb{R}$ is convex iff for all $x,y\in$, we have
\begin{equation}
f(y)\geq f(x)+f'(x)(y-x)\label{eq:soc.1}
\end{equation}
Suppose that $f$ is convex, thus $\text{dom}\hspace{0.1cm}f$ is convex.

Let $x,y$ be two points in $\text{dom}\hspace{0.1cm}f$. We therefore have that for any $0\lt\theta\leq 1$, $(1-\theta)x+\theta y\in\text{dom}\hspace{0.1cm}f$ and
\begin{equation}
f\big((1-\theta)x+\theta y\big)\leq(1-\theta)f(x)+\theta f(y),
\end{equation}
which give us
\begin{equation}
f(y)\geq f(x)+\frac{f\big(x+\theta(y-x)\big)-f(x)}{\theta}
\end{equation}
Let $t\to 0$, we obtain
\begin{equation}
f(y)\geq f(x)+f'(x)(y-x)
\end{equation}
Given $f$ such that \eqref{eq:soc.1} satisfies for all $x,y$ in $\text{dom}\hspace{0.1cm}f$ and $\text{dom}\hspace{0.1cm}f$ is convex.

Choose any $x\neq y$ and let $z=\theta x+(1-\theta)y$, for some $0\leq\theta\leq 1$, we then have $z\in\text{dom}\hspace{0.1cm}f$, which implies that
\begin{equation}
f(x)\geq f(z)+f'(z)(x-z)
\end{equation}
and
\begin{equation}
f(y)\geq f(z)+f'(z)(y-z)
\end{equation}
Since $0\leq\theta\leq 1$, these two results above give us
\begin{equation}
\theta f(x)+(1-\theta)f(y)\geq f(z)=\theta x+(1-\theta)y,
\end{equation}
which proves our claim.

For the general case of $f:\mathbb{R}^n\to\mathbb{R}$. Let $x,y\in\mathbb{R}^n$ and consider $f$ restricted to the line passing through $x,y$, i.e. the function
\begin{equation}
g(t)=f(ty+(1-t)x),
\end{equation}
whose derivative is given as
\begin{equation}
g'(t)=\nabla f(ty+(1-t)x)^\text{T}(y-x)
\end{equation}
First suppose that $f$ is convex, and thus so is $g$. Using the above argument for the case of $n=1$, we have that
\begin{equation}
g(1)\geq g(0)+g'(0)
\end{equation}
or
\begin{equation}
f(y)\geq f(x)+\nabla f(x)^\text{T}(y-x)\label{eq:soc.2}
\end{equation}
We continue with assuming that \eqref{eq:soc.2} holds for any $x,y\in\text{dom}\hspace{0.1cm}f$ and $\text{dom}\hspace{0.1cm}f$ is convex.

The convexity of $\text{dom}\hspace{0.1cm}f$ implies that for any $x,y\in\text{dom}\hspace{0.1cm}f$ and for any $0\leq t_1,t_2\leq 1$, we have
\begin{equation}
t_1 y+(1-t_1)x,t_2 y+(1-t_2)x\in\text{dom}\hspace{0.1cm}f
\end{equation}
By \eqref{eq:soc.2} we have
\begin{equation}
f(t_1 y+(1-t_1)x)\geq f(t_2 y+(1-t_2)x)+\nabla f(t_2 y+(1-t_2)x)^\text{T}(y-x)(t_1-t_2)
\end{equation}
or
\begin{equation}
g(t_1)\geq g(t_2)+g'(t_2)(t_1-t_2),
\end{equation}
which implies that $g$ is convex, and hence so is $f$.

#### Second-order conditions{#nd-order-conds}
Consider a function $f$ such that $f$ is twice differentiable. Then $f$ is convex iff $\text{dom}\hspace{0.1cm}f$ is convex and its Hessian is positive semidefinite, i.e. for all $x\in\text{dom}\hspace{0.1cm}f$ we have
\begin{equation}
\nabla^2 f(x)\succeq 0
\end{equation}
Analogously, we have that $f$ is concave iff $\text{dom}\hspace{0.1cm}f$ is convex and for all $x\in\text{dom}\hspace{0.1cm}f$, $\nabla^2 f(x)\preceq 0$.

On the other hands, $f$ is stricly convex (or concave) if $\text{dom}\hspace{0.1cm}f$ is convex and $\nabla^2 f(x)\succ 0$ (or $\nabla^2 f(x)\prec 0$) for all $x\in\text{dom}\hspace{0.1cm}f$. The reverse order is not true.

### Examples{#cvx-funcs-eg}

#### Functions on $\mathbb{R}${#func-on-r}
<ul id='number-list'>
	<li>
		<b>Exponential</b>. $\hspace{0.1cm}e^{ax}$ is convex on $\mathbb{R}$, for any $a\in\mathbb{R}$. Since
		\begin{equation}
		(e^{ax})''=(a e^{ax})'=a^2 e^{ax}\geq 0,
		\end{equation}
		for $a\in\mathbb{R}$.
	</li>
	<li>
		<b>Powers</b>. $\hspace{0.1cm}x^a$ is convex on $\mathbb{R}_{++}$ when $a\geq 1$ or $a\leq 0$, and concave for $0\leq a\leq 1$.<br>
		For $x\in\mathbb{R}_{++}$ and for $a\geq 1$ or $a\leq 0$, we have
		\begin{equation}
		(x^a)''=(a x^{a-1})'=a(a-1)x^{a-2}\geq 0
		\end{equation}
		On the other hands, for $x\in\mathbb{R}_{++}$ and for $0\leq a\leq 1$, we have
		\begin{equation}
		(x^a)''=a(a-1)x^{a-2}\leq 0
		\end{equation}
	</li>
	<li>
		<b>Powers of absolute value</b>. $\hspace{0.1cm}\vert x\vert^p$ is convex on $\mathbb{R}$ for $p\geq 1$.<br>
		For any $0\leq\theta\leq 1$, by triangle inequality we have
		\begin{align}
		f(\theta x+(1-\theta)y)&=\vert\theta x+(1-\theta)y\vert^p \\ &\leq\left(\theta\vert x\vert+(1-\theta)\vert y\vert\right)^p
		\end{align}
	</li>
	<li>
		<b>Logarithm</b>. $\hspace{0.1cm}\log x$ is concave on $\mathbb{R}_{++}$.<br>
		For $x\in\mathbb{R}_{++}$ we have
		\begin{equation}
		(\log x)''=\left(\frac{1}{x}\right)'=\frac{-1}{x^2}\lt 0
		\end{equation}
	</li>
	<li>
		<b>Negative entropy</b>. $\hspace{0.1cm}x\log x$ (either on $\mathbb{R}_{++}$ or on $\mathbb{R}_+$ and defined as $0$ for $x=0$) is convex.<br>
		We have for all $x\in\mathbb{R}_{++}$
		\begin{equation}
		(x\log x)''=(1+\log x)'=\frac{1}{x}>0
		\end{equation}
	</li>
</ul>

#### Functions on $\mathbb{R}^n${#func-on-rn}
<ul id='number-list'>
	<li>
		<b>Norms</b>. Every norm on $\mathbb{R}^n$ is convex.<br>
		For $f:\mathbb{R}^n\to\mathbb{R}$ is a norm and for any $0\leq\theta\leq 1$, by triangle inequality, we have:
		\begin{equation}
		f\big(\theta x+(1-\theta)y\big)\leq f(\theta x)+f\big((1-\theta)y\big)=\theta f(\theta)+(1-\theta)f(y)
		\end{equation}
	</li>
	<li>
		<b>Max function</b>. $\hspace{0.1cm}f(x)=\max{x_1,\ldots,x_n}=\max_{i=1,\ldots,n}x_i$ is convex on $\mathbb{R}^n$.<br>
		For any $0\leq\theta\leq 1$, we have
		\begin{align}
		f(\theta x+(1-\theta)y)&=\max_i(\theta x+(1-\theta)y) \\ &\leq\theta\max_i x_i+(1-\theta)\max_i y_i \\ &=\theta f(x)+(1-\theta)f(y)
		\end{align}
	</li>
	<li>
		<b>Quadratic-over-linear function</b>. The function $f(x,y)=x^2/y$ with
		\begin{equation}
		\text{dom}\hspace{0.1cm}f=\{(x,y)\in\mathbb{R}^2:y>0\}
		\end{equation}
		is convex, since we have its Hessian:
		\begin{equation}
		\nabla^2 f(x,y)=\nabla^2\frac{x^2}{y}=\left[\begin{matrix}2/y&-2x/y^2 \\ -2x/y^2&2x^2/y^3\end{matrix}\right]=\frac{2}{y^3}\left[\begin{matrix}y \\ -x\end{matrix}\right]\left[\begin{matrix}y \\ -x\end{matrix}\right]^\text{T}\succeq 0
		\end{equation}
	</li>
	<li>
		<b>Log-sum-exp</b>. $\hspace{0.1cm}f(x)=\log(e^{x_1}+\ldots+e^{x_n})$ is convex on $\mathbb{R}^n$.<br>
		We have the Hessian of $f(x)$ is
		\begin{equation}
		\nabla^2 f(x)=\nabla^2\log(e^{x_1}+\ldots+e^{x_n})
		\end{equation}
	</li>
</ul>

### Sub-level sets{#sub-lvl-sets}

### Inequalities{#inequalities}

#### Jensen's inequality{#jensens-inequality}

### Operations that preserve convexity{#operations-funcs}

### The conjugate function{#conjugate-func}

### Quasiconvex functions{#quasi-cvx-funcs}

## References
[1] Stephen Boyd & Lieven Vandenberghe. [Convex Optimization](http://www.stanford.edu/∼boyd/cvxbook/). Cambridge UP, 2004.

## Footnotes