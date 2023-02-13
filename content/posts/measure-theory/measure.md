---
title: "Measures"
date: 2021-07-03 07:00:00 +0700
tags: [mathematics, measure-theory, random-stuffs]
math: true
eqn-number: true
---
> When talking about **measure**, you might associate it with the idea of **length**, the measurement of something in one dimension. And then probably, you will extend your idea into two dimensions with **area**, or even three dimensions with **volume**.  

<!--more-->
Despite of having different number of dimensions, all **length**, **area**,  and **volume** share the same properties:
<ul id='roman-list'>
	<li>
		<b>Non-negative</b>: in principle, length, area, and volume can take any positive value. But negative length has no meaning. Same thing happens with negative area and negative volume.
	</li>
	<li>
		<b>Additivity</b>: to get from Hanoi to Singapore by air, you have to transit at Ho Chi Minh city (HCMC). If we cut that path into two non-overlapping pieces, say Hanoi - HCMC, and HCMC - Singapore, then the total length of the two pieces must be equal to the length of original path. If we divide a rectangular into non-overlapping pieces, the area of pieces combined must be the same as the original one. The same is true for volume as well.
	</li>
	<li>
		<b>Empty set</b>: an empty cup of water has volume zero.
	</li>
	<li>
		<b>Other null sets</b>: the length of a point is $0$. The area of a line, or a curve is $0$. The volume of a plane or a surface is also $0$.
	</li>
	<li>
		<b>Translation invariance</b>: length, area and volume are unchanged (<b>invariant</b>) under shifts (<b>translation</b>) in space.
	</li>
	<li>
		<b>Hyper-rectangles</b>: an interval of form $[a, b]\subset\mathbb{R}^3$ has length $b-a$. The area of a rectangle $[a_1,b_1]\times[a_2,b_2]$ is $(b_1-a_1)(b_2-a_2)$. And the volume of a rectangular $[a_1,b_1]\times[a_2,b_2]\times[a_3,b_3]$ is $(b_1-a_1)(b_2-a_2)(b_3-a_3)$. 
	</li>
</ul>

<figure>
	<img src="/images/measure/lego.jpg" alt="Lego" width="70%" height="70%"/>
</figure>

## Lebesgue Measure
Is an extension of the classical notion of length in $\mathbb{R}$, area in $\mathbb{R}^2$ to any $\mathbb{R}^k$ using k-dimensional hyper-rectangles.

**Definition**  
Given an open set $S\equiv\sum_k(a_k,b_k)$ containing disjoint intervals, the **Lebesgue measure** is defined by:
\begin{equation}
\mu_L(S)\equiv\sum_{k}(b_k-a_k)
\end{equation}
Given a closed set $S'\equiv[a,b]-\sum_k(a_k,b_k)$,
\begin{equation}
\mu_L(S')\equiv(b-a)-\sum_k(b_k-a_k)
\end{equation}

## Measures
**Definition**  
Let $\mathcal{X}$ be any set. A *measure* on $\mathcal{X}$ is a function $\mu$ that maps the set of subsets on $\mathcal{X}$ to $[0,\infty]$ ($\mu:2^\mathcal{X}\rightarrow[0,\infty]$) that satisfies:
<ul id='roman-list'>
	<li>
		$\mu(\emptyset)=0$
	</li>
	<li>
		<b>Countable additivity property</b>: for any countable and pairwise disjoint collection of subsets of $\mathcal{X},\mathcal{A_1},\mathcal{A_2},\dots$, we have
		\begin{equation}
		\mu\left(\bigcup_i\mathcal{A_i}\right)=\sum_i\mu(\mathcal{A_i})
		\end{equation}
		$\mu(\mathcal{A})$ is called <b>measure of the set $\mathcal{A}$</b>, or <b>measure of $\mathcal{A}$</b>.
	</li>
</ul>

**Properties**
<ul id='roman-list'>
	<li>
		<b>Monotonicity</b>. If $\mathcal{A}\subset\mathcal{B}$, then $\mu(\mathcal{A})\leq\mu(\mathcal{B})$
	</li>
	<li>
		<b>Subadditivity</b>. If $\mathcal{A_1},\mathcal{A_2},\dots$ is a countable collection of sets, not necessarily disjoint, then
		\begin{equation}
		\mu\left(\bigcup_i\mathcal{A_i}\right)\leq\sum_i\mu(\mathcal{A_i})
		\end{equation}
	</li>
</ul>

**Examples**
<ul id='number-list'>
	<li>
		<b>Cardinality of a set</b>. $\#\mathcal{A}$
	</li>
	<li>
		<b>A point mass at </b>$0$. Consider a measure $\delta_{\{0\}}$ on $\mathbb{R}$ defined to give measure $1$ to any set that contains $0$ and measure $0$ to any set that does not
		\begin{equation}
		\delta_{\{0\}}(\mathcal{A})=\#\left(A\cap\{0\}\right)=\begin{cases}
		1\quad\textsf{if }0\in\mathcal{A} \\ 0\quad\textsf{otherwise}
		\end{cases}
		\end{equation}
		where $\mathcal{A}\subset\mathbb{R}$.
	</li>
	<li>
		<b>Counting measure on the integers</b>. Consider a measure $\mu_\mathbb{Z}$ that assigns to each set $\mathcal{A}$ the number of integers contained in $\mathcal{A}$
		\begin{equation}
		\delta_\mathbb{Z}(\mathcal{A})=\#\left(\mathcal{A}\cap\mathbb{Z}\right)
		\end{equation}
	</li>
	<li>
		<b>Geometric measure</b>. Suppose that $0\lt r\lt 1$. We define a measure on $\mathbb{R}$ that assigns to a set $\mathcal{A}$ a geometrically weighted sum over non-negative integers in $\mathcal{A}$
		\begin{equation}
		\mu(\mathcal{A})=\sum_{i\in\mathcal{A}\cap\mathbb{Z}^+}r^i
		\end{equation}
	</li>
	<li>
		<b>Binomial measure</b>. Let $n\in\mathbb{N}^+$ and let $0\lt p\lt 1$. We define $\mu$ as:
		\begin{equation}
		\mu(\mathcal{A})=\sum_{k\in\mathcal{A}\cap\{0,1,\dots,n\}}{n\choose k}p^k(1-p)^{n-k}
		\end{equation}
	</li>
	<li>
		<b>Bivariate Gaussian</b>. We define a measure on $\mathbb{R}^2$ by:
		\begin{equation}
		\mu({\mathcal{A}})=\int_\mathcal{A}\dfrac{1}{2\pi}\exp\left({\dfrac{-1}{2}(x^2+y^2)}\right)\,dx\,dy
		\end{equation}
	</li>
	<li>
		<b>Uniform on a Ball in </b>$\mathbb{R}^3$. Let $\mathcal{B}$ be the set of points in $\mathbb{R}^3$ that are within a distance $1$ from the origin (unit ball in $\mathbb{R}^3$). We define a measure on $\mathbb{R}^3$ as:
		\begin{equation}
		\mu(\mathcal{A})=\dfrac{3}{4\pi}\mu_L(\mathcal{A}\cap\mathcal{B})
		\end{equation}
	</li>
</ul>

## Integration with respect to a Measure: The Idea
Consider $f:\mathcal{X}\rightarrow\mathbb{R}$, where $\mathcal{X}$ is any set and a measure $\mu$ on $\mathcal{X}$ and compute the integral of $f$ w.r.t $\mu$: $\int f(x)\mu(dx)$. We have:
<ul id='number-list'>
	<li>
		For any function $f$,
		\begin{equation}
		\int g(x)\hspace{0.1cm}\mu_L(dx)=\int g(x)\,dx,
		\end{equation}
		since $\mu_L(dx)\equiv\mu_L([x,x+dx[)=dx$
	</li>
	<li>
		For any function $f$,
		\begin{equation}
		\int g(x)\delta_{\{\alpha\}}(dx)=g(\alpha)
		\end{equation}
		Consider the infinitesimal $\delta_{\{\alpha\}}(dx)$ as $x$ ranges over $\mathbb{R}$. If $x\neq\alpha$, then the infinitesimal interval $[x,x+dx[$ does not contain $\alpha$, so
		\begin{equation}
		\delta_{\{\alpha\}}(dx)\equiv\delta_{\{\alpha\}}([x,x+dx[)=0
		\end{equation}
		If $x=\alpha,\delta_{\{\alpha\}}(dx)\equiv\delta_{\{\alpha\}}([x,x+dx[)=1$. Thus, when we add up all of the infinitesimals, we get $g(\alpha)\cdot1=g(\alpha)$
	</li>
	<li>
		For any function $f$,
		\begin{equation}
		\int g(x)\hspace{0.1cm}\delta_\mathbb{Z}(dx)=\sum_{i\in\mathbb{Z}}g(i)
		\end{equation}
		Similarly, consider the infinitesimal $\delta_\mathbb{Z}(dx)$ as $x$ ranges over $\mathbb{R}$. If $x\notin\mathbb{Z}$, then $\delta_\mathbb{Z}(dx)\equiv\delta_\mathbb{Z}([x,x+dx[)=0$. And otherwise if $x\in\mathbb{Z}$, $\delta_\mathbb{Z}(dx)\equiv\delta_\mathbb{Z}([x,x+dx[)=1$ since an infinitesimal interval can contain at most one integer.<br>
		Hence, $g(x)\hspace{0.1cm}\delta_\mathbb{Z}=g(x)$ if $x\in\mathbb{Z}$ and $=0$ otherwise. When we add up all of the infinitesimals over $x$, we get the sum above.
	</li>
	<li>
		Suppose $\mathcal{C}$ is a countable set. We can define <b>counting measure</b> on $\mathcal{C}$ to map $\mathcal{A}\rightarrow\#(\mathcal{A}\cap\mathcal{C})$ (recall that $\delta_\mathcal{C}(\mathcal{A})=\#(\mathcal{A}\cap\mathcal{C})$). For any function $f$,
		\begin{equation}
		\int g(x)\hspace{0.1cm}\delta_\mathcal{C}(dx)=\sum_{v\in\mathcal{C}}g(v),
		\end{equation}
		using the same basic argument as in the above example.
	</li>
</ul>

From the above examples, we have that *integrals w.r.t to Lebesgue measure are just ordinary integrals, and that integrals w.r.t Counting measure are just ordinary summation*.  
Consider measures built from Lebesgue and Counting measure, we have:
<ul id='number-list'>
	<li>
		Suppose $\mu$ is a measure that satisfies $\mu(dx)=f(x)\mu_L(dx)$, then for any function $g$,
		\begin{equation}
		\int g(x)\mu(dx)=\int g(x)f(x)\mu_L(dx)=\int g(x)f(x)\,dx
		\end{equation}
		We say that $f$ is the density of $\mu$ w.r.t Lebesgue measure in this case.
	</li>
	<li>
		Suppose $\mu$ is a measure that satisfies $\mu(dx)=p(x)\delta_\mathcal{C}(dx)$ for a countable set $\mathcal{C}$, then for any function g,
		\begin{equation}
		\int g(x)\mu(dx)=\int g(x)p(x)\delta_\mathcal{C}(dx)=\sum_{v\in\mathcal{C}}g(v)f(v)
		\end{equation}
		We say that $p$ is the density of $\mu$ w.r.t Counting measure on $\mathcal{C}$.
	</li>
</ul>

## Properties of the Integral
A function is said to be **integrable** w.r.t $\mu$ if
\begin{equation}
\int\vert f(x)\vert\mu(dx)<\infty
\end{equation}
An integrable function has a well-defined and finite integral. If $f(x)\geq0$, the integral is always well-defined but may be $\infty$.

Suppose $\mu$ is a measure on $\mathcal{X},\mathcal{A}\subset\mathcal{X}$, and $g$ is a real-valued function on $\mathcal{X}$. We define the integral of $g$ over the set $\mathcal{A}$, denoted by $\int_\mathcal{A}g(x)\hspace{0.1cm}\mu(dx)$, as
\begin{equation}
\int_\mathcal{A}g(x)\mu(dx)=\int g(x)𝟙\_\mathcal{A}(x)\hspace{0.1cm}\mu(dx),
\end{equation}
where $𝟙_\mathcal{A}$ is an **indicator function** ($𝟙_\mathcal{A}(x)=1$ if $x\in\mathcal{A}$, and $=0$ otherwise).

Let $\mu$ is a measure on $\mathcal{X},\mathcal{A},\mathcal{B}\subset\mathcal{X},c\in\mathbb{R}$ and $f,g$ are integrable functions. The following properties hold for every $\mu$
<ul id='roman-list'>
	<li>
		<b>Constant functions</b>.
		\begin{equation}
		\int_\mathcal{A}c\,\mu(dx)=c\cdot\mu(\mathcal{A})
		\end{equation}
	</li>
	<li>
		<b>Linearity</b>.
		\begin{align}
		\int_\mathcal{A}cf(x)\mu(dx)&=c\int_\mathcal{A}f(x)\mu(dx) \\ \int_\mathcal{A}\big(f(x)+g(x)\big)\mu(dx)&=\int_\mathcal{A}f(x)\mu(dx)+\int_\mathcal{A}g(x)\mu(dx)
		\end{align}
	</li>
	<li>
		<b>Monotonicity</b>. If $f\leq g$, then
		\begin{equation}
		\int_\mathcal{A}f(x)\mu(dx)\leq\int_\mathcal{A}g(x)\mu(dx),\forall\mathcal{A},
		\end{equation}
		which implies:
		<ul>
			<li>
				If $f\geq0$, then $\int f(x)\mu(dx)\geq0$.
			</li>
			<li>
				If $f\geq0$ and $\mathcal{A}\subset\mathcal{B}$, then $\int_\mathcal{A}f(x)\mu(dx)\leq\int_\mathcal{B}f(x)\mu(dx)$.
			</li>
		</ul>
	</li>
	<li>
		<b>Null sets</b>. If $\mu(\mathcal{A})=0$, then $\int_\mathcal{A}f(x)\mu(dx)=0$.
	</li>
	<li>
		<b>Absolute values</b>.
		\begin{equation}
		\left\vert\int f(x)\mu(dx)\right\vert\leq\int\left\vert f(x)\right\vert\mu(dx)
		\end{equation}
	</li>
	<li>
		<b>Monotone convergence</b>. If $0\leq f_1\leq f_2\leq\dots$ is an increasing sequence of integrable functions that converge to $f$, then
		\begin{equation}
		\lim_{k\to\infty}\int f_k(x)\mu(dx)=\int f(x)\mu(dx)
		\end{equation}
	</li>
	<li>
		<b>Linearity in region of integration</b>. If $\mathcal{A}\cap\mathcal{B}=\emptyset$,
		\begin{equation}
		\int_{\mathcal{A}\cup\mathcal{B}}f(x)\mu(dx)=\int_\mathcal{A}f(x)\mu(dx)+\int_\mathcal{B}f(x)\mu(dx)
		\end{equation}
	</li>
</ul>

## Integration with respect to a Measure: The Details
<ul id='number-list'>
	<li>
		<b>Step 1</b>.<br>
		- Define the integral for <b>simple functions</b>, i.e. functions that take only a finite number of different values and have following properties:
		<ul id='roman-list'>
			<li>
				All constant functions are simple functions.
			</li>
			<li>
				The indicator function ($𝟙_\mathcal{A}$) of a set $\mathcal{A}\subset\mathcal{X}$ is a simple function (taking values in $\{0,1\}$).
			</li>
			<li>
				Any constant times an indicator ($c𝟙_\mathcal{A}$) is also a simple function (taking values in $\{0,c\}$).
			</li>
			<li>
				Similarly, given disjoint sets $\mathcal{A_1},\mathcal{A_2}$, the linear combination $c_1𝟙_\mathcal{A_1}+c_2𝟙_\mathcal{A_2}$ is a simple function (taking values in $\{0,c_1,c_2\}$)[^1].
			</li>
			<li>
				In fact, any simple function can be expressed as a linear combination of a finite number of indicator functions. That is, if $f$ is *any* simple function on $\mathcal{X}$, then there exists some finite integer $n$, non-zero constants $c_1,\dots,c_n$ and *disjoint* sets $\mathcal{A_1},\dots\mathcal{A_n}\subset\mathcal{X}$ such that
				\begin{equation}
				f=c_1𝟙\_\mathcal{A_1}+\dots+c_n𝟙\_\mathcal{A_n}
				\end{equation}
			</li>
		</ul>
		- So, if $f:\mathcal{X}\to\mathbb{R}$ is a simple function as just defined, we have that
		\begin{equation}
		\int \mu(dx)=c_1\mu(\mathcal{A_1})+\dots+c_n\mu(\mathcal{A_n})\
		\end{equation}
	</li>
	<li>
		<b>Step 2</b>.<br>
		- Define the integral for general non-negative functions, approximating the general function by simple functions.<br>
		- The idea is that we can approximate any general non-negative function $f:\mathcal{X}\to[0,\infty[$ well by some non-negative simple functions that $\leq f$[^2].<br>
		- If $f:\mathcal{X}\to[0,\infty[$ is a general function and $0\leq s\leq f$ is a simple function (then $\int s(x)\mu(dx)\leq\int f(x)\mu(dx)$). The closer that $s$ approximates $f$, the closer we expect $\int s(x)\mu(dx)$ and $\int f(x)\mu(x)$ to be.<br>
		- To be more precise, we define the integral $\int f(x)\mu(dx)$ to be the smallest value $I$ such that $\int s(x)\mu(x)\leq I$, for all simple functions $0\leq s\leq f$.
		\begin{equation}
		\int f(x)\mu(dx)\approx\sup\left\{\int s(x)\mu(dx)\right\}
		\end{equation}
	</li>
	<li>
		<b>Step 3</b>.<br>
		- Define the integral for general real-valued functions by separately integrating the positive and negative parts of the function.<br>
		If $f:\mathcal{X}\to\mathbb{R}$ is a general function, we can define its <b>positive part</b> $f^+$ and its <b>negative part</b> $f^-$ by
		\begin{align}
		f^+(x)&=\max\left(f(x),0\right) \\ f^-(x)&=\max\left(-f(x),0\right)
		\end{align}
		- Since both $f^+$ and $f^-$ are non-negative functions and $f=f^+-f^-$, we have
		\begin{equation}
		\int f(x)\mu(dx)=\int f^+(x)\mu(dx)-\int f^-(x)\mu(dx)
		\end{equation}
		- This is a well-defined number (possibly infinite) if and only if at least one of $f^+$ and $f^-$ has a finite integral.
	</li>
</ul>

## Constructing Measures from old ones
<ul id='number-list'>
	<li>
		<b>Sums and multiples</b>.<br>
		- Consider the point mass measures at $0$ and $1$, $\delta_{\{0\}},\delta_{\{1\}}$, and construct a two new measures on $\mathbb{R}$, $\mu=\delta_{\{0\}}+\delta_{\{1\}}$ and $v=4\delta_{\{0\}}$, defined by
		\begin{align}
		\mu(\mathcal{A})&=\delta_{\{0\}}(\mathcal{A})+\delta_{\{0\}}(\mathcal{A}) \\ v(\mathcal{A})&=4\delta_{\{0\}}(\mathcal{A})
		\end{align}
		- The measure $\mu$ counts how many elements of $\{0,1\}$ are in its argument. Thus, the counting measure of the integers can be re-expressed as
		\begin{equation}
		\delta_\mathbb{Z}=\sum_{i=-\infty}^{\infty}\delta_{\{i\}}
		\end{equation}
		- By combining the operations of summation and multiplication, we can write the Geometric measure in the above example 
		\begin{equation}
		\sum_{i=0}^{\infty}r^i\delta_{\{i\}}
		\end{equation}
	</li>
	<li>
		<b>Restriction to a subset</b>.<br>
		Suppose $\mu$ is a measure on $\mathcal{X}$ and $\mathcal{B}\subset\mathcal{X}$. We can define a new measure on $\mathcal{B}$ which maps $\mathcal{A}\subset\mathcal{B}\to\mu(\mathcal{A})$. This is called the restriction of $\mu$ to the set $\mathcal{B}$.
	</li>
	<li>
		<b>Measure induced by a function</b>.
		- Suppose $\mu$ is a measure on $\mathcal{X}$ and $g:\mathcal{X}\to\mathcal{Y}$. We can use $\mu$ and $g$ to define a new measure $v$ on $\mathcal{Y}$ by
		\begin{equation}
		v(\mathcal{A})=\mu(g^{-1}(\mathcal{A})),
		\end{equation}
		for $\mathcal{A}\subset\mathcal{Y}$. This is called the *measure induced from $\mu$ by $g$*.
		- Therefore, for any $f:\mathcal{Y}\to\mathbb{R}$,
		\begin{equation}
		\int f(y)\hspace{0.1cm}v(dy)=\int f(g(x))\hspace{0.1cm}\mu(dx)
		\end{equation}
	</li>
	<li>
		<b>Integrating a density</b>.<br>
		- Suppose $\mu$ is a measure on $\mathcal{X}$ and $f:\mathcal{X}\to\mathbb{R}$. We can define a new measure $v$ on $\mathcal{X}$ as
		\begin{equation}
		v(\mathcal{A})=\int_\mathcal{A}f(x)\hspace{0.1cm}\mu(dx)\label{eq:1}
		\end{equation}
		- We say that $f$ is the <b>density</b> of the measure $v$ w.r.t $\mu$.<br>
		- If $v,\mu$ are measures for which the equation \eqref{eq:1} holds for every $\mathcal{A}\subset\mathcal{X}$, we say that $v$ has a density $f$ w.r.t $\mu$. This implies two useful results:<br>
		<ul id='roman-list'>
			<li>
				$\mu(\mathcal{A})=0$ implies $v(\mathcal{A})=0$.
			</li>
			<li>
				$v(dx)=f(x)\hspace{0.1cm}\mu(dx)$.
			</li>
		</ul>
</ul>

## Other types of Measures
Suppose that $\mu$ is a measure on $\mathcal{X}$
<ul id='number-list'>
	<li>
		If $\mu(\mathcal{X})=\infty$, we say that $\mu$ is an <b>infinite measure</b>.
	</li>
	<li>
		If $\mu(\mathcal{X}<\infty)$, we say that $\mu$ is a <b>finite measure</b>.
	</li>
	<li>
		If $\mu(\mathcal{X}<1)$, we say that $\mu$ is a <b>probability measure</b>.
	</li>
	<li>
		If there exists a countable set $\mathcal{S}$ such that $\mu(\mathcal{X}-\mathcal{S})=0$, we say that $\mu$ is a <b>discrete measure</b>. Equivalently, $\mu$ has a density w.r.t <b>counting measure</b> on $\mathcal{S}$.
	</li>
	<li>
		If $\mu$ has a density w.r.t Lebesgue measure, we say that $\mu$ is a <b>continuous measure</b>.
	</li>
	<li>
		If $\mu$ is neither <b>continuous</b> nor <b>discrete</b>, we say that $\mu$ is a <b>mixed measure</b>.
	</li>
</ul>

## References
[1] Literally, this note is mainly written from a source that I've lost the reference :(. Hope that I can update this line soon.  

[2] [Lebesgue Measure](https://mathworld.wolfram.com/LebesgueMeasure.html).  

[3] [Measure Theory for Probability: A Very Brief Introduction](https://www.countbayesie.com/blog/2015/8/17/a-very-brief-and-non-mathematical-introduction-to-measure-theory-for-probability).

## Other Resources
1. [Music and Measure Theory - 3Blue1Brown](https://www.youtube.com/watch?v=cyW5z-M2yzw) - this is my favourite Youtube channel.

## Footnotes
[^1]: If $\mathcal{A_1},\mathcal{A_2}$ were not disjoint, we could define $\mathcal{B_1}=\mathcal{A_1}-\mathcal{A_2}$, $\mathcal{B_2}=\mathcal{A_2}-\mathcal{A_1}$, and $\mathcal{B_3}=\mathcal{A_1}\cap\mathcal{A_2}$. Then the function is equal to $$c_1𝟙_\mathcal{B_1}+c_2𝟙_\mathcal{B_2}+(c_1+c_2)𝟙_\mathcal{B_3}$$.
[^2]: 

