---
title: "Read-through: Measure theory - the Lebesgue integral"
date: 2022-08-21 13:00:00 +0700
tags: [mathematics, measure-theory, lebesgue-integral]
math: true
eqn-number: true
---
> (Temporary being stopped) Note III of the measure theory series. Materials are mostly taken from [Tao's book]({{< ref "measure-theory-p3#taos-book" >}}), except for some needed notations extracted from [Stein's book]({{< ref "measure-theory-p3#steins-book" >}}).
<!--more-->

## Integration of simple functions{#int-simp-funcs}
Analogy to how the [**Riemann integral**]({{< ref "measure-theory-p1#riemann-integrability" >}}) was established by using the integral for [**piecewise constant functions**]({{< ref "measure-theory-p1#pc-func" >}}), the **Lebesgue integral** is set up using the integral for **simple functions**.

### Simple function{#simp-func}
A (complex-valued) **simple function** $f:\mathbb{R}^d\to\mathbb{C}$ is a finite linear combination
\begin{equation}
f=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k},\label{eq:sf.1}
\end{equation}
of indicator functions $1_{E_i}$ of Lebesgue measurable sets $E_i\subset\mathbb{R}^d$ for $i=1,\ldots,k$, for natural number $k\geq 0$ and where $c_1,\ldots,c_k\in\mathbb{C}$ are complex numbers.

An **unsigned simple function** $f:\mathbb{R}^d\to[0,+\infty]$ is given as \eqref{eq:sf.1} but with the $c_i$ taking values in $[0,+\infty]$ rather than $\mathbb{C}$.

### Integral of a unsigned simple function{#int-unsgn-simp-func}
If $f=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}$ is an unsigned simple function, the integral $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx$ is defined by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx\doteq c_1 m(E_1)+\ldots+c_k m(E_k),
\end{equation}
which means $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx\in[0,+\infty]$.

### Well-definedness of simple integral{#well-dfn-simp-int}
**Lemma 1**  
*Let $k,k'\geq 0$ be natural, $c_1,\ldots,c_k,c_1',\dots,c_{k'}'\in[0,+\infty]$ and $E_1,\ldots,E_k,E_1',\ldots,E_{k'}'\subset\mathbb{R}^d$ be Lebesgue measurable sets such that the identity
\begin{equation}
c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}=c_1' 1\_{E_1'}+\ldots+c_{k'}' 1\_{E_{k'}'}\label{eq:lemma1.1}
\end{equation}
holds identically on $\mathbb{R}^d$. Then we have*
\begin{equation}
c_1 m(E_1)+\ldots+c_k m(E_k)=c_1' m(E_1')+\ldots+c_{k'}' m(E_{k'}')
\end{equation}

**Proof**  
The $k+k'$ sets $E_1,\ldots,E_k,E_1',\ldots,E_{k'}'$ partition $\mathbb{R}^d$ into $2^{k+k'}$ disjoint sets, each of which is an intersection of some of the $E_1,\ldots,E_k,E_1',\ldots,E_{k'}'$ and their complements[^1].

Removing any sets that are empty, we end up with a partition of $R^d$ of $m$ non-empty disjoint sets $A_1,\ldots,A_m$ for some $0\leq m\leq 2^{k+k'}$. It easily seen that $A_1,\ldots,A_m$ are then Lebesgue measurable due to the Lebesgue measurability of $E_1,\ldots,E_k,E_1',\ldots,E_{k'}'$.

With this set up, each of the $E_1,\ldots,E_k,E_1',\ldots,E_{k'}'$ are unions of some of the $A_1,\ldots,A_m$. Or in other words, we have
\begin{equation}
E_1=\bigcup_{j\in J_i}A_j,
\end{equation}
and
\begin{equation}
E_{i'}'=\bigcup_{j'\in J_{i'}'}A_j',
\end{equation}
for all $i=1,\ldots,k$ and $i'=1,\ldots,k'$, and some subsets $J_i,J_{i'}'\subset\\{1,\ldots,m\\}$. By finite additivity property of Lebesgue measure, we therefore have
\begin{equation}
m(E_i)=\sum_{j\in J_i}m(A_j)
\end{equation}
and
\begin{equation}
m(E_{i'}')=\sum_{j\in J_{i'}'}m(A_j)
\end{equation}
Hence, the problem remains to show that
\begin{equation}
\sum_{i=1}^{k}c_i\sum_{j\in J_i}m(A_j)=\sum_{i'=1}^{k'}c_{i'}'\sum_{j\in J_{i'}'}m(A_j)\label{eq:lemma1.2}
\end{equation}
Fix $1\leq j\leq m$, we have that at each point $x$ in the non-empty set $A_j$, $1\_{E_i}(x)$ is equal to $1\_{J_i}(j)$, and similarly $1\_{E_{i'}'}(x)$ is equal to $1\_{J_{i'}'}(j)$. Then from \eqref{eq:lemma1.1} we have
\begin{equation}
\sum_{i=1}^{k}c_i 1\_{J_i}(j)=\sum_{i'=1}^{k'}c_{i'}'1\_{J_{i'}'}(j)
\end{equation}
Multiplying both sides by $m(A_j)$ and then summing over all $j=1,\ldots,m$, we obtain \eqref{eq:lemma1.2}

### Almost everywhere and support{#alm-evwhr-spt}
A property $P(x)$ of a point $x\in\mathbb{R}^d$ is said to hold **(Lebesgue) almost everywhere** in $\mathbb{R}^d$ or for **(Lebesgue) almost every point** $x\in\mathbb{R}^d$, if the set of $x\in\mathbb{R}^d$ for which $P(x)$ fails has Lebesgue measure of zero (i.e. $P$ is true outside of a null set).

Two functions $f,g:\mathbb{R}^d\to Z$ into an arbitrary range $Z$ are referred to **agree almost everywhere** if we have $f(x)=g(x)$ almost every $x\in\mathbb{R}^d$.

The **support** of a function $f:\mathbb{R}^d\to\mathbb{C}$ or $f:\mathbb{R}^d\to[0,+\infty]$ is defined to be the set $\\{x\in\mathbb{R}^d:f(x)\neq 0\\}$ where $f$ is non-zero.

**Remark 2**
<ul id='number-list'>
	<li>
		If $P(x)$ holds for almost every $x$, and $P(x)$ implies $Q(x)$, then $Q(x)$ holds for almost every $x$.
	</li>
	<li>
		If $P_1(x),P_2(x),\ldots$ are an at most countable family of properties, each of which individually holds for almost every $x$, then they will simultaneously holds for almost every $x$, since the countable union of null sets is still a null set.
	</li>
</ul>

### Basic properties of the simple unsigned integral{#bsc-prop-simp-unsgn-int}
Let $f,g:\mathbb{R}^d\to[0,+\infty]$ be simple unsigned functions.
<ul id='roman-list'>
	<li>
		<b>Unsigned linearity</b>. We have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\hspace{0.1cm}dx=\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
		and
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\hspace{0.1cm}dx=c\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx,
		\end{equation}
		for all $c\in[0,+\infty]$.
	</li>
	<li>
		<b>Finiteness</b>. We have $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx<\infty$ iff $f$ is finite almost everywhere, and its support has finite measure.
	</li>
	<li>
		<b>Vanishing</b>. We have $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=0$ iff $f$ is zero almost everywhere.
	</li>
	<li>
		<b>Equivalence</b>. If $f$ and $g$ agree almost everywhere, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
	</li>
	<li>
		<b>Monotonicity</b>. If $f(x)\leq g(x)$ for almost every $x\in\mathbb{R}^d$, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx\leq\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b>. For any Lebesgue measurable $E$, we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}1_E(x)\hspace{0.1cm}dx=m(E)
		\end{equation}
	</li>
</ul>

**Proof**  
Since $f,g:\mathbb{R}^d\to[0,+\infty]$ are simple unsigned functions, we can assume that
\begin{align}
f&=c_1 1\_{E_1}+\ldots+c_k 1\_{E_k}, \\\\ g&=c_1' 1\_{E_1'}+\ldots+c_{k'}' 1\_{E_{k'}'},
\end{align}
where $c_1,\ldots,c_k,c_1',\ldots,c_{k'}'\in[0,+\infty]$.
<ul id='roman-list'>
	<li>
		<b>Unsigned linearity</b><br>
		We have
		\begin{align}
		\hspace{-1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\hspace{0.1cm}dx&=c_1 m(E_1)+\ldots+c_k m(E_k)+c_1' m(E_1')+\ldots+c_{k'}' m(E_{k'}') \\ &=\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{align}
		For any $c\in[0,+\infty]$, we have
		\begin{align}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\hspace{0.1cm}dx&=c\left(c_1 m(E_1)+\ldots+c_k m(E_k)\right) \\ &=c\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx
		\end{align}
	</li>
	<li>
		<b>Finiteness</b><br>
		Given $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx<\infty$, then for every $i=1,\ldots,k$ we have that
		\begin{equation}
		c_i m(E_i)<\infty\label{eq:bpsui.1}
		\end{equation}
		Suppose that $f$ is not finite almost everywhere, which means that there exists $1\leq i\leq k$ such that $E_i$ is a non-null set and $c_i=\infty$, or
		\begin{equation}
		c_i m(E_i)=\infty,
		\end{equation}
		which is in contrast with \eqref{eq:bpsui.1}.<br>
		Suppose that the support of $f$ has infinite measure, or in other word
		\begin{equation}
		c_i\neq 0,\hspace{1cm}i=1,\ldots,k\label{eq:bpsui.2}
		\end{equation}
		and
		\begin{equation}
		m\left(\bigcup_{n=1}^{k}E_n\right)=\infty,
		\end{equation}
		Since any $k$ subsets $E_1,\ldots,E_k$ of $\mathbb{R}^d$ partition $\mathbb{R}^d$ into $2^k$ disjoint sets, say $F_1,\ldots,F_{2^k}$. Hence, by finite additivity property of Lebesgue measure, we have
		\begin{equation}
		\sum_{n=1}^{2^k}m(F_n)=\infty,
		\end{equation}
		which implies that there exists $1\leq n\leq 2^k$ such that $m(F_n)=\infty$. And therefore, for a particular $1\leq i\leq k$ such that $F_n\subset E_i$, by monotonicity property of Lebesgue measure
		\begin{equation}
		m(E_i)\geq m(F_n)=\infty
		\end{equation}
		Thus, combining with \eqref{eq:bpsui.2} gives us
		\begin{equation}
		c_i m(E_1)=\infty,
		\end{equation}
		which again contradicts to \eqref{eq:bpsui.1}.<br>
		Given $f$ is finite almost everywhere and its support has finite measure, suppose that its integral is infinite, or
		\begin{equation}
		c_1 m(E_1)+\ldots+c_k m(E_k)=\infty,
		\end{equation}
		which implies that there exists $1\leq i\leq k$ such that either<br>
		(1) $c_i=\infty$ and $E_i$ is a non-null set, or<br>
		(2) $c_i\neq 0$ and $m(E)=\infty$.<br>
		If (1) happens, we then have that
		\begin{equation}
		f\geq c_i 1_{E_i}=\infty,
		\end{equation}
		which contradicts to our hypothesis.<br>
		If (2) happens, by monotonicity of Lebesgue measure, the support of $f$ then has infinite measure, which also contradicts to our hypothesis.
	</li>
	<li>
		<b>Vanishing</b><br>
		Given $\text{Simp}\int_{\mathbf{R^d}}f(x)\hspace{0.1cm}dx=0$, we then have
		\begin{equation}
		c_1 m(E_1)+\ldots+c_k m(E_k)=0,
		\end{equation}
		which implies that for every $1\leq i\leq k$, we have that $c_i=0$ or $E_i$ is a null set.
		Therefore, $f$ is zero almost everywhere because in this case $f$ takes the value of non-zero iff $x$ is in a particular null set $E_j$.<br>
		Given $f$ is zero almost everywhere, for every $i=1,\ldots,k$, we have that either<br>
		(1) $c_i=0$, or<br>
		(2) $c_i\neq 0$ and $x\notin E_i$ with $E_i$ is a null set, or<br>
		(3) $c_i=0$ and and $x\notin E_i$ with $E_i$ is a null set.<br>
		Therefore, the integral of $f$:
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=c_1 m(E_1)+\ldots+c_k m(E_k)=0
		\end{equation}
	</li>
	<li>
		<b>Equivalence</b><br>
		Given $f$ and $g$ agree almost everywhere, we have that at any point $x\in\mathbb{R}^d$ such that $f(x)=g(x)$, by <b>lemma 1</b>, we obtain
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
		For more convenient, let $K=\{E_i\cap E_{i'}':1\leq i\leq k,1\leq i'\leq k'\}$. The set $K$ then has cardinality of $kk'$. Thus, without loss of generality, we can denote $K$ as
		\begin{equation}
		K=\{K_1,\ldots,K_{kk'}\}
		\end{equation}
		With this definition of $K$, the functions $f$ and $g$ can be rewritten by
		\begin{equation}
		f=a_1 1_{K_1}+\ldots+a_{kk'}1_{K_{kk'}}\label{eq:bpsui.3}
		\end{equation}
		and
		\begin{equation}
		g=b_1 1_{K_1}+\ldots+b_{kk'}1_{K_{kk'}}\label{eq:bpsui.4}
		\end{equation}
		On the other hand, the set in which $f(x)\neq g(x)$ is a null set. Thus by \eqref{eq:bpsui.3} and \eqref{eq:bpsui.4}, we have $x\in A$, where some $A\subset K$ is a null set, and for each $i$ such that $K_i\subset A$ (thus is also a null set, or $m(K_i)=0$), $a_i\neq b_i$, otherwise if $K_i\notin A$, $a_i=b_i$. Therefore, we obtain
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=\sum_{i,K_i\notin A}c_i m(K_i)
		\end{equation}
		and
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx=\sum_{i,K_i\notin A}c_i m(K_i)
		\end{equation}
		which proves our claim.
	</li>
	<li>
		<b>Monotonicity</b><br>
		Using the same procedure as the proof for equivalence, our claim can be proved.
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b><br>
		This follows directly from definition
	</li>
</ul>

### Absolutely convergence simple integral{#abs-cvg-simp-int}
A complex valued simple function $f:\mathbb{R}^d\to\mathbb{C}$ is known as **absolutely integrable** if
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}\vert f(x)\vert\hspace{0.1cm}dx<\infty
\end{equation}
If $f$ is absolutely integrable, the integral $\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx$ is defined for real signed $f$ by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx\doteq\text{Simp}\int_{\mathbb{R}^d}f_+(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}f_-(x)\hspace{0.1cm}dx,
\end{equation}
where
\begin{align}
f_+(x)&\doteq\max\left(f(x),0\right), \\\\ f_-(x)&\doteq\max\left(-f(x),0\right),
\end{align}
and for complex-valued $f$ by the formula
\begin{equation}
\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx\doteq\text{Simp}\int_{\mathbb{R}^d}\text{Re}\hspace{0.1cm}f(x)\hspace{0.1cm}dx+i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Im}\hspace{0.1cm}f(x)\hspace{0.1cm}dx
\end{equation}

### Basic properties of the complex-valued simple integral{#bsc-prop-cmplx-simp-int}
Let $f,g:\mathbb{R}^d\to\mathbb{C}$ be absolutely integrable simple functions
<ul id='roman-list'>
	<li>
		<b>*-linearity</b>. We have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\hspace{0.1cm}dx=\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
		and
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\hspace{0.1cm}dx=c\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx,
		\end{equation}
		for all $c\in\mathbb{C}$. Also we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}\overline{f}(x)\hspace{0.1cm}dx=\overline{\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx}
		\end{equation}
	</li>
	<li>
		<b>Equivalence</b>. If $f$ and $g$ agree almost everywhere, then
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx=\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{equation}
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b>. For any Lebesgue measurable $E$, we have
		\begin{equation}
		\text{Simp}\int_{\mathbb{R}^d}1_E(x)\hspace{0.1cm}dx=m(E)
		\end{equation}
	</li>
</ul>

**Proof**  
We first consider the case of real-valued $f$ and $g$.
<ul id='roman-list'>
	<li>
		<b>*-linearity</b><br>
		Using the identity
		\begin{equation}
		f+g=(f+g)_{+}-(f+g)_{-}=(f_{+}-f_{-})+(g_{+}-g_{-})
		\end{equation}
	</li>
	<li>
		<b>Equivalence</b><br>
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b><br>
	</li>
</ul>
For complex-valued $f$ and $g$ we have:
<ul id='roman-list'>
	<li>
		<b>*-linearity</b><br>
		By definition of complex-valued simple integral and by linearity of simple unsigned integral we have
		\begin{align}
		&\text{Simp}\int_{\mathbb{R}^d}f(x)+g(x)\hspace{0.1cm}dx\nonumber \\ &=\text{Simp}\int_{\mathbb{R}^d}\text{Re}(f(x)+g(x))\hspace{0.1cm}dx+i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Im}(f(x)+g(x))\hspace{0.1cm}dx \\ &=\text{Simp}\int_{\mathbb{R}^d}\text{Re}f(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}\text{Re}g(x)\hspace{0.1cm}dx\nonumber \\ &\hspace{2cm}+i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Im}f(x)\hspace{0.1cm}dx+i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Im}g(x)\hspace{0.1cm}dx \\ &=\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx+\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx
		\end{align}
		For the complex conjugate $\overline{f}$, we have its integral can be written as
		\begin{align}
		\text{Simp}\int_{\mathbb{R}^d}\overline{f}(x)\hspace{0.1cm}dx&=\text{Simp}\int_{\mathbb{R}^d}\text{Re}f(x)\hspace{0.1cm}dx-\text{Simp}\int_{\mathbb{R}^d}\text{Im}f(x)\hspace{0.1cm}dx \\ &=\overline{\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx}
		\end{align}
		Also, for any $c\in\mathbb{C}$, using linearity of simple unsigned integrals once again gives us
		\begin{align}
		\text{Simp}\int_{\mathbb{R}^d}cf(x)\hspace{0.1cm}dx&=\text{Simp}\int_{\mathbb{R}^d}c\hspace{0.1cm}\text{Re}f(x)\hspace{0.1cm}dx+i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}c\hspace{0.1cm}\text{Im}f(x)\hspace{0.1cm}dx \\ &=c\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Re}f(x)\hspace{0.1cm}dx+c i\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}\text{Im}f(x)\hspace{0.1cm}dx \\ &=c\hspace{0.1cm}\text{Simp}\int_{\mathbb{R}^d}f(x)\hspace{0.1cm}dx
		\end{align}
	</li>
	<li>
		<b>Equivalence</b><br>
	</li>
	<li>
		<b>Compatibility with Lebesgue measure</b><br>
	</li>
</ul>

## Measurable functions{#msr-funcs}
Just as how the piecewise constant integral can be extended to the Riemann integral, the unsigned simple integral can be extended to the unsigned Lebesgue integral, by expanding the class of unsigned simple functions to the broader class of **unsigned Lebesgue measurable functions**.

### Unsigned measurable functions{#unsgn-msr-funcs}
An unsigned function $f:\mathbb{R}^d\to[0,+\infty]$ is **unsigned Lebesgue measurable**, or **measurable**, if it is the pointwise limit of unsigned simple functions, i.e. if there exists a sequence $f_1,f_2,\ldots:\mathbb{R}\to[0,+\infty]$ of unsigned simple functions such that $f_n(x)\to f(x)$ for every $x\in\mathbb{R}^d$.

### Equivalent notions of measurability{#equiv-ntn-msrb}
**Lemma 3**  
Let $f:\mathbb{R}\to[0,+\infty]$ be an unsigned function. The following are then equivalent:
<ul id='roman-list'>
	<li>
		$f$ is unsigned Lebesgue measurable.
	</li>
	<li>
		$f$ is the pointwise limit of unsigned simple functions $f_n$ (hence $\lim_{n\to\infty}f_n(x)$ exists and is equal to $f(x)$ for all $x\in\mathbb{R}^d$).
	</li>
	<li>
		$f$ is the pointwise almost everywhere limit of unsigned simple function $f_n$ (thus $\lim_{n\to\infty}f_n(x)$ exists and is equal to $f(x)$ for almost every $x\in\mathbb{R}^d$).
	</li>
	<li>
		$f(x)=\sup_n f_n(x)$, where $0\leq f_1\leq f_2\leq\ldots$ is an increasing sequence of unsigned simple functions, each of which are bounded with finite measure support.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)>\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)\geq\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)<\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every $\lambda\in[0,+\infty]$, the set $\{x\in\mathbb{R}^d:f(x)\leq\lambda\}$ is Lebesgue measurable.
	</li>
	<li>
		For every interval $I\subset[0,+\infty)$, the set $f^{-1}(I)\doteq\{x\in\mathbb{R}^d:f(x)\in I\}$ is Lebesgue measurable.
	</li>
	<li>
		For every (relatively) open set $U\subset[0,+\infty)$, the set $f^{-1}(U)\doteq\{x\in\mathbb{R}^d:f(x)\in U\}$ is Lebesgue measurable.
	</li>
	<li>
		For every (relatively) closed set $K\subset[0,+\infty)$, the set $f^{-1}(K)\doteq\{x\in\mathbb{R}^d:f(x)\in K\}$ is Lebesgue measurable.
	</li>
</ul>

**Proof**

### Examples of measurable function{#eg-msr-func}
<ul id='roman-list'>
	<li>
		Every continuous function $f:\mathbb{R}^d\to[0,+\infty]$ is measurable.
	</li>
	<li>
		Every unsigned simple function is measurable.
	</li>
	<li>
		The supremum, infimum, limit superior, or limit inferior of unsigned measurable functions is unsigned measurable.
	</li>
	<li>
		An unsigned function that is equal almost everywhere to an unsigned measurable function, is also measurable.
	</li>
	<li>
		If a sequence $f_n$ of unsigned measurable functions converges pointwise almost everywhere to an unsigned limit $f$, then $f$ is also measurable.
	</li>
	<li>
		If $f:\mathbb{R}^d\to[0,+\infty]$ is measurable and $\phi:[0,+\infty]\to[0,+\infty]$ is continuous, then $\phi\circ f:\mathbb{R}^d\to[0,+\infty]$ is measurable.
	</li>
	<li>
		If $f,g$ are unsigned measurable functions, then $f+g$ and $fg$ are measurable.
	</li>
</ul>

**Proof**


### Complex measurability{#cmplx-msrb}
An almost everywhere defined complex-valued function $f:\mathbb{R}^d\to\mathbb{C}$ is **Lebesgue measurable**, or **measurable**, if it is the pointwise almost everywhere limit of complex-valued simple functions.

### Equivalent notions of complex measurability{#equiv-ntn-cmplx-msrb}
Let $f:\mathbb{R}^d\to\mathbb{C}$ be an almost everywhere defined complex-valued function. The following are then equivalent:
<ul id='roman-list'>
	<li>
		$f$ is measurable.
	</li>
	<li>
		$f$ is the pointwise almost everywhere limit of complex-valued simple functions.
	</li>
	<li>
		The (magnitudes of the) positive and negative parts of $\text{Re}(f)$ and $\text{Im}(f)$ are unsigned measurable functions.
	</li>
	<li>
		$f^{-1}(U)$ is Lebesgue measurable for every open set $U\subset\mathbb{C}$.
	</li>
	<li>
		$f^{-1}(K)$ is Lebesgue measurable for every closed set $K\subset\mathbb{C}$.
	</li>
</ul>

**Proof**

## Unsigned Lebesgue integrals{#unsgn-lebesgue-int}

### Lower unsigned Lebesgue integral{#lwr-unsgn-lebesgue-int}
Let $f:\mathbb{R}^d\to[0,+\infty]$ be an unsigned functions (not necessarily measurable). We define the **lower unsigned Lebesgue integral**, denoted as $\underline{\int_{\mathbb{R}^d}}f(x)\hspace{0.1cm}dx$, to be the quantity
\begin{equation}
\underline{\int_\mathbb{R}^d}f(x)\hspace{0.1cm}dx\doteq\sup_{0\leq g\leq f;g\text{ simple}}\text{Simp}\int_{\mathbb{R}^d}g(x)\hspace{0.1cm}dx,
\end{equation}
where $g$ ranges over all unsigned simple functions $g:\mathbb{R}^d\to[0,+\infty]$ that are pointwise bounded by $f$.

We can also define the **upper unsigned Lebesgue integral** as
\begin{equation}
\overline{\int_\mathbb{R}^d}f(x)\hspace{0.1cm}dx\doteq\inf_{h\geq f;h\text{ simple}}\text{Simp}\int_{\mathbb{R}^d}h(x)\hspace{0.1cm}dx
\end{equation}

## Absolute integrability{#abs-intb}

## Littlewood's three principles{#littlewoods-prncpl}

## References
[1] <span id='taos-book'>Terence Tao. [An introduction to measure theory](https://terrytao.wordpress.com/books/an-introduction-to-measure-theory/). Graduate Studies in Mathematics, vol. 126.</span>

[2] <span id='steins-book'>Elias M. Stein & Rami Shakarchi. [Real Analysis: Measure Theory, Integration, and Hilbert Spaces](http://www.cmat.edu.uy/~mordecki/courses/medida2013/book.pdf). Princeton University Press, 2007.</span>

## Footnotes
[^1]: It should be simpler to consider the case of $k=2$, in particular with two sets $E_1,E_2\subset\mathbb{R}^d$. These two sets partition $\mathbb{R}^d$ into four disjoint sets: $E_1\cap E_2,E_1\cap E_2^c,E_1^c\cap E_2,E_1^c\cap E_2^c$.
