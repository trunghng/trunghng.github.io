---
title: "Graph generation with predefined chromatic number"
date: 2024-05-19T17:37:18+07:00
tags: [graph-theory]
math: true
eqn-number: true
---
> Generating predefined-chromatic-number graphs with Leighton's algorithm.
<!--more-->

## Leighton's algorithm
The algorithm is used to generate an $n$-vertex graph $G$ with $e$ edges and predefined chromatic number $k$ where $k\vert n$. This leads to the condition:
\begin{equation}
\frac{k(k-1)}{2}\leq e\leq\frac{n^2(k-1)}{2k}
\end{equation}
The algorithm begins by choosing positive integer $a,c,m$ such that
<ul class='roman-list'>
	<li>
		$m\gg n$
	</li>
	<li>
		$\gcd(n,m)=k$
	</li>
	<li>
		$\gcd(c,m)=1$
	</li>
	<li>
		If $p\vert m$ then $p\vert(a-1)$ for all primes $p$.
	</li>
	<li>
		If $4\vert m$ then $4\vert(a-1)$.
	</li>
</ul>

We then use the **linear congruential generator** method to generate a uniform sequence of random number $\\{x_i\\}$ on $[0,m-1]$. This can be done by beginning from a starting number $x_0$, and continuing with the following update rule:
\begin{equation}
x_i=(ax_{i-1}+c)\mod m
\end{equation}
The generated sequences has two crucial properties
<ul class='roman-list'>
	<li>
		For every $i,j$ such that $0\leq j\leq m-1$ and $i\geq 0$, there exists an $r$ such that $i\leq r\leq i+m-1$ and $x_r=j$. i.e., There is no duplication in every $m$ consecutive elements of the sequence.
	</li>
	<li>
		$x_i=x_{i+m}$ for every $i\geq 0$.
	</li>
</ul>

Since the random sequence $\\{x_i\\}$ we have got in $[0,m]$ and our vertices are in $[0,n-1]$ where $m\gg n$, we continue with construct $\\{y_i\\}$ in $[0,n-1]$ by letting
\begin{equation}
y_i=x_i\mod n
\end{equation}
Next, we define the $(k-1)$-vector
\begin{equation}
\mathbf{b}=(b_k,\ldots,b_2)
\end{equation}
so that $b_k\geq 1$ and $b_i\geq 0$ for $2\leq i\leq k-1$ and each $b_i$ corresponds to the number of [$i$-cliques]({{<ref"pgm-representation#clique">}}) to be implanted in $G$.

Given $\\{y_i\\}$ and $\mathbf{b}$, we proceed as follows.
<ul class='number-list'>
	<li>
		Select the first $k$ values of $\{y_i\}$ beginning with $y_1$ and add the corresponding edges to $E$.
	</li>
	<li>
		If $b_k>1$, select the next $k$ values of $\{y_i\}$ and add the corresponding edges to $E$.
	</li>
	<li>
		Repeat (1), (2) until $b_k$ $k$-cliques have been implanted in $G$.
	</li>
	<li>
		Add, in an identical fashion, $b_{k-1}$ $(k-1)$-cliques to $G$.
	</li>
	<li>
		Continue the process until $b_2$ $2$-cliques (edges) have been added to $E$.
	</li>
</ul>

## References
[1] Leighton, Frank Thomson. [A Graph Coloring Algorithm for Large Scheduling Problems](https://doi.org/10.6028/jres.084.024). Journal of research of the National Bureau of Standards 84 6 (1979): 489-506.
