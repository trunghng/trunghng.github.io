---
layout: post
title:  "My very first post"
date:   2021-06-05 17:00:00 +0700
categories: random stuffs
---
Enjoy my index-zero-ed post while staying tuned for next posts!

{% highlight python %}
from math import sqrt

def fibonacci(i)
	"""
	generate i-th number of the Fibonacci sequence, python code obvs :p
	"""
	return 1/sqrt(5)*(pow((1+sqrt(5))/2, i)-pow((1-sqrt(5))/2, i))
{% endhighlight %}

Why did numbers $$ \frac{1+\sqrt{5}}{2} $$ and $$ \frac{1-\sqrt{5}}{2} $$ come out of nowhere?  
In fact, these two numbers are eigenvalues of matrix $$A=\big(\begin{smallmatrix}1 & 1\\1 & 0\end{smallmatrix}\big)$$, which is retrieved from
\begin{equation}
u_{k+1}=Au_k,
\end{equation}
where $$ u_k=\big(\begin{smallmatrix}F_{k+1}\\F_k\end{smallmatrix}\big)$$.
And thus, $$ u_k=A^k u_0 $$.  
Then, the thing is, how can we compute $$A^k$$ quickly? This is where diagonalizing plays its role. Diagonalizing produces a factorization:
\begin{equation}
A=S\Lambda S^{-1},
\end{equation}
where $$S=\big(\begin{smallmatrix}x_1\dots x_n\end{smallmatrix}\big)$$ is eigenvector matrix, $$\Lambda=\big(\begin{smallmatrix}\lambda_1&&\\&\ddots&\\&&\lambda_n\end{smallmatrix}\big)$$ is a diagonal matrix established from eigenvalues of $$A$$.  
When taking the power of $$A$$,
\begin{equation}
A^k u_0=(S\Lambda S^{-1})\dots(S\Lambda S^{-1})u_0=S\Lambda^k S^{-1} u_0
\end{equation}
Writing $$u_0$$ as a combination $$c_1x_1+\dots+c_nx_n$$ of the eigenvectors, we have that $$c=S^{-1}u_0$$. Hence:
\begin{equation}
u_k=A^ku_0=c_1{\lambda_1}^kx_1+\dots+c_n{\lambda_n}^kx_n
\end{equation}
*Fact*: The $$\frac{1+\sqrt{5}}{2}\approx 1.618$$ is so-called "*golden ratio*". And for some reason a rectangle with sides 1.618 and 1 looks especially graceful.

