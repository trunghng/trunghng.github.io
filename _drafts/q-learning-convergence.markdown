---
layout: post
title:  "The Convergence of Q-learning"
date:   2022-08-21 07:00:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning q-learning dynamic-programming
description: Convergence of Q-learning, TD(lambda)
comments: true
---
> A note on convergence proofs for Q-learning by exploiting the connection with stochastic approximation and the idea of parallel asynchronous.
<!-- excerpt-end -->

- [Preliminaries](#preliminaries)
- [The convergence of Q-learning](#q-learning-convergence)
- [Preferences](#preferences)
- [Footnotes](#footnotes)

In Q-learning, transition probabilities and costs are unknown but information of them is obtained either by simulation or by experimenting. Q-learning uses simulation or experimental information to estimate the expected cost-to-go. Additionally, the algorithm is recursive and each new piece of information is used for computing an additive correction term to the old estimates. As these correction terms are random, Q-learning therefore has the same general structure as the stochastic approximation algorithms.


## Preliminaries
{: #preliminaries}

## The convergence of Q-learning
{: #q-learning-convergence}




## Preferences
{: #preferences}
[1] T. Jaakkola & M. I. Jordan & S. P. Singh. [On the Convergence of Stochastic Iterative Dynamic Programming Algorithms](doi: 10.1162/neco.1994.6.6.1185) in Neural Computation, vol. 6, no. 6, pp. 1185-1201, Nov. 1994. 

[2] Dvoretzky A. [On stochastic approximation](https://projecteuclid.org/proceedings/berkeley-symposium-on-mathematical-statistics-and-probability/Proceedings-of-the-Third-Berkeley-Symposium-on-Mathematical-Statistics-and/Chapter/On-Stochastic-Approximation/bsmsp/1200501645?tab=ArticleFirstPage). Berkeley Symposium on Mathematical Statistics and Probability, 1956: 39-55 (1956).

## Footnotes
{: #footnotes}