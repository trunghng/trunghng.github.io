---
layout: post
title:  "The Convergence of Q-learning"
date:   2022-08-21 07:00:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning Q-learning dynamic-programming
description: Convergence of Q-learning
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
[1] John N. Tsitsiklis. [Asynchronous Stochastic Approximation and Q-Learning](https://doi.org/10.1023/A:1022689125041). Machine Learning 16, 185–202 (1994).


## Footnotes
{: #footnotes}