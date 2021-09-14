---
layout: post
title:  "Temporal-Difference Learning. Deep-Q Network"
date:   2021-09-14 18:16:00 +0700
categories: artificial-intelligent reinforcement-learning
tags: artificial-intelligent reinforcement-learning td-learning q-learning dqn my-rl
description: Temporal-Difference Learning, Q-learning, Deep-Q Network
comments: true
---
> So far in this [series](/tag/my-rl), we have gone through ideas of [**dynamic programming** (DP)]({% post_url 2021-07-25-dp-in-mdp %}) and [**Monte Carlo**]({% post_url 2021-08-21-monte-carlo-in-rl %}). What will happen if we combine these ideas together? **Temporal-deffirence (TD) learning** is our answer.

<!-- excerpt-end -->


- [References](#references)
- [Footnotes](#footnotes)


## References
[1] Richard S. Sutton & Andrew G. Barto. [Reinforcement Learning: An Introduction](https://mitpress.mit.edu/books/reinforcement-learning-second-edition)  

[2] David Silver. [UCL course on RL](https://www.davidsilver.uk/teaching/)  

[3] Mnih, V., Kavukcuoglu, K., Silver, D. et al. [Human-level control through deep reinforcement learning](https://doi.org/10.1038/nature14236). Nature 518, 529–533 (2015). 


## Footnotes
