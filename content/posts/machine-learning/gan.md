---
title: "GAN"
date: 2023-08-13 13:00:00 +0700
tags: [machine-learning, generative-model]
math: true
eqn-number: true
draft: true
---
> Notes on Generative Adversarial Networks.
<!--more-->

## Generative Adversarial Networks (GAN){#gan}
A **generative adversarial network** consists of two components:
<ul id='number-list'>
	<li>
		<b>Generator</b>: A generative model, denoted $G$, parameterized by $\theta_G$
	</li>
	<li>
		<b>Discriminator</b>: A discriminative model, denoted $D$, parameterized by $\theta_D$
	</li>
</ul>
In the most trivial case, each model will be formulated as an MLP. The framework can be considered as a two-player minimax game in which we are trying to optimize the value function $V(D,G)$
\begin{equation}
\min_{\theta_G}\max_{\theta_D}V(D,G)=\mathbb{E}_{\mathbf{x}\sim p_\text{data}(\mathbf{x})}\big[\log D(\mathbf{x};\theta_D)\big]+\mathbb{E}_{\mathbf{z}\sim p_\mathbf{z}(\mathbf{z})}\big[\log\big(1-D(G(\mathbf{x};\theta_G);\theta_D)\big)\big]
\end{equation}


## Preferences
[1] Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio. [Generative Adversarial Nets](http://papers.neurips.cc/paper/5423-generative-adversarial-nets.pdf). NIPS, 2014.

[2] 

## Footnotes


