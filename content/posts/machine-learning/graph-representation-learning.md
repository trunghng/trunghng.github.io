---
title: "Graph Representation Learning"
date: 2024-04-16T14:43:59+07:00
tags: [machine-learning, neural-network, graph-neural-network, graph-theory]
math: true
eqn-number: true
hideSummary: true
---

## Traditional feature-based approaches
In traditional Machine Learning methods, we first represent our data points (nodes, links, entire graphs) as vectors of (hand-designed) features and then train a classical ML model (random forest, SVM, neural network) on top of that.

Let us consider a network $\mathcal{G}=(\mathcal{V},\mathcal{E})$, where $\mathcal{V}$ is the set of nodes and $\mathcal{E}$ is the set of edges between these nodes. Also let $\mathbf{A}\in\mathbb{R}^{\vert\mathcal{V}\vert\times\vert\mathcal{V}\vert}$ be the adjacency matrix of $\mathcal{G}$, i.e. for $u,v\in\mathcal{V}$
\begin{equation}
\mathbf{A}\_{u,v}=\begin{cases}1&\text{if }(u,v)\in\mathcal{E} \\\\ 0&\text{if }(u,v)\notin\mathcal{E}\end{cases}
\end{equation}
In some graph with weighted edges, entries of $\mathbf{A}$ will be arbitrary real-values rather than $0$ or $1$.

### Node-level features
The goal of designing node features is to characterize the structures and positions of nodes in the network. For every node $u\in\mathcal{V}$, we have
<ul class='number-list'>
	<li>
		<b>Node degree</b>. Denoted as $d_u$, it counts the number of neighbors of $u$.
		\begin{equation}
		d_u=\sum_{v\in\mathcal{V}}\mathbf{A}_{u,v}
		\end{equation}
	</li>
	<li>
		<b>Node centrality</b>. Denoted as $c_u$, it takes the node importance into account.
		<ul>
			<li>
				<b>Eigenvector centrality</b>. It captures how importance the neighbors of $u$ are, i.e. $u$ is important if it is surrounded by important neighboring nodes.
				\begin{equation}
				c_u=\frac{1}{\lambda}\sum_{v\in\mathcal{V}}\mathbf{A}_{u,v}c_v
				\end{equation}
				where $\lambda$ is some constant. Rewriting the above equation in vector form with $\mathbf{c}$ denotes the centrality vector we have
				\begin{equation}
				\lambda\mathbf{c}=\mathbf{A}\mathbf{c},
				\end{equation}
				which is the standard eigenvector equation for the adjacency matrix $\mathbf{A}$. By further assuming that we only use positive centrality, we can  replace the constant $\lambda$ by the largest eigenvector $\lambda_\text{max}$, which by <b>Perron-Frobenius theorem</b> is proved to be positive and unique. And the centrality vector $\mathbf{c}$ is given by the eigenvector corresponding to $\lambda_\text{max}$.
			</li>
			<li>
				<b>Betweenness centrality</b>. It measures how often $u$ lies on the shortest path between other nodes.
				\begin{equation}
				c_u=\sum_{s\neq t\neq u}\frac{\text{#(shortest paths between }s\text{ and }t\text{ containing }u\text{)}}{\text{#(shortest paths between }s\text{ and }t\text{)}}
				\end{equation}
			</li>
			<li>
				<b>Closeness centrality</b>. It measures the average shortest path length between $u$ and all other nodes.
				\begin{equation}
				c_u=\frac{\sum_{v\neq u}\text{shortest path length between }u\text{ and }v}{\vert\mathcal{V}\vert-1}
				\end{equation}
			</li>
		</ul>
	</li>
	<li>
		<b>Clustering coefficient</b>.
	</li>
	<li>
		<b>Graphlets</b>.
	</li>
</ul>

#### Graph Laplacians and Spectral Methods
Spectral methods are used in clustering the nodes in a graph. We first begin with the definition of some important matrices.

##### Graph Laplacians
In addition to adjacency matrices, **Laplacians** are another matrix representations that can represent the matrix without loss of information.

###### Unnormalized Laplacian
The (unnormalized) Laplacian matrix is defined as
\begin{equation}
\mathbf{L}=\mathbf{D}-\mathbf{A},
\end{equation}
where $\mathbf{D}$ is the degree matrix. These are some properties corresponding to the Laplacian matrix $\mathbf{L}$ of a simple graph
<ul class='roman-list'>
	<li>
		It is symmetric and positive semi-definite.
	</li>
	<li>
		This holds for all $\mathbf{x}\in\mathbb{R}^{\vert\mathcal{V}\vert}$.
		\begin{equation}
		\mathbf{x}^\text{T}\mathbf{L}\mathbf{x}=\frac{1}{2}\sum_{u\in\mathcal{V}}\sum_{v\in\mathcal{V}}\mathbf{A}_{u,v}(\mathbf{x}_u-\mathbf{x}_v)^2=\sum_{(u,v)\in\mathcal{E}}(\mathbf{x}_u-\mathbf{x}_v)^2
		\end{equation}
	</li>
</ul>

**Theorem 1**. *The geometric multiplicity of the $0$ eigenvalue of the Laplacian $\mathbf{L}$ corresponds to the number of connected components in the graph.*

###### Normalized Laplacian
The symmetric normalized Laplacian is given by
\begin{equation}
\mathbf{L}\_\text{sym}=\mathbf{D}^{-\frac{1}{2}}\mathbf{L}\mathbf{D}^{-\frac{1}{2}}
\end{equation}
while the random walk Laplacian is defined as
\begin{equation}
\mathbf{L}\_\text{RW}=\mathbf{D}^{-\frac{1}{2}}\mathbf{L}
\end{equation}
Both of these matrices have similar properties as the (unnormalized) Laplacian, but their algebraic properties differ by small constants due to the normalization.

##### Graph Cuts and Clustering
We can see that Theorem 1 can be used to assign nodes to clusters based on which connected component they belong to. This method is trivial since it only lets us cluster nodes that are already in disconnected components. Fortunately, we will be showing that Laplacian can be applied to give an optimal clustering of nodes within a fully connected graph.

In order to define what an optimal cluster means, we begin with the notion of a **cut** on a graph.

###### Graph cuts
Let $\mathcal{A}\subset\mathcal{V}$ be a subset of the nodes in the graph and $\bar{\mathcal{A}}$ denote its complement. Given a partitioning of the graph in $K$ non-overlapping subsets $\mathcal{A}\_1,\ldots,\mathcal{A}\_K$ we define the cut value of this partition as the number of edges crossing the boundary between the partition of nodes
\begin{equation}
\text{cut}(\mathcal{A}\_1,\ldots,\mathcal{A}\_K)=\frac{1}{2}\sum_{k=1}^{K}\big\vert(u,v)\in\mathcal{E}:u\in\mathcal{A}\_k,v\in\bar{\mathcal{A}}\_k\big\vert
\end{equation}
Here, it is possible to define an optimal clustering of the nodes into $K$ clusters as selecting a partition that minimizes this cut value. However, this approach also tends to give clusters containing only a single node.

To overcome this, beside minimizing the cut we also try to make the partitions are all reasonably large as well. We could do this by minimizing the **Ratio Cut**
\begin{equation}
\text{RatioCut}(\mathcal{A}\_1,\ldots,\mathcal{A}\_K)=\frac{1}{2}\sum_{k=1}^{K}\frac{\big\vert(u,v)\in\mathcal{E}:u\in\mathcal{A}\_k,v\in\bar{\mathcal{A}}\_k\big\vert}{\vert\mathcal{A}\_k\vert},
\end{equation}
which penalizes the solution for choosing small cluster sizes. Or we can instead minimize the **Normalized Cut (NCut)**
\begin{equation}
\text{NCut}(\mathcal{A}\_1,\ldots,\mathcal{A}\_K)=\frac{1}{2}\sum_{k=1}^{K}\frac{\big\vert(u,v)\in\mathcal{E}:u\in\mathcal{A}\_k,v\in\bar{\mathcal{A}}\_k\big\vert}{\sum_{u\in\mathcal{A}\_k}d_u},
\end{equation}
which on the other hand enforces that all clusters have the same number of edges incident to their nodes.

###### Approximating the RatioCut with the Laplacian spectrum

##### Generalized spectral clustering
<ul class='number-list'>
	<li>
		Find eigenvectors corresponding to $K$ smallest eigenvalues of $\mathbf{L}$ with the smallest one excluded.
	</li>
	<li>
		Form the matrix $\mathbf{U}\in\mathbb{R}^{\vert\mathcal{V}\times(K-1)\vert}$ with the eigenvectors obtained from (1) as columns.
	</li>
	<li>
		Choose each row of $\mathbf{U}$ as the embedding for a corresponding node from $\mathcal{V}$
		\begin{equation}
		\mathbf{z}_u=\mathbf{U}_u\hspace{1cm}\forall u\in\mathcal{V}
		\end{equation}
	</li>
	<li>
		Run $K$-means clustering on the feature vectors $\mathbf{z}_u,\forall u\in\mathcal{V}$.
	</li>
</ul>

### Link-level features
The goal of designing edge features is to quantify the relationships between nodes. Let $\mathbf{S}\in\mathbb{R}^{\vert\mathcal{V}\vert\times\vert\mathcal{V}\vert}$ denote the similarity matrix, where each entry $S_{u,v}$ denotes the value quantifying the relationship between nodes $u,v\in\mathcal{V}$.
<ul class='number-list'>
	<li>
		<b>Distance-based feature</b>. It measures the shortest path distance between two nodes
	</li>
	<li>
		<b>Local neighborhood overlap</b>. It counts the number of common neighbors shared by two nodes
		<ul>
			<li>
				<b>Common neighbors</b>. Which is the naive function
				\begin{equation}
				S_{u,v}=\vert\mathcal{N}(u)\cap\mathcal{N}(v)\vert
				\end{equation}
			</li>
			<li>
				<b>Sorenson index</b>. It defines a matrix $\mathbf{S}_\text{Sorenson}\in\mathbb{R}^{\vert\mathcal{V}\vert\times\vert\mathcal{V}\vert}$ of node-node neighborhood overlaps with entries given by
				\begin{equation}
				S_{u,v}=\frac{2\vert\mathcal{N}(u)\cap\mathcal{N}(v)\vert}{d_u+d_v}
				\end{equation}
			</li>
			<li>
				<b>Salton index</b>.
				\begin{equation}
				S_{u,v}=\frac{2\vert\mathcal{N}(u)\cap\mathcal{N}(v)\vert}{\sqrt{d_u d_v}}
				\end{equation}
			</li>
			<li>
				<b>Jaccard index</b>.
				\begin{equation}
				S_{u,v}=\frac{\vert\mathcal{N}(u)\cap\mathcal{N}(v)\vert}{\vert\mathcal{N}(u)\cup\mathcal{N}(v)\vert}
				\end{equation}
			</li>
			<li>
				<b>Resource Allocation index</b>. It counts the inverse degrees of the common neighbors.
				\begin{equation}
				S_{v_1,v_2}=\sum_{u\in\mathcal{N}(v_1)\cap\mathcal{N}(v_2)}\frac{1}{d_u}
				\end{equation}
			</li>
			<li>
				<b>Adamic-Adar index</b>. It counts the inverse logarithm of the degrees of the common neighbors.
				\begin{equation}
				S_{v_1,v_2}=\sum_{u\in\mathcal{N}(v_1)\cap\mathcal{N}(v_2)}\frac{1}{\log d_u}
				\end{equation}
			</li>
		</ul>
	</li>
	<li>
		<b>Global neighborhood overlap</b>. The local neighborhood overlap always returns zero when two nodes have no common neighbor even if they are potentially connected in the future. The global metric resolves this limitation by taking the entire graph into account.
		<ul>
			<li>
				<b>Katz index</b>. It counts the number of walks of all lengths between two nodes
				\begin{equation}
				S_{u,v}=\sum_{l=1}^{\infty}\beta^l\mathbf{A}_{u,v}^l,
				\end{equation}
				where $0<\beta<1$ is the discount factor and each $\mathbf{A}_{u,v}^l$ is an entry of the power matrix $\mathbf{A}^l$.<br><br>
				To compute the number of walks between two nodes, we consider the number of walks of each length. Let $\mathbf{P}^{(k)}$ be a matrix where each entry $\mathbf{P}_{u,v}^{(k)}$ denote the number of walks of length $k$ between a pair of nodes $(u,v)$. We will show that this matrix is also the power of $k$ of $\mathbf{A}$.
				\begin{equation}
				\mathbf{P}^{(k)}=\mathbf{A}^k
				\end{equation}
				It is easily seen that
				\begin{equation}
				\mathbf{P}_{u,v}^{(1)}=\mathbf{A}_{u,v}
				\end{equation}
				Let us continue by considering $\mathbf{P}_{u,v}^{(2)}$. This can be calculated by first computing the number of walks between each of $u$'s neighbors and $v$ then summing them up across $u$'s neighbors
				\begin{equation}
				\mathbf{P}_{u,v}^{(2)}=\sum_i\mathbf{A}_{u,i}\mathbf{P}_{i,v}^{(1)}=\sum_i\mathbf{A}_{u,i}\mathbf{A}_{i,v}=\mathbf{A}_{u,v}
				\end{equation}
				We can keep doing this to show that it is true that $\mathbf{P}_{u,v}^{(k)}=\mathbf{A}_{u,v}^k$. Or in other words, $\mathbf{A}_{u,v}^k$ is also the number of walks of length $k$ between $u$ and $v$.<br><br>
				<b>Theorem 2</b>. <i>Let $\mathbf{X}$ be a real-valued diagonalizable square matrix and let $\lambda_\text{max}$ denote the largest eigenvalue of $\mathbf{X}$. Then</i>
				\begin{equation}
				(\mathbf{I}-\mathbf{X})^{-1}=\sum_{i=0}^{\infty}\mathbf{X}^i
				\end{equation}
				<i>iff $\vert\lambda_\text{max}\vert<1$ and $(\mathbf{I}-\mathbf{X})$ is non-singular.</i><br><br>
				<b>Proof</b>. Let $s_n=\sum_{i=0}^{n}\mathbf{X}^i$, we have
				\begin{align}
				s_n-\mathbf{X}s_n&=\sum_{i=0}^{n}\mathbf{X}^i-\mathbf{X}\sum_{i=0}^{n}\mathbf{X}^i \\ s_n(\mathbf{I}-\mathbf{X})&=\mathbf{I}-\mathbf{X}^{n+1} \\ s_n&=(\mathbf{I}-\mathbf{X}^{n+1})(\mathbf{I}-\mathbf{X})^{-1}\label{eq:llf.1}
				\end{align}
				Consider the eigendecomposition of $\mathbf{X}$, we have
				\begin{equation}
				\mathbf{X}=\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1},
				\end{equation}
				which leads us to
				\begin{equation}
				\mathbf{X}^n=(\mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^{-1})^n=\mathbf{Q}\mathbf{\Lambda}^n\mathbf{Q}^{-1}=\mathbf{Q}\left[\begin{matrix}\lambda_1^n&& \\ &\ddots& \\ &&\lambda_D^n\end{matrix}\right]\mathbf{Q}^{-1}
				\end{equation}
				Hence if $\vert\lambda_\text{max}\vert<1$, as $n\to\infty$, we have that $\mathbf{\Lambda}^n\to 0$, and thus $\mathbf{X}^n\to 0$. Applying this result into \eqref{eq:llf.1} gives us
				\begin{equation}
				\lim_{n\to\infty}s_n=\lim_{n\to\infty}(\mathbf{I}-\mathbf{X}^{n+1})(\mathbf{I}-\mathbf{X})^{-1}=\mathbf{I}(\mathbf{I}-\mathbf{X})^{-1}=(\mathbf{I}-\mathbf{X})^{-1}
				\end{equation}
				Based on Theorem 2, we have the closed-form of Katz index is given by
				\begin{equation}
				\mathbf{S}_\text{Katz}=(\mathbf{I}-\beta\mathbf{A})^{-1}-\mathbf{I}
				\end{equation}
			</li>
			<li>
				<b>Leicht, Holme and Newman (LHN) similarity</b>.
			</li>
			<li>
				<b>Random walk methods</b>.
			</li>
		</ul>
	</li>
</ul>

### Graph-level features

## Node Embeddings
The goal of node embedding learning methods is to encode nodes as low-dimensional vectors that summarize their graph position and the structure of their local graph neighborhood. Or in other words, nodes are encoded so that the similarity in the latent space approximates the similarity in the original graph.

### Encoder-Decoder framework
In this framework, the graph representation learning problem is divided into steps
<ul class='number-list'>
	<li>
		An <b>encoder</b> maps each node in the original graph to a low-dimensional vector in the embedding space.
	</li>
	<li>
		Define a similarity function in the original graph.
	</li>
	<li>
		A <b>decoder</b> takes embedding vectors and use them to reconstruct information about each node's neighborhood in the original graph.
	</li>
	<li>
		Optimize the encoder and the decoder so that the similarity in the embedding space approximates the similarity in the original graph.
	</li>
</ul>

#### The Encoder
The encoder, denoted $\text{ENC}$, is a function that maps each node $v\in\mathcal{V}$ to an embedding vector $\mathbf{z}_v\in\mathbb{R}^d$.
\begin{equation}
\text{ENC}(v)=\mathbf{z}\_v
\end{equation}
In the simplest form, used in the **shallow embedding** approach, the encoder is simply an embedding-lookup
\begin{equation}
\text{ENC}(v)=\mathbf{z}\_v=\mathbf{Z}\mathbf{v},
\end{equation}
where $Z\in\mathbb{R}^{d\times\vert\mathcal{V}\vert}$ is a matrix whose each column is a node embedding and where each $\mathbf{v}\in\mathbb{R}^{\vert\mathcal{V}\vert}$ is an indicator vector (i.e., all zeroes except a one in column indicating the ID of node $v$).

#### The Decoder
The decoder, denoted $\text{DEC}$, reconstruct certain graph statistics (e.g., set of neighbors $\mathcal{N}(u)$) from the embedding $\mathbf{z}_u$ generated by $\text{ENC}$ of node $u$.

A pairwise decoder, $\text{DEC}:\mathbb{R}^d\times\mathbb{R}^d\mapsto\mathbb{R}^+$ maps each pair of embeddings $(\mathbf{z}_u,\mathbf{z}_v)$ to a similarity score, which describes the relationship between nodes $u$ and $v$.

Given this similarity score, our goal is to optimize the encoder and decoder so that the decoded similarity approximates the similarity in the original graph.
\begin{equation}
\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v)\approx\mathbf{S}(u,v),\label{eq:td.1}
\end{equation}
where $\mathbf{S}(u,v)$ is a graph-based similarity metric between nodes $u$ and $v$.

#### Model optimization
The reconstruction objective \eqref{eq:td.1} can be accomplished by minimizing an empirical reconstruction loss $\mathcal{L}$ over a set of training node pairs $\mathcal{D}$
\begin{equation}
\mathcal{L}=\sum_{(u,v)\in\mathcal{D}}\ell(\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v),\mathbf{S}(u,v)),
\end{equation}
where $\ell:\mathbb{R}\times\mathbb{R}\mapsto\mathbb{R}$ is a loss function measures the difference between the decoded similarity values $\text{DEC}(\mathbf{z}_u,\mathbf{z}_v)$ and the true similarity values $\mathbf{S}(u,v)$.

### Factorization-based approaches

#### Laplacian eigenmaps
In this approach, the decoder is defined as the L2-distance between the embeddings
\begin{equation}
\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v)=\Vert\mathbf{z}\_u-\mathbf{z}\_v\Vert_2^2
\end{equation}
And the loss function is then given as
\begin{equation}
\mathcal{L}=\sum_{(u,v)\in\mathcal{D}}\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v)\cdot\mathbf{S}(u,v)
\end{equation}

#### Inner-product methods
As suggested by their name, the decoder in these approaches is defined as the inner product
\begin{equation}
\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v)=\mathbf{z}\_u^\text{T}\mathbf{z}\_v
\end{equation}
These methods have the loss function given as
\begin{equation}
\mathcal{L}=\sum_{(u,v)\in\mathcal{D}}\Vert\text{DEC}(\mathbf{z}\_u,\mathbf{z}\_v)-\mathbf{S}(u,v)\Vert_2^2
\end{equation}
The above approaches are referred to as matrix-factorization methods, since their loss function can be minimized using factorization algorithm, such as SVD. Stacking the embeddings $\mathbf{z}_u\in\mathbb{R}^d$ into a matrix $\mathbf{Z}\in\mathbb{R}^{\vert\mathcal{V}\vert\times d}$ the reconstruction objective can be rewritten as
\begin{equation}
\mathcal{L}\approx\Vert\mathbf{Z}\mathbf{Z}^\text{T}-\mathbf{S}\Vert_2^2,
\end{equation}
where $\mathbf{S}$ is a matrix containing pairwise similarity measures.

### Random walk embeddings

## Graph Neural Networks

### Graph Convolution Networks

## References
[1] William L. Hamilton. [Graph Representation Learning](https://www.cs.mcgill.ca/~wlh/grl_book/). Morgan and Claypool, Synthesis Lectures on Artificial Intelligence and Machine Learning.

[2] Jure Leskovec. [CS224W - Machine Learning with Graphs](https://web.stanford.edu/class/cs224w/).

## Footnotes