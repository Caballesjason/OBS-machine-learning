![Decision Tree](img/Decision%20Trees/img_01.avif)


## Overview
A Decision Tree (DT) is a machine learning model that is created by a divide and conquer algorithm.  They are great when you need quick computation and interpretability.

In a DT, the root and intermediate nodes correspond to a feature in a feature set, with it's edges indicating possible values for that feature. The leaf nodes represent your target values (classes in a classification task and binned numerical values for regression). 

|              When to use Decision Trees               |             When Not to use Decision Trees              |
| :---------------------------------------------------: | :-----------------------------------------------------: |
|     When your data shows non-linear relationships     |       If the data is small with high variablility       |
|   When working with small or medium sized datasets    | If the data contains too many irrelevant/noisy features |
|         When there are missing values present         | When you have high dimensional data; e.g. text, images  |
| When there are significant interaction with features  |             When you probabilistic outputs              |
| If you require fast prediction times for applications |             When your data is highly Sparse             |

## Basic process
Below is the algorithm used to train a decision tree.

__Decision Tree Algorithm__: 

__Inputs:__
	Training set $D = \{ (x_{0}, y_{0}), (x_{1}, y_{1}), \ldots (x_{m}, y_{m}) \}$ 
	 Feature set $A = \{ a_{1}, a_{2}  \ldots a_{d} \}$
	
__Process:__
	$\text{Function } TreeGenerator(D, A$)
	Generate node $i$;
	__IF__ $A = \emptyset$ __OR__ all samples in $D$ take the same value on $A$ __THEN__
		Mark node $i$ as class $C$

%% Complete algorithm later using the pseudocode plugin%%
## Split Selection
Before we decide what feature to use for a node, we need to decide on a formula to use to select the feature that maximizes __purity__, which how well we partition target values in each node.

There are three types of split formulas we will discuss

1. Information Gain
2. Gain Ratio
3. Gini Index

### Information Gain
Before we can talk about Information Gain, we must talk about Information Entropy, or Entropy for short. Entropy is a great measure of purity.

Let $p_{k}$ denote the proportion of examples in the $k^{\text{th}}$ class of our dataset $D$, where $k = 1, 2, \ldots, |\mathcal{y}|$, and $|y|$ is the total number of distinct classes.  Then entropy is defined as 

$$
\begin{equation}
Ent(D) = - \sum^{|\mathcal{y}|}_{k=1}{p_{k}\log_{2}{p_{k}}}
\end{equation}
$$

where

$$
\begin{equation}
0 \leq Ent(D) \leq \log_{2}{|\mathcal{y}|}
\end{equation}
$$

The lower the value for $Ent(D)$, the higher the purity of $D$.  If we have a discrete feature, $a$, with $V$ possible values $\{ a^{0}, a^{2}, \ldots, a^{V} \}$, then we can create $V$ child nodes from this feature's node.  We can then compute the proportion of examples in $D$ that are placed in each child node, $\frac{|D^{v}|}{|D|}$.  This proportion explains the importance of this node in the model and will allow us to calculate the information gain from this node.

The information gain of a node is defined as

$$
\begin{equation}
Gain(D, a) = Ent(D) - \sum^{V}_{v=1}{\frac{|D^{V}|}{|D|} \cdot Ent(D^{v})}
\end{equation}
$$

For some feature $a$ and dataset $D$.

Remember, the lower the entropy for a dataset, $Ent(D)$, the higher the purity of $D$.  Therefore, the higher the information gain for a feature, the more purity we can obtain when splitting $D$ by that feature. Therefore when creating a DT, for each node, we aim to find $a_{*}$, or the feature that maximizes information gain.  Mathematically, this is denoted as

$$
a_{*} = \underset{a \in A}{\arg\max} \text{ } Gain(D, a)
$$

__Note:__
> Information Gain is biased towards features that contain a lot of possible values.  If a feature has a lot of values, then the proportion of examples being placed in each node, $\frac{|D^{v}|}{|D|}$, will be very small.  This will lead to a high purity.

### Gain Ratio
To handle the bias obtained from Information Gain, we can use a Gain Ratio.  The Gain Ratio is defined as 

$$
\begin{equation}
Gain\_Ratio(D, a) = \frac{Gain(D, a)}{IV(a)}
\end{equation}
$$

where

$$
\begin{equation}
IV(a) = -\sum_{v=1}^{V}{\frac{|D^{V}|}{|D|} \log_{2} \frac{|D^{V}|}{|D|}}
\end{equation}
$$

$IV$ is short for _Intrinsic Value_ of a feature $a$.  $IV(a)$ is large when feature $a$ has many possible values and small when $a$ has only a few possible values.

__Note:__
> Gain Ratio is biased towards features that contain only a few possible values.  If a feature has only a few values, then the proportion of examples being placed in each node, $\frac{|D^{v}|}{|D|}$, will be very small, making the Gain Ratio small.


### Gini Index
The _Gini Value_ of a dataset, $D$, represents the likelihood that two randomly selected examples from $D$ belong to different classes.  Mathematically, the Gini Value is defined as

$$
\begin{equation}
Gini(D) = \sum_{k = 1}^{|\mathcal{y}|}{\sum_{k' \neq k}{p_{k}p'_{k}} = 1 - \sum_{k=1}^{|\mathcal{y}|}{p^{2}_{k}}}
\end{equation}
$$

The _Gini Index_ split selection formula for DTs is defined as

$$
\begin{equation}
Gini\_Index(D, a) = \sum_{v=1}^{V}{\frac{|D^{V}|}{|D|} \cdot Gini(D)}
\end{equation}
$$

The Gini Index can be used in both cases of features that can have a lot of values or features that can only have few values.

## Pruning
The recursive nature of the DT training makes them very prone to overfitting.  Training a DT doesn't stop until all examples are classified, therefore we introduce pruning which is the action of removing branches depending on the improvement of purity.

There are two types of pruning strategies, pre-pruning and post-pruning.

### Pre-pruning
  When pre-pruning a tree, the model is evaluated for some performance metric at a given node. We then split the node to its potential child nodes, and then we evaluate if these nodes improve the model.  If the model's performance improves, then the child nodes stay in the DT.  If the model's performance does not improve, then the child nodes are connected to the tree.   If the child nodes do connect to the DT, this process gets repeated for each child.

Since this process avoids training and traversing to future possible children, we are less prone to overfitting.  However, there is a chance that subsequent child node will yield better performance.  In other words, pre-pruning is susceptible to underfitting.

Here is a visual example of the pre-pruning process

![Pre-pruning Example](img/Decision%20Trees/img_02.png)


### Post-pruning
When post-pruning, the entire tree is trained, then branches are pruned if the model's performance improves.  The model performance is evaluated on using a validation set (cross validation or leave one out validation).  In other words, we will consider each intermediate node, starting from the parent node of a leaf node.  If pruning the parent node (making the parent node a leaf node) does not reduce the model validation performance, then the parent node gets pruned.  We then continue traversing the DT's intermediate nodes until we obtain the best performing model.

While post-pruning is less susceptible to underfitting and typically generalizes a DT better than pre-pruning, it is also more computationally expensive since it must first train the model, then traverse the tree in a bottom-up fashion.

## Continuous and Missing Values
## Multivariate Decision Trees
