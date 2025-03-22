![Decision Tree](img/Decision%20Trees/img_01.avif)


## Overview
A Decision Tree (DT) is a machine learning model that is created by a divide and conquer algorithm.  They are great for its quick computation and interpretability.

In a DT, the root and intermediate nodes correspond to a feature in a feature set having each edge indication a decision, or a value an example can take on for a feature (categorical values or bins for numerical values). The leaf nodes represent your target values (classes in a classification task and binned numerical values for regression). 

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
For every split (branching of each intermediate node), we must select some formula to decide what examples go to each branch.  Each formula must maximize __purity__, which how well we partition classes in each node.

There are three types of split formulas we will discuss

1. Information Gain
2. Gain Ratio
3. Gini Index

### Information Gain
Before we can talk about Information Gain, we must talk about Information Entropy, or entropy for short. Entropy is a great measure of purity.

Let $p_{k}$ denote the proportion of examples in the $k^{\text{th}}$ class of our dataset $D$, where $k = 1, 2, \ldots, |y|$, and $|y|$ is the total number of distinct classes.  Then entropy is defined as 

$$
Ent(D) = - \sum^{|y|}_{k=1}{p_{k}\log_{2}{p_{k}}}
$$

where

$$
0 \leq Ent(D) \leq \log_{2}{|y|}
$$

The lower value for $Ent(D)$, the higher the purity of $D$.  If we have a discrete feature, $a$, with $V$ possible values $\{ a^{0}, a^{2}, \ldots, a^{V} \}$, then when making a decision tree, we can create $V$ child nodes from this feature's node.  We can then compute the proportion of examples in $D$ that are placed in each child node, $\frac{|D^{v}|}{|D|}$, and find its proportion.  This proportion explains the importance of this node in the model.  This proportion will also allow us to calculate the information gain from this node.

The information gain of a node is defined as

$$
Gain(D, a) = Ent(D) - \sum^{V}_{v=1}{\frac{|D^{V}|}{|D|}Ent(D^{v})}
$$

For some feature $a$ and dataset $D$.

Remember, the lower entropy for a dataset $Ent(D)$, the higher the purity of $D$.  Therefore, the higher the information gain for a feature, the more purity we can achieve when splitting $D$ by that feature. Therefore when creating a DT, for each node, we aim to find $a_{*}$, such that

$$
a_{*} = \underset{a \in A}{\arg\max} \text{ } Gain(D, a)
$$



### Gain Ratio
### Gini Index
## Pruning
## Continuous and Missing Values
## Multivariate Decision Trees
