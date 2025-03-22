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

> __Inputs:__
> Training set $D = \{ (x_{0}, y_{0}), (x_{1}, y_{1}), \ldots (x_{m}, y_{m}) \}$ 
> 
> Feature set $A = \{ a_{1}, a_{2}  \ldots a_{d} \}$
> 
> __Process:__



## Split Selection
## Pruning
## Continuous and Missing Values
## Multivariate Decision Trees
