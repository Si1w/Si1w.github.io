---
title: "Data Splits, Models & Cross-Validation"
date: "2025-01-05 19:21:00"
categories: 
    - Machine Learning
    - CS229
tags: 
    - Machine Learning
    - Stanford
    - CS229
mathjax: true
---
# Bias

**High Bias:** When a model has high bias, it means that it is too simple and does not capture the underlying structure of the data well. (**Underfitting**)

**Low Bias:** When a model has low bias, it means that it does capture the underlying structure of the data well. (**Overfitting**)

# Regularizaion

## L2 Regularization

L2 Regularization is a technique used to prevent overfitting in a model. It does this by adding a penalty term to the cost function. This penalty term is the sum of the squares of the weights. The cost function with L2 Regularization is given by:

$$
\min_{\theta} \sum_{i=1}^{m} \Vert y^{(i)} - \theta^T x^{(i)} \Vert^2 + \frac{\lambda}{2} \Vert \theta \Vert^2
$$

or more generally:

$$
arg\min_{\theta} \sum_{i=1}^{m} L(y^{(i)}, \theta^T x^{(i)}) + \frac{\lambda}{2} \Vert \theta \Vert^2
$$

where $L(y^{(i)}, \theta^T x^{(i)})$ is the loss function.

## L1 Regularization

L1 Regularization is a technique used to feature select in a model. It does this by adding a penalty term to the cost function. This penalty term is the sum of the absolute values of the weights. The cost function with L1 Regularization is given by:

$$
\min_{\theta} \sum_{i=1}^{m} \Vert y^{(i)} - \theta^T x^{(i)} \Vert^2 + \frac{\lambda}{2} \vert \theta_{i} \vert
$$

# Train/Dev/Test Splits

Suppose we have a dataset $S$, which we split into three parts: $S_{train}$, $S_{dev}$, and $S_{test}$. 

- Train each model $i$ (option for degree of polynomial) on $S_{train}$. Get some hypothesis $h_{i}$.

- Measure the error of $h_{i}$ on $S_{dev}$. Pick the model with the lowest error on $S_{dev}$.

- Evaluate the model on $S_{test}$ to estimate the generalization error.

# Model Selection & Cross-Validation

## k-Fold Cross-Validation

Suppose we have a train set $S_{train} = \{ (x^{(1)}, y^{(1)}), \ldots, (x^{(100)}, y^{(100)}) \}$.

Let $k = 5$ (but $k=10$ is typical).

1. Divide $S_{train}$ into $k$ equal-sized subsets $S_{1}, \ldots, S_{k}$.

2. For $i = 1, \ldots, k$:
    - Train on $k-1$ pieces.
    - Test on the remaining $1$ pieces ($S_{i}$).

## Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation is a special case of k-Fold Cross-Validation where $k = m$ (the number of training examples).