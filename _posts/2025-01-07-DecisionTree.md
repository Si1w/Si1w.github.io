---
title: "Decision Tree"
date: "2025-01-07 19:59:00"
categories: 
    - Machine Learning
    - CS229
tags: 
    - Machine Learning
    - Stanford
    - CS229
mathjax: true
---
# Decision Tree

In Decision Tree, we are going to partition the feature space into a set of separate regions. In each region, we are going to make a decision based on the majority of the training data.

The characteristic of Decision Tree is **Greedy**, **Top-Down** and **Recursive**.

$$
\text{Partition} \\
\downarrow \\
\text{Decision1} \\
\text{Y } \swarrow  \quad \searrow \text{N} \\
\text{Decision2} \quad \text{Decision3} \\
$$

Mathematically, Given a region $R_{p}$, we are looking for a split $s_{p}$ such that

$$
\begin{aligned}
s_{p}(j,t) &= (\{X \vert X_{j} < t, X \in R_{p} \}, \{X \vert X_{j} \geq t, X \in R_{p} \}) \\ 
&= (R_1, R_2)
\end{aligned}
$$

where

- $j$ is the feature index
- $t$ is the threshold

Define $L(R)$ as the loss function for region $R$. Given $C$ classes, define $\hat{p}_{c}$ to be the proportion of examples in region $R$ that are class $c$.

$$
L_{\text{missclassfication}} = 1 - \max_{c} \hat{p}_{c}
$$

Therefore, the goal is to find the split $s_{p}$ that minimizes the loss function

$$
\max_{j,t} L(R_{p}) - (L(R_{1}) + L(R_{2}))
$$

where $L(R_{p})$ is the parent loss and $L(R_{1}) + L(R_{2})$ is the children loss.

However, the above loss function is not sensitive, since no matter how to split the region, the children loss will always equal to the parent loss. Therefore, we need to use the **Cross-Entropy** as loss function.

$$
L_{\text{cross-entropy}} = - \sum_{c} \hat{p}_{c} \log_{2} \hat{p}_{c}
$$

or We can use **Gini Loss** as loss function.

$$
L_{\text{gini}} = \sum_{c} \hat{p}_{c} (1 - \hat{p}_{c})
$$

## Regression Trees

Given that region $R_{m}$, we have the prediction $\hat{y}_{m}$:

$$
\hat{y}_{m} = \frac{1}{\vert R_{m} \vert} \sum_{i \in R_{m}} y_{i}
$$

and so the loss function is:

$$
L(R_{m}) = \frac{1}{\vert R_{m} \vert} {\sum_{i \in R_{m}} (y_{i} - \hat{y}_{m})^{2}}
$$

## Categorical Variables

If there exists $q$ categories, then we will have $2^{q}$ possible splits.

## Regularization of Decision Trees

1. Minimize the leaf size

2. Maximize the depth of the tree

3. Maximize the number of nodes

4. Minimize the decrease of loss

5. Pruning (misclassification with validation set)

## Runtime

- $m$ examples

- $n$ features

- $d$ depth

$$
\begin{aligned}
Test Time &= O(d) \\
Train Time &= O(m \cdot n \cdot d)
\end{aligned}
$$

# Ensemble Methods

Take $X_{i}$'s which are ramdom variables, that are independent identically distributed.

$$
Var(X_{i}) = \sigma^{2},  Var(\bar{X}) = \frac{\sigma^{2}}{n}
$$

If we drop the independence assumption, then $X_{i}$'s are only identically distributed and correlated by $\rho$.

$$
Var(\bar{X}) = \rho \sigma^{2} + \frac{1 - \rho}{n} \sigma^{2}
$$

## Ways to ensemble

1. different algorithms

2. different training sets

3. Bagging

4. Boosting

# Bagging

Bagging, short for **Bootstrap Aggregating**, involves creating multiple subsets of the training data by sampling with replacement. Given a true population $P$ and a training set $S \sim P$, bootstrapping assumes $P = S$ and generates bootstrap samples $Z \sim S$.

Given that boostrap samples $Z_{1}, Z_{2}, \cdots, Z_{m}$, and train model $G_{m}$ on $Z_{m}$

$$
G(m) = \frac{1}{m} \sum_{i=1}^{m} G_{i}
$$

## Bias-Variance Analysis

$$
Var(\bar{X}) = \rho \sigma^{2} + \frac{1 - \rho}{m} \sigma^{2}
$$

Bootstrapping is driving down $\rho$ and we have $M \uparrow \implies Var \downarrow$, meanwhile, Bias slightly increased due to the random subsampling.

Decision Tree is a algorithm with **high variance** and **low bias** which is ideal for bagging.

# Random Forests

At each split, consider only a fraction of your total features.

- Decrease $\rho$

- Decorrelate Models

# Boosting

Boosting is algorithm that combines **multiple weak learners** to **create a strong learner**.

Determine for classifier $G_{m}$, the weight of the model $\alpha_{m}$

$$
\alpha_(m) \propto log(\frac{1 - \epsilon_{m}}{\epsilon_{m}})
$$

where $\epsilon_{m}$ is the error rate of the model.

A better classifier will have a higher weight vice versa.

And the final model is

$$
G(x) = \sum_{m=1}^{M} \alpha_{m} G_{m}(x)
$$