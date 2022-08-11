# Constraint-Adversarial-MPC
Allerton 2022 submission. [link](https://arxiv.org/abs/2207.06982)

Adversarial attack for input-driven LQR on cost and constraints with CVXPY.

## Prerequisites

python version: 3.9.12
|package|version|
|:-----:|------:|
|numpy  | 1.20.1|
|pandas  | 1.2.3|
|matplotlib|3.4.3|
|seaborn|0.11.2|
|cvxpy|1.1.18|
|statannotations|0.4.4|


## Getting Started

* To run the code, execute: 
```constraint_adv.ipynb```

* Functions for plotting:
```plot_utilities.py```

* Generate input-driven LQR controllers (both constrained and unconstrained):
```system_dynamics.py```

* Dimension of action space, state space, and other parameters:
```parameters.yml```

## Contact

Po-han Li - pohanli@utexas.edu
