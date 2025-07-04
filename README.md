# Machine Learning for Energy Minimization Using PINN

## Overview
This project investigates **deep learning methods for scientific computing problems**, specifically focusing on **energy minimization in elastic crystals**. The goal is to determine how materials should deform to achieve **minimum potential energy** under **Dirichlet boundary constraints**. Inspired by **Physics-Informed Neural Networks (PINNs)**, this approach leverages neural networks to approximate solutions to this optimization problem. The complete results can be found in this [report](report.pdf). The implementation is given in [`src`](src).

## Background
Elastic molecular crystals are widely used in **sensors, actuators, and electronics** due to their mechanical flexibility. The stable geometry of these crystals corresponds to **minimum energy configurations**, making it an essential problem in material science. Traditionally, solving these problems involves numerical PDE solvers, but deep learning—especially **PINNs**—offers a promising alternative.

This project builds a **1D PINN-based energy minimization framework**, which can potentially be extended to **higher-dimensional problems**.

## Mathematical Formulation
For a **1D elastic crystal**, the potential energy is defined as:

$$
E_\gamma [u] = \int_{0}^{1} W_\gamma (u'(x)) dx
$$

for some real constant $` \gamma `$, where $`W_\gamma (z) = (z + \gamma)^2 (1 - \gamma - z)^2 `$ is a double-well potential function with minima at $` z = 0 `$ and $` z = 1 `$.

We seek an **energy-minimizing function** $` u(x) `$ satisfying boundary constraints:

$$
u(0) = u(1) = 0 .
$$

## Methods
Four different **PINN architectures** are used based on different formulations of the problem:

1. **Method 1:** Direct minimization for $` \gamma \geq 1 `$
2. **Method 2:** Symmetry transformation for $` \gamma \leq 0 `$
3. **Method 3:** Standard approach for $` \gamma \in [0.5, 1] `$
4. **Method 4:** Alternative transformation for $` \gamma \in [0, 0.5] `$

Each method applies **loss function adaptations**, boundary constraints, and **self-adaptive weights** to ensure accurate learning.

## Implementation
The project is implemented using **TensorFlow** and consists of:

- **PINN Neural Network** for energy minimization
- **Adaptive loss function** incorporating boundary constraints
- **Training strategies** with gradient optimization
- **Comparisons of different architectures & hyperparameters**

## Results
### **Performance Across Different γ Values**
The results demonstrate that the **PINNs successfully approximate the minimizers** for different values of **γ**, converging to the expected solutions.

- **Method 1 & 2**: Fast convergence for extreme γ values.
- **Method 3 & 4**: Slower convergence near γ = 0.5 due to non-uniqueness of minimizers.

### Effect of Hyperparameters
Experiments show that **network depth, activation functions, optimizers, and learning rates** significantly affect performance. The best combination identified is:
- **Single hidden layer** with **1 neuron** per layer.
- **ReLU activation function**.
- **SGD optimizer** with a learning rate of **0.1**.

## Future Work
- Extend to **higher-dimensional energy minimization problems**.
- Apply **adaptive collocation techniques** to refine training points dynamically.
- Investigate **meta-learning approaches** where γ itself is an **input variable**.

## References
This work is based on prior research in PINNs and deep learning for PDEs:
- [Raissi et al., 2018] Physics-Informed Neural Networks: A Deep Learning Framework for Solving PDEs.
- [McClenny & Braga-Neto, 2020] Self-adaptive PINNs using attention mechanisms.
- Various references on molecular and material science.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


