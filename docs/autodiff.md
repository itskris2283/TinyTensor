# TinyTensor Reverse-Mode Automatic Differentiation Engine

## 1. Mathematical Foundations

Given a scalar loss function $\mathcal{L}: \mathbb{R}^n \to \mathbb{R}$ computed by a directed acyclic computation graph, reverse-mode automatic differentiation computes the gradient of $\mathcal{L}$ with respect to all intermediate and input nodes:
$$\bar{v}_i = \frac{\partial \mathcal{L}}{\partial v_i}$$
where $\bar{v}_i$ denotes the *adjoint* of node $v_i$.

By the multivariable chain rule:
$$\bar{v}_i = \sum_{j \in \text{Consumers}(v_i)} \bar{v}_j \cdot \frac{\partial v_j}{\partial v_i}$$

## 2. Vector-Jacobian Products (VJPs) for Supported Operations

### 2.1 Addition & Subtraction
For $Z = X + Y$:
$$\bar{X} = \text{unbroadcast}(\bar{Z}, \text{shape}(X)), \quad \bar{Y} = \text{unbroadcast}(\bar{Z}, \text{shape}(Y))$$

For $Z = X - Y$:
$$\bar{X} = \text{unbroadcast}(\bar{Z}, \text{shape}(X)), \quad \bar{Y} = \text{unbroadcast}(-\bar{Z}, \text{shape}(Y))$$

### 2.2 Multiplication & Division
For $Z = X \odot Y$:
$$\bar{X} = \text{unbroadcast}(\bar{Z} \odot Y, \text{shape}(X)), \quad \bar{Y} = \text{unbroadcast}(\bar{Z} \odot X, \text{shape}(Y))$$

For $Z = X / Y$:
$$\bar{X} = \text{unbroadcast}(\bar{Z} / Y, \text{shape}(X)), \quad \bar{Y} = \text{unbroadcast}(-\bar{Z} \odot X / Y^2, \text{shape}(Y))$$

### 2.3 Non-linear Activations
For $Z = \text{ReLU}(X)$:
$$\bar{X} = \bar{Z} \odot \mathbb{I}(X > 0)$$

For $Z = \text{Sigmoid}(X)$:
$$\bar{X} = \bar{Z} \odot Z \odot (1 - Z)$$

### 2.4 Matrix Multiplication
For $C = A B$ where $A \in \mathbb{R}^{M \times K}, B \in \mathbb{R}^{K \times N}, C \in \mathbb{R}^{M \times N}$:
$$\bar{A} = \bar{C} B^T, \quad \bar{B} = A^T \bar{C}$$

For batched matrix multiplication $C_b = A_b B_b$, the transposition and multiplications are performed batch-wise.

### 2.5 Reductions
For $Z = \text{reduce\_sum}(X, \text{axes})$:
$$\bar{X} = \text{broadcast\_to}(\bar{Z}, \text{shape}(X))$$

For $Z = \text{reduce\_mean}(X, \text{axes})$:
$$\bar{X} = \frac{1}{\prod_{a \in \text{axes}} d_a} \cdot \text{broadcast\_to}(\bar{Z}, \text{shape}(X))$$

## 3. Unbroadcasting Algorithm

When an operation $Z = f(X, Y)$ broadcasts $X$ of shape $(1, N)$ to shape $(M, N)$, the gradient $\bar{Z}$ has shape $(M, N)$. TinyTensor's autodiff engine automatically collapses broadcasted dimensions back to the original operand shape:
1. Prepended dimensions (due to rank expansion) are summed out.
2. Singleton dimensions where $d_i = 1$ in the operand but $d'_i > 1$ in the output are summed along axis $i$ with `keepdims=True`.
3. The resulting tensor exactly matches the shape of the input parameter.

## 4. Verification Protocol

Every analytical gradient derivation in TinyTensor is strictly verified against numerical central finite differences:
$$\frac{\partial \mathcal{L}}{\partial \theta_j} \approx \frac{\mathcal{L}(\theta + \epsilon \cdot e_j) - \mathcal{L}(\theta - \epsilon \cdot e_j)}{2\epsilon}$$
All gradients maintain relative error $\frac{|\bar{\theta}_{\text{analytical}} - \bar{\theta}_{\text{numerical}}|}{|\bar{\theta}_{\text{numerical}}| + 10^{-5}} < 10^{-3}$ across test suites.
