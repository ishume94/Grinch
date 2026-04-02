# Grinch

This repository contains code for **Bottom-Up Automated Design of Quantum Optical Circuits Using Reward-Driven Generative Models.**

- `driver.py` is the main file to run and includes the training configuration (target states, maximum number of edges, etc.), sampling function to generate optical circuits and result analysis.
- Clone the repositary and install with `pip install -e .`
- Required packages: `scipy`, `torch`, `torch-geometric`, `matplotlib`, `sympy`, `tqdm`.

## Overview

For each episode, the model constructs a graph by sequentially adding edges (up to a user-defined maximum). After the final graph (state) is generated, the code performs a continuous optimization over edge weights to maximize the fidelity with a target quantum state. The sampler for state generation is learned through a Generative Flow Network (GFlowNet) employing the Trajectory Balance loss function. Different models for parametrizing the GFlowNet are available.

## Trajectory Balance loss and adjusting exploration vs explotation

The TB objective used in training is:

\[
\left(\log Z + \log P_F(\tau) - \beta \log(\max(R(\tau), 10^{-30})) - \log P_B(\tau)\right)^2
\]

where $\beta$ scales the reward term and is an inverse temperature term. At lower values of $\beta$ exploration is promoted while larger values promote exploitation.

You can control this behavior in `driver.py` with:

- `beta = 1.0`
- `beta_curve = False`
- `beta_start = 1e-3`
- `beta_warmup_steps = 1000`

Behavior:

- If `beta_curve=False`, training uses a constant `beta`.
- If `beta_curve=True`, `beta` is linearly increased from `beta_start` to `beta` over `beta_warmup_steps` episodes.

The main use of the beta curve is to start with a somewhat explorative policy and then adjusting for a more exploitative approach.

## Fidelity optimization

After a terminal graph is sampled, the edge weights are optimized via L-BFGS-B:

- **Unregularized optimization** minimizes:
  $-\mathcal{F}(\mathcal{G},\boldsymbol{w})$
  where $\mathcal{F}$ is the fidelity between the generated and target states.

- **Pruning mode (`pruning=True`)** uses L1-regularized optimization:
  $-\mathcal{F}(\mathcal{G},\boldsymbol{w}) + \alpha \lVert w \rVert_1$
  with default $\alpha = 0.1$.

## Pruning

If `pruning=True`, the code attempts to prune terminal graphs using the optimized weights:

- Any edge with $|w| < 0.01$ is removed.
- The pruned graph is kept **only if** its fidelity remains high (default threshold: $\mathcal{F}(\mathcal G,\boldsymbol{w}^*) \ge 0.99$ with $\boldsymbol{w}^*$ being the previously optimized weights).
- Accepted pruned graphs are saved for result analysis.

## Target state examples

You can define target states directly in `driver.py`. All states are properly normalized inside the code. Example expressions:

- **(4,2)-GHZ**
  - Target: `1|0000⟩ + 1|1111⟩` 
  - Interpreted as $\frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$ inside the code for Fidelity calculation.

- **(6,3)-GHZ**
  - Target: `1|000000⟩ + 1|111111⟩ + 1|222222⟩`

### Targets requiring ancilla photons
The default ancilla state is $|0\rangle$. With multiple ancilla photons, the full state is interpreted as:

$|\psi\rangle_{target} \otimes |0\ldots 0\rangle$

For states that require ancilla photons, explicitly include the ancilla in the target state expression. E.g., for the state requiring one ancilla photon, $(|000\rangle + |111\rangle)\otimes |0\rangle$ the target expression should be `1|0000⟩ + 1|1110⟩`. Examples:

- **(3,2)-GHZ with one ancilla photon**
  - Target: `1|0000⟩ + 1|1110⟩`

- **$|D(3,(1,1,1))\rangle \otimes |0\rangle$**
  - Target: `1|0120⟩ + 1|0210⟩ + 1|1020⟩ + 1|1200⟩ + 1|2010⟩ + 1|2100⟩`

### Ancilla action space (`n_a` and `c_a`)

- `n_a` controls how many ancilla nodes are added.
- `c_a` controls how many ancilla colors are allowed for ancilla edges (`0, 1, ..., c_a-1`).
- If `c_a` is not provided, behavior is unchanged from before: ancilla color is fixed to `0` only.
- `c_a` must satisfy `1 <= c_a <= num_colors` when `n_a > 0`.

## Running

```bash
# No ancilla photons (even number of photons/no ancilla required)
python driver.py > out.log

# With ancilla photons (default ancilla color fixed to 0)
python driver.py n_a=1 > out.log

# With ancilla photons and variable ancilla colors (here: colors 0 and 1)
python driver.py n_a=1 c_a=2 > out.log
```
### Available models (GFlowNets with Trajectory Balance)

The repository includes several architectures for Trajectory Balance training:

- **`TB_train`** — Multi-Layer Perceptron (MLP), can use a node embedding.
- **`GIN_TB_train`** — Graph Isomorphism Network (GIN)
- **`GINE_TB_train`** — GIN with edge features (GINE)
- **`GAT_TB_train`** — Graph Attention Network (GAT)
- **`Transformer_TB_train`** — Graph Transformer
