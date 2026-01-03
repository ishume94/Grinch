# Grinch

This repository contains code for **Grinch**.

- `driver.py` is the main entry point and includes the training configuration (target states, maximum number of edges, etc.), training function to employ and result analysis.

## Overview

For each episode, the model constructs a graph by sequentially adding edges (up to a user-defined maximum). After the final graph (state) is generated, the code performs a continuous optimization over edge weights to maximize the fidelity with a target quantum state. The sampler for state generation is learned through a Generative Flow Network (GFlowNet) employing the Trajectory Balance loss function. Different models for parametrizing the GFlowNet are available.

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
- The pruned graph is kept **only if** its fidelity remains high (default threshold: \(F \ge 0.99\)).
- Accepted pruned graphs are saved for later inspection.

## Target state examples

You can define target states directly in `driver.py`. All states are properly normalized inside the code. Example expressions:

- **(4,2)-GHZ**
  - `1|0000⟩ + 1|1111⟩` 
  - Interpreted as $\frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$

- **(6,3)-GHZ**
  - `1|000000⟩ + 1|111111⟩ + 1|222222⟩`

### Targets requiring ancilla photons
The default ancilla state is $|0\rangle$. With multiple ancilla photons, the full state is interpreted as:

$|\psi\rangle_{target} \otimes |0\ldots 0\rangle$

For states that require ancilla photons, explicitly include the ancilla in the target state expression. E.g., for the state requiring one ancilla photon, $(|000\rangle + |111\rangle)\otimes |0\rangle$ the target expression should be `1|0000⟩ + 1|1110⟩`. Examples:

- **(3,2)-GHZ with one ancilla photon**
  - Target: `1|0000⟩ + 1|1110⟩`

- **\(|D(3,(1,1,1))\rangle \otimes |0\rangle\)**
  - Target: `1|0120⟩ + 1|0210⟩ + 1|1020⟩ + 1|1200⟩ + 1|2010⟩ + 1|2100⟩`



## Running

```bash
# No ancilla photons (even number of photons/no ancilla required)
python driver.py > out.log

#With ancilla photons
python driver.py n_a=1 > out.log
```

### Available models (GFlowNets w/Trajectory Balance training objective)

The repository includes several architectures for Trajectory Balance training:

TB_train: MLP

GIN_TB_train: Graph Isomorphism Network (GIN)

GINE_TB_train: GIN with edge features (GINE)

GAT_TB_train: Graph Attention Network (GAT)

Transformer_TB_train: Graph Transformer
