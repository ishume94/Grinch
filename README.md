# Grinch

This repository contains code for **Bottom-Up Automated Design of Quantum Optical Circuits Using Reward-Driven Generative Models.**

- `driver.py` is the main file to run and includes the training configuration (target states, maximum number of edges, etc.), sampling function to generate optical circuits and result analysis.
- Clone the repositary and install with `pip install -e .`
- Required packages: `scipy`, `torch`, `torch-geometric`, `matplotlib`, `sympy`, `tqdm`.

## Overview

For each episode, the model constructs a graph by sequentially adding edges (up to a user-defined maximum). After the final graph (state) is generated, the code performs a continuous optimization over edge weights to maximize the fidelity with a target quantum state. The sampler for state generation is learned through a Generative Flow Network (GFlowNet) employing the Trajectory Balance loss function. Different models for parametrizing the GFlowNet are available.

## Target state examples

You can define target states directly in `driver.py`. All states are properly normalized inside the code. Example expressions:

- **(4,2)-GHZ**
  - Target: `1|0000⟩ + 1|1111⟩` 
  - Interpreted as $\frac{1}{\sqrt{2}}(|0000\rangle + |1111\rangle)$ inside the code for Fidelity calculation.

- **(6,3)-GHZ**
  - Target: `1|000000⟩ + 1|111111⟩ + 1|222222⟩`

### Targets requiring ancilla photons
The default ancilla state is $|0\rangle$. With multiple ancilla photons, the full state is interpreted as:

$$
|\psi\rangle_{target} \otimes |0\ldots 0\rangle
$$

For states that require ancilla photons, explicitly include the ancilla in the target state expression. E.g., for the state requiring one ancilla photon, $(|000\rangle + |111\rangle)\otimes |0\rangle$ the target expression should be `1|0000⟩ + 1|1110⟩`. Examples:

- **(3,2)-GHZ with one ancilla photon**
  - Target: `1|0000⟩ + 1|1110⟩`

- **$|D(3,(1,1,1))\rangle \otimes |0\rangle$**
  - Target: `1|0120⟩ + 1|0210⟩ + 1|1020⟩ + 1|1200⟩ + 1|2010⟩ + 1|2100⟩`

### Ancilla action space (`n_a` and `c_a`)

- `n_a` controls how many ancilla nodes are added.
- `c_a` controls how many ancilla colors are allowed for ancilla edges (`0, 1, ..., c_a-1`).
- If `c_a` is not provided, ancilla color is fixed to `0` only.
- `c_a` must satisfy `1 <= c_a <= num_colors` when `n_a > 0`.


### Quantum-gate targets (`--q-gate`)

Run `driver.py` with `--q-gate` to synthesize a non-local photonic quantum gate. In this mode, `num_nodes` is the number of non-ancilla nodes and must be even. The first `num_nodes / 2` nodes encode the input, the next `num_nodes / 2` nodes encode the output, and any nodes added with `n_a` are ancillas appended after the input and output nodes.

The target expression encodes the desired gate action using

$$
|\mathrm{in}\rangle \otimes |\mathrm{out}\rangle
= |\mathrm{in},\mathrm{out}\rangle.
$$

For example, a CNOT gate acts as

$$
|00\rangle\to|00\rangle,\quad
|01\rangle\to|01\rangle,\quad
|10\rangle\to|11\rangle,\quad
|11\rangle\to|10\rangle,
$$

To implement this gate, two ancilla nodes in state $|0\rangle$ are required. It is therefore encoded by

`1|000000⟩ + 1|010100⟩ + 1|101100⟩ + 1|111000⟩`.

Here, the first two entries in each ket are the input, the next two are the output, and the final two are the ancillas in $|00\rangle$. As with state targets, the expression is normalized internally. Run this CNOT example with `python driver.py --q-gate n_a=2`. In q-gate mode, `FEATURE_KEYS` excludes every edge whose two endpoints are input nodes. Circuit generation, optimization, pruning, and plotting otherwise remain unchanged.

### Edges between ancilla nodes (`--a-edges`)

Add `--a-edges` to allow edges between ancilla nodes, for example `python driver.py --q-gate --a-edges n_a=2`. Allowing these edges helps implement arrays with lower edge counts for nonlocal Toffoli and CNOT(2,3). Both endpoints use the ancilla colors set by `c_a` (color `0` by default). These edges are disabled by default, and the flag has no effect with fewer than two ancillas.

## Running

```bash
# No ancilla photons (even number of photons/no ancilla required)
python driver.py > out.log

# With ancilla photons (default ancilla color fixed to 0)
python driver.py n_a=1 > out.log

# With ancilla photons and variable ancilla colors (here: colors 0 and 1)
python driver.py n_a=1 c_a=2 > out.log

# Quantum-gate mode
python driver.py --q-gate > out.log

# CNOT example with two appended |0⟩ ancilla nodes
python driver.py --q-gate n_a=2 > out.log
```
### Available models (GFlowNets with Trajectory Balance)

The repository includes several architectures for Trajectory Balance training:

- **`TB_train`** — Multi-Layer Perceptron (MLP), can use a node embedding.
- **`GIN_TB_train`** — Graph Isomorphism Network (GIN)
- **`GINE_TB_train`** — GIN with edge features (GINE)
- **`GAT_TB_train`** — Graph Attention Network (GAT)
- **`Transformer_TB_train`** — Graph Transformer

## Trajectory Balance loss and adjusting exploration vs explotation

The TB objective used in training is:

$$
\left(\log Z + \log P_F(\tau) - \beta \log(\max(R(\tau), 10^{-30})) - \log P_B(\tau)\right)^2
$$

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

$$
-\mathcal{F}(\mathcal{G},\boldsymbol{w})
$$

  where $\mathcal{F}$ is the fidelity between the generated and target states.

- **Pruning mode (`pruning=True`)** can make use of L1-regularized optimization, the option is commented and will be implemented as a flag at a later point:

$$
-\mathcal{F}(\mathcal{G},\boldsymbol{w}) + \alpha \lVert w \rVert_1
$$

  with default $\alpha = 0.1$.

## Pruning

Pruning is applied only at terminal states (after the final edge is sampled and weights are optimized).

In the current default training path (`Transformer_TB_train` in `driver.py`), the reward used for TB is $R=\mathcal{F}^2$. Pruning is logic-based:

1. With pruning enabled, a terminal state is considered for pruning only if $R \ge 0.95^2$ (equivalently $\mathcal{F}\ge 0.95$).
2. `prune_state_by_logic(...)` then greedily removes edges in increasing $|w|$ order.
   A removal is accepted only if the remaining graph still satisfies the support clause:
   every component of the target state (amplitude magnitude $>10^{-12}$) is generated by at least one perfect matching.
3. Fidelity is recomputed on the pruned graph using the pruned weights, and the graph is kept only if $\mathcal{F}_{\text{pruned}}\ge 0.95$.
4. Accepted pruned graphs are stored in `pruned_states` for later analysis.

Note: other training functions (`TB_train`, `GIN_TB_train`, `GINE_TB_train`, `GAT_TB_train`) still use weight-threshold pruning (`|w|>0.01`) with a `0.99` post-pruning fidelity check.

### Logic used for pruning states (`prune_state_by_logic`)

**Action masking.** For state $s$ and action $a$, $\mathcal{M}(s,a)=1$ allows the action and $\mathcal{M}(s,a)=0$ excludes it. Forward masks reject repeated edges, edges incompatible with the target or graph constraints, and additions that fail necessary perfect-matching completion or remaining-edge-budget checks. These checks do not guarantee joint completion of all target bases. A policy bonus favors progress toward unsupported target bases. Backward masks allow removal of present edges. If all forward actions are masked, sampling stops without adding duplicates; incomplete target support receives zero reward.

**Pruning.** The S clause requires every target basis with amplitude magnitude above `support_tol=1e-12` to retain at least one perfect matching over the full target node set, including ancillas.

1. Remove duplicate colored edges, keeping the first occurrence and its weight. If the graph fails S or its supplied weights fail the fidelity threshold, return it without further pruning.
2. Try deleting edges in increasing $|w|$ order. Reject any deletion that violates S.
3. Check fidelity with the surviving weights. If it fails and `reoptimize=True` (default), try L-BFGS-B from those weights. Accept only if the resulting fidelity is finite and meets `fidelity_threshold=0.95`; otherwise restore the edge and weights.
4. Repeat until no edge is removed or `max_passes=10` is reached. Return the pruned graph, aligned weights, and a keep-mask over the original edges.

The unwanted-basis C clause remains disabled: weight optimization can suppress intruder amplitudes while retaining edges needed by the target. With `weights=None`, pruning checks support only. Fidelity acceptance uses $\mathcal{F}$, while the training reward remains $\mathcal{F}^2$.
