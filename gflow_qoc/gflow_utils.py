import numpy as np
import torch
import re
import itertools
from functools import lru_cache
from sympy import sympify
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from .utils import *
from .utils import _infer_num_nodes_from_state_dimension
import random
import torch.nn as nn

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_feature_keys(num_nodes, num_colors, n_a=0, c_a=None, verbose=True):
    """
    Build action-space feature keys for optical and ancilla edges.

    Args:
        num_nodes (int): Number of optical paths (before ancillas).
        num_colors (int): Number of colors/modes.
        n_a (int): Number of ancilla nodes to append.
        c_a (Optional[int]): Number of ancilla colors allowed (0..c_a-1).
            If None, ancilla color is fixed to 0 (legacy behavior).
        verbose (bool): Print construction details.

    Returns:
        tuple[list, int]:
            FEATURE_KEYS and total number of nodes (optical + ancilla).
    """
    num_nodes = int(num_nodes)
    num_colors = int(num_colors)
    n_a = int(n_a)
    verbose = bool(verbose)

    if n_a < 0:
        raise ValueError("n_a must be >= 0.")

    if verbose:
        print(f"Number of optical paths: {num_nodes}, Number of modes: {num_colors}")

    node_pairs = list(itertools.combinations(range(num_nodes), 2))
    color_pairs = list(itertools.product(range(num_colors), repeat=2))
    feature_keys = [((n1, n2), (c1, c2)) for (n1, n2) in node_pairs for (c1, c2) in color_pairs]

    if verbose:
        print("Size of feature keys = {}".format(len(feature_keys)))

    total_nodes = num_nodes

    if n_a > 0:
        ancilla_colors = int(c_a) if c_a is not None else 1
        if ancilla_colors < 1:
            raise ValueError("c_a must be >= 1 when ancilla nodes are used.")
        if ancilla_colors > num_colors:
            raise ValueError(f"c_a={ancilla_colors} exceeds num_colors={num_colors}.")

        if verbose:
            print(f"Number of ancilla nodes: {n_a}")
            print(f"Ancilla colors allowed: {list(range(ancilla_colors))}")

        ancilla_nodes = range(num_nodes, num_nodes + n_a)
        ancilla_feature_keys = [
            ((i, a), (c, ancilla_color))
            for a in ancilla_nodes
            for i in range(num_nodes)
            for c in range(num_colors)
            for ancilla_color in range(ancilla_colors)
        ]

        feature_keys.extend(ancilla_feature_keys)
        total_nodes = num_nodes + n_a

        if verbose:
            print(f"Added ancilla features = {len(ancilla_feature_keys)}")
            print("Size of feature keys w/ancilla= {}".format(len(feature_keys)))
            print(f"Updated number of nodes including ancillas: {total_nodes}")

    return feature_keys, total_nodes
    
###################################################################################
#### MODELS FOR TB MODEL ##########################################################
###################################################################################

class TBModel(nn.Module):
  def __init__(self, num_hid, FEATURE_KEYS):
    super().__init__()
    self.mlp = nn.Sequential(
        nn.Linear(len(FEATURE_KEYS), num_hid),  # len(FEATURE_KEYS) input features.
        nn.Tanh(), #nn.LeakyReLU() nn.ReLU()
        nn.Linear(num_hid, 2*len(FEATURE_KEYS)),  # 2*len(FEATURE_KEYS) outputs: len(FEATURE_KEYS) for P_F and len(FEATURE_KEYS) for P_B.
    )
    self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

  def forward(self, x, FEATURE_KEYS):
    logits = self.mlp(x)
    # Slice the logits into forward and backward policies.
    P_F = logits[..., :len(FEATURE_KEYS)]
    P_B = logits[..., len(FEATURE_KEYS):]

    return P_F, P_B

class embTBModel(nn.Module):
  def __init__(self, num_hid, FEATURE_KEYS, n_emb):
    super().__init__()
    num_emb_dim = n_emb  # Dimension of the embedding layer.
    layer_1d = num_hid
    layer_2d = num_hid 
    layer_3d = num_hid
    emb_layer = nn.Embedding(len(FEATURE_KEYS), embedding_dim=num_emb_dim)  # Embedding layer to convert input features to embeddings.
    self.emb_layer = emb_layer
    self.encode_layer = nn.Sequential(
        nn.Linear(len(FEATURE_KEYS)*num_emb_dim, layer_1d),  # layers (Number of Paulis) input features.
        nn.LeakyReLU(),
        nn.Linear(layer_1d, layer_2d),
        nn.LeakyReLU(),
        nn.Linear(layer_2d, layer_3d),
        nn.LeakyReLU(),
    )
    self.logits_layer = nn.Linear(layer_3d, 2*len(FEATURE_KEYS))
    self.logZ = nn.Parameter(torch.ones(1))  # log Z is just a single number.

  def forward(self, x, FEATURE_KEYS):
    x_emb = self.emb_layer(x.long())  # Convert input features to embeddings.
    x_encoded = self.encode_layer(x_emb.flatten().unsqueeze(0))
    logits = self.logits_layer(x_encoded)
    # Slice the logits into forward and backward policies.
    P_F = logits[..., :len(FEATURE_KEYS)]
    P_B = logits[..., len(FEATURE_KEYS):]

    return P_F, P_B
  
class GINEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(in_channels if _ == 0 else hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINConv(mlp))

    def forward(self, x, edge_index):
        for conv in self.convs:
            x = conv(x, edge_index)
        return x

class GIN_TBModel(nn.Module):
    def __init__(self, node_feat_dim, hidden_dim, FEATURE_KEYS):
        super().__init__()
        self.encoder = GINEncoder(node_feat_dim, hidden_dim)
        self.pool = global_add_pool  
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * len(FEATURE_KEYS)),  # P_F and P_B
        )
        self.logZ = nn.Parameter(torch.ones(1))
        self.FEATURE_KEYS = FEATURE_KEYS

    def forward(self, data):  # `data` is a torch_geometric.data.Data object
        x, edge_index, batch = data.x, data.edge_index, data.batch
        node_emb = self.encoder(x, edge_index)
        graph_emb = self.pool(node_emb, batch)
        logits = self.decoder(graph_emb)
        P_F = logits[..., :len(self.FEATURE_KEYS)] #Forward policy logits
        P_B = logits[..., len(self.FEATURE_KEYS):] #Backward polilcy logits
        return P_F, P_B

class GINEEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)  
        self.edge_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            edge_encoder = nn.Sequential(
                nn.Linear(edge_feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.edge_encoders.append(edge_encoder)

            conv_nn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn=conv_nn, edge_dim=hidden_dim))

        self.final = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        # if edge_index.size(1) == 0:
        #     return torch.zeros((x.size(0), self.final.in_features), dtype=x.dtype, device=x.device)
        # else:
        x = self.node_encoder(x)  # project to hidden_dim
        for conv, edge_encoder in zip(self.convs, self.edge_encoders):
            edge_attr_transformed = edge_encoder(edge_attr)
            x = conv(x, edge_index, edge_attr_transformed)
        return self.final(x)
        
class GINE_TBModel(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, FEATURE_KEYS):
        super().__init__()
        self.encoder = GINEEncoder(node_feat_dim, edge_feat_dim, hidden_dim)
        self.pool = global_add_pool
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * len(FEATURE_KEYS))  # Forward and Backward
        )
        self.logZ = nn.Parameter(torch.ones(1))
        self.FEATURE_KEYS = FEATURE_KEYS

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.pool(node_emb, batch)
        logits = self.decoder(graph_emb)
        P_F = logits[..., :len(self.FEATURE_KEYS)]
        P_B = logits[..., len(self.FEATURE_KEYS):]
        return P_F.squeeze(), P_B.squeeze()

class GATEncoder(nn.Module):
    """
    Graph Attention encoder using GATv2Conv.
    Supports edge features by projecting edge_attr to hidden_dim and passing
    it via edge_dim (PyG).
    """
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.edge_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)

        for _ in range(num_layers):
            self.edge_encoders.append(
                nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            # concat=False keeps output dim = hidden_dim regardless of heads
            self.convs.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    add_self_loops=True,
                )
            )

        self.final = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        for conv, edge_encoder in zip(self.convs, self.edge_encoders):
            h_in = x
            e = edge_encoder(edge_attr) if edge_attr is not None else None

            x = conv(x, edge_index, e)
            x = nn.functional.elu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

            if self.use_residual:
                x = x + h_in

        return self.final(x)


class GAT_TBModel(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        FEATURE_KEYS,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = GATEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.pool = global_add_pool
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * len(FEATURE_KEYS)),  # Forward and Backward
        )
        self.logZ = nn.Parameter(torch.ones(1))
        self.FEATURE_KEYS = FEATURE_KEYS

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.pool(node_emb, batch)
        logits = self.decoder(graph_emb)
        P_F = logits[..., : len(self.FEATURE_KEYS)]
        P_B = logits[..., len(self.FEATURE_KEYS) :]
        return P_F.squeeze(), P_B.squeeze()


# ---------------------------------------------------------------------
# Graph Transformer encoder + TB model
# ---------------------------------------------------------------------

class GraphTransformerEncoder(nn.Module):
    """
    Graph Transformer encoder using PyG's TransformerConv (attention on graphs).
    Supports edge features via edge_dim.
    """
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
        use_residual: bool = True,
    ):
        super().__init__()
        self.node_encoder = nn.Linear(node_feat_dim, hidden_dim)

        self.edge_encoders = nn.ModuleList()
        self.convs = nn.ModuleList()
        self.dropout = float(dropout)
        self.use_residual = bool(use_residual)

        for _ in range(num_layers):
            self.edge_encoders.append(
                nn.Sequential(
                    nn.Linear(edge_feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )
            # concat=False keeps output dim = hidden_dim regardless of heads
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim,
                    heads=heads,
                    concat=False,
                    dropout=dropout,
                    edge_dim=hidden_dim,
                    beta=True,  # learnable skip connection gate
                )
            )

        self.final = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, edge_index, edge_attr):
        x = self.node_encoder(x)

        for conv, edge_encoder in zip(self.convs, self.edge_encoders):
            h_in = x
            e = edge_encoder(edge_attr) if edge_attr is not None else None

            x = conv(x, edge_index, e)
            x = nn.functional.relu(x)
            x = nn.functional.dropout(x, p=self.dropout, training=self.training)

            if self.use_residual:
                x = x + h_in

        return self.final(x)


class Transformer_TBModel(nn.Module):
    def __init__(
        self,
        node_feat_dim: int,
        edge_feat_dim: int,
        hidden_dim: int,
        FEATURE_KEYS,
        num_layers: int = 3,
        heads: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.encoder = GraphTransformerEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
        self.pool = global_add_pool
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * len(FEATURE_KEYS)),  # Forward and Backward
        )
        self.logZ = nn.Parameter(torch.ones(1))
        self.FEATURE_KEYS = FEATURE_KEYS

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        node_emb = self.encoder(x, edge_index, edge_attr)
        graph_emb = self.pool(node_emb, batch)
        logits = self.decoder(graph_emb)
        P_F = logits[..., : len(self.FEATURE_KEYS)]
        P_B = logits[..., len(self.FEATURE_KEYS) :]
        return P_F.squeeze(), P_B.squeeze()

####################################################################################################
##### END OF MODELS FOR TB #########################################################################
####################################################################################################

def scheduled_beta(
    episode,
    beta=1.0,
    beta_curve=False,
    beta_start=1e-3,
    beta_warmup_steps=1000,
):
    """
    Return the TB beta value for this episode.

    If `beta_curve` is False: returns `beta`.
    If `beta_curve` is True: linearly increases from `beta_start` to `beta`
    and reaches `beta` after `beta_warmup_steps` iterations.
    """
    beta_max = float(beta)
    if not beta_curve:
        return beta_max

    if beta_warmup_steps is None or int(beta_warmup_steps) <= 0:
        return beta_max

    beta_min = min(float(beta_start), beta_max)
    progress = min(1.0, float(episode + 1) / float(beta_warmup_steps))
    return beta_min + (beta_max - beta_min) * progress


def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward=None, beta=1.0):
    """Trajectory balance objective converted into mean squared error loss."""
    if reward is None:
        raise ValueError("`reward` must be provided.")

    reward = torch.as_tensor(reward, dtype=logZ.dtype, device=logZ.device)
    return (
        logZ
        + log_P_F
        - float(beta) * torch.log(torch.clamp(reward, min=1e-30))
        - log_P_B
    ).pow(2)

def state_hash(state, FEATURE_KEYS):
    """Returns a binary hash for each submitted state."""
    return tuple([i in state for i in FEATURE_KEYS])

def state_to_tensor(state, FEATURE_KEYS):
  """Encodes a state as a binary tensor (converted to float32)."""
  return torch.tensor(state_hash(state, FEATURE_KEYS)).float()

@lru_cache(maxsize=128)
def _expression_support(target_expr):
    """Collect nonzero coefficients, including cancellation of repeated kets."""
    expression = ''.join(target_expr.split()).replace('⟩', '').replace('>', '')
    amplitudes, position, num_nodes = {}, 0, None
    for match in re.finditer(r'([^|]*)\|([0-9]+)', expression):
        if match.start() != position:
            raise ValueError("Invalid target expression.")
        coefficient, basis = match.groups()
        coefficient = {'': '1', '+': '1', '-': '-1'}.get(coefficient, coefficient)
        value = sympify(coefficient)
        if value.is_number is not True or value.is_finite is not True:
            raise ValueError("Target coefficients must be finite numbers.")
        if num_nodes is not None and len(basis) != num_nodes:
            raise ValueError("All target basis strings must have the same length.")
        num_nodes = len(basis)
        amplitudes[basis] = amplitudes.get(basis, 0) + value
        position = match.end()
    if position != len(expression) or not amplitudes:
        raise ValueError("Invalid target expression.")
    support = tuple(tuple(map(int, basis)) for basis, value in amplitudes.items()
                    if value.simplify() != 0)
    if not support:
        raise ValueError("Zero-norm target state.")
    return num_nodes, support


def extract_basis_strings(target_expr):
    return [''.join(map(str, basis)) for basis in _expression_support(target_expr)[1]]


class _MatchingLogic:
    """Target-specific perfect-matching bounds on a fixed action universe."""

    def __init__(self, feature_keys, num_nodes, support):
        self.feature_keys = feature_keys
        self.num_nodes = num_nodes
        self._indices = {edge: index for index, edge in enumerate(feature_keys)}
        self._compatible = np.zeros(len(feature_keys), dtype=bool)
        basis_incidence = np.zeros(len(feature_keys), dtype=int)
        self._basis_pairs = []
        for basis in support:
            pairs = {}
            for index, ((i, j), (ci, cj)) in enumerate(feature_keys):
                if (0 <= i < num_nodes and 0 <= j < num_nodes and i != j
                        and (ci, cj) == (basis[i], basis[j])):
                    self._compatible[index] = True
                    basis_incidence[index] += 1
                    pairs.setdefault(tuple(sorted((i, j))), []).append(index)
            self._basis_pairs.append(tuple(
                (i, j, sum(1 << index for index in indices), tuple(indices))
                for (i, j), indices in pairs.items()
            ))
        # Edges outside every full matching contribute no amplitude at all.
        # Ignore colors here: an edge used only by unwanted-basis matchings may
        # still help suppress an intruder and must remain available.
        adjacency = [0] * num_nodes
        for allowed, ((i, j), _) in zip(self._compatible, feature_keys):
            if allowed:
                adjacency[i] |= 1 << j
                adjacency[j] |= 1 << i

        @lru_cache(maxsize=None)
        def has_matching(vertices):
            if not vertices:
                return True
            first = vertices & -vertices
            remaining = vertices ^ first
            partners = adjacency[first.bit_length() - 1] & remaining
            while partners:
                partner = partners & -partners
                if has_matching(remaining ^ partner):
                    return True
                partners ^= partner
            return False

        full = (1 << num_nodes) - 1
        for index, ((i, j), _) in enumerate(feature_keys):
            if self._compatible[index]:
                self._compatible[index] = has_matching(full ^ (1 << i) ^ (1 << j))
        self._max_basis_per_edge = max(1, int(np.max(basis_incidence[self._compatible], initial=0)))
        # Bound cache growth across episodes, without enumerating edge subsets.
        self._completion_costs = lru_cache(maxsize=512)(self._completion_costs)

    def _completion_costs(self, state_bits):
        full = (1 << self.num_nodes) - 1
        impossible = self.num_nodes + 1
        before = np.full(len(self._basis_pairs), impossible, dtype=int)
        after = np.full((len(before), len(self.feature_keys)), impossible, dtype=int)
        if self.num_nodes % 2:
            return before, after
        for row, pairs in enumerate(self._basis_pairs):
            adjacency = [[] for _ in range(self.num_nodes)]
            for i, j, action_bits, _ in pairs:
                cost = 0 if action_bits & state_bits else 1
                adjacency[i].append((1 << j, cost))
                adjacency[j].append((1 << i, cost))

            @lru_cache(maxsize=None)
            def solve(vertices):
                if not vertices:
                    return 0
                first = vertices & -vertices
                remaining = vertices ^ first
                best = impossible
                for partner, cost in adjacency[first.bit_length() - 1]:
                    if partner & remaining:
                        best = min(best, cost + solve(remaining ^ partner))
                        if best == 0:
                            break
                return best

            before[row] = solve(full)
            after[row].fill(before[row])
            for i, j, _, indices in pairs:
                # Force the proposed edge into a matching at zero additional cost.
                # All other matching edges use the original state's 0/1 costs.
                forced = solve(full ^ (1 << i) ^ (1 << j))
                after[row, list(indices)] = min(before[row], forced)
        return before, after

    def supports_target(self, state):
        """Whether the graph currently has a full matching for every target ket."""
        present = {self._indices[(tuple(edge[0]), tuple(edge[1]))] for edge in state}
        before, _ = self._completion_costs(sum(1 << index for index in present))
        return bool(np.all(before == 0))

    def forward(self, state, max_edges=None):
        """Return an allowed-action mask and finite bonuses in [0, 1].

        Each required basis must have a completion within the remaining edge
        budget. A shared-budget bound also limits total missing-edge incidence
        by the maximum number of target bases one edge can help. These bounds
        are necessary, not a joint feasibility guarantee. Preparatory edges and
        extra matchings remain allowed; no unwanted-basis exclusion is applied.
        """
        present = {self._indices[(tuple(edge[0]), tuple(edge[1]))] for edge in state}
        state_bits = sum(1 << index for index in present)
        before, after = self._completion_costs(state_bits)
        mask = self._compatible.copy()
        if present:
            mask[list(present)] = False
        # Even without a budget, every target basis must admit a full matching.
        limit = self.num_nodes // 2 if max_edges is None else int(max_edges) - len(present) - 1
        mask &= np.all(after <= limit, axis=0)
        if max_edges is not None:
            mask &= np.sum(after, axis=0) <= limit * self._max_basis_per_edge
        unsupported = before > 0
        progress = np.zeros(len(mask), dtype=np.float32)
        if np.any(unsupported):
            # A single edge reduces any one basis's minimum completion cost by at most one.
            progress = (before[unsupported, None] - after[unsupported]).mean(axis=0).astype(np.float32)
            progress[~mask] = 0
        return torch.from_numpy(mask), torch.from_numpy(progress)


@lru_cache(maxsize=32)
def _cached_matching_logic(feature_keys, num_nodes, support):
    return _MatchingLogic(feature_keys, num_nodes, support)


def prepare_matching_logic(target_expr, FEATURE_KEYS, target_state=None, num_colors=None):
    """Prepare reusable logic; a supplied target vector defines the true support."""
    feature_keys = tuple((tuple(edge[0]), tuple(edge[1])) for edge in FEATURE_KEYS)
    if target_state is None:
        num_nodes, support = _expression_support(target_expr)
        if num_colors is not None and any(max(basis) >= num_colors for basis in support):
            raise ValueError("Target basis exceeds the supplied number of colors.")
    else:
        vector = np.asarray(target_state)
        if vector.ndim != 1:
            raise ValueError("Target state must be a one-dimensional vector.")
        if not np.all(np.isfinite(vector)):
            raise ValueError("Target amplitudes must be finite.")
        if num_colors is None:
            num_colors = max(2, 1 + max((max(edge[1]) for edge in feature_keys), default=0))
        num_nodes = _infer_num_nodes_from_state_dimension(vector.size, num_colors)
        support = []
        for index in np.flatnonzero(vector):
            digits = [0] * num_nodes
            for node in range(num_nodes - 1, -1, -1):
                index, digits[node] = divmod(int(index), int(num_colors))
            support.append(tuple(digits))
        if not support:
            raise ValueError("Zero-norm target state.")
        support = tuple(support)
    return _cached_matching_logic(feature_keys, num_nodes, support)


def calculate_forward_mask_from_state(state, target_expr, FEATURE_KEYS, max_edges=None,
                                      target_state=None, num_colors=None):
    """Backward-compatible mask-only entry point; prepare once for training loops."""
    logic = prepare_matching_logic(target_expr, FEATURE_KEYS, target_state, num_colors)
    return logic.forward(state, max_edges=max_edges)[0]


def masked_policy_logits(logits, mask, progress=None):
    """Apply finite policy bonuses and strict masks; callers handle empty masks."""
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    result = torch.nan_to_num(logits, nan=-100.0, posinf=100.0, neginf=-100.0)
    if progress is not None:
        progress = torch.as_tensor(progress, dtype=logits.dtype, device=logits.device)
        result = result + torch.nan_to_num(progress, nan=0.0, posinf=0.0, neginf=0.0)
    return result.masked_fill(~mask, -torch.inf)

def calculate_backward_mask_from_state(state, FEATURE_KEYS):
    """Every present edge is a valid parent action for a forward-reachable state.

    Removing one edge raises each minimum matching-completion cost by at most
    one, and their sum by at most the maximum basis incidence of one edge,
    while restoring one unit of budget. Thus every parent also satisfies the
    forward bounds; the soft progress preference excludes no transitions.
    """
    return torch.Tensor(
        [1 if feature in state else 0 for feature in FEATURE_KEYS]
    ).bool()
