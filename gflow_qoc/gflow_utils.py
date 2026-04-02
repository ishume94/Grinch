import numpy as np
import torch
import re
import itertools
from sympy import sympify
from torch_geometric.nn import GINConv, GINEConv
from torch_geometric.nn import GATv2Conv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_add_pool
from torch_geometric.data import Data
from .utils import *
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

###IMPROVED FORWARD MASK!!!!
def extract_basis_strings(target_expr):
    target_expr = target_expr.replace("⟩", "").replace(" ", "")
    terms = re.findall(r'([^\+]+?\|[0-9]+)', target_expr)
    basis_strings = []
    for term in terms:
        match = re.match(r'([^|]+)\|([0-9]+)', term)
        if match:
            _, bitstring = match.groups()
            basis_strings.append(bitstring)
    return basis_strings

def calculate_forward_mask_from_state(state, target_expr, FEATURE_KEYS):
    mask = np.ones(len(FEATURE_KEYS))  # 1 = allowed, 0 = masked

    basis_strings = extract_basis_strings(target_expr)

    for i, edge in enumerate(FEATURE_KEYS):
        # Rule 1: already in state → disallowed
        if edge in state:
            mask[i] = 0
            continue

        (n1, n2), (c1, c2) = edge
        compatible = False
        for bitstring in basis_strings:
            if len(bitstring) <= max(n1, n2):
                continue  # Skip if bitstring is shorter than required node positions
            if int(bitstring[n1]) == c1 and int(bitstring[n2]) == c2:
                compatible = True
                break

        if not compatible:
            mask[i] = 0  # Rule 2: incompatible with target expression

    return torch.tensor(mask).bool()

def calculate_backward_mask_from_state(state, FEATURE_KEYS):
    """Here, we mask backward actions to only select parent nodes."""
    # This mask should be 1 for any action that could have led to the current state,
    # otherwise it should be zero.
    return torch.Tensor(
        [1 if feature in state else 0 for feature in FEATURE_KEYS]
    ).bool()
