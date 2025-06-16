import numpy as np
import torch
import re
from sympy import sympify
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
  def __init__(self, num_hid, FEATURE_KEYS):
    super().__init__()
    num_emb_dim = 64  # Dimension of the embedding layer.
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

def trajectory_balance_loss(logZ, log_P_F, log_P_B, reward):
    """Trajectory balance objective converted into mean squared error loss."""
    reward=torch.tensor(reward).float()
    return (logZ + log_P_F - torch.log(torch.clamp(reward, min=1e-30)) - log_P_B).pow(2)

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