import torch
import torch.nn as nn
import networkx as nx

# Define the graph coloring environment
class GraphColoringEnv:
    def __init__(self, graph):
        self.graph = graph
        self.num_nodes = len(graph.nodes)
        self.state = {node: None for node in graph.nodes}
        self.available_colors = set(range(self.num_nodes))  # At most num_nodes colors
        self.done = False

    def get_valid_actions(self):
        """Returns a mask indicating valid actions."""
        mask = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.bool)
        for node in self.graph.nodes:
            if self.state[node] is None:
                used_colors = {self.state[n] for n in self.graph.neighbors(node) if self.state[n] is not None}
                valid_colors = self.available_colors - used_colors
                for color in valid_colors:
                    mask[node, color] = True
        return mask

    def step(self, node, color):
        """Applies the action (coloring a node)."""
        self.state[node] = color
        if all(v is not None for v in self.state.values()):
            self.done = True

    def get_reward(self):
        """Returns the reward based on the number of colors used."""
        num_colors = len(set(self.state.values()))
        print(num_colors)
        return 1.0 / num_colors if self.done else 0.0

# Define the GFlowNet policy model
class GFlowNet(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.policy_net = nn.Sequential(
            nn.Linear(num_nodes * 2, 128),
            nn.ReLU(),
            nn.Linear(128, num_nodes * num_nodes)  # Flattened action space
        )

    def forward(self, state_tensor):
        logits = self.policy_net(state_tensor)
        return logits.view(state_tensor.shape[0], num_nodes, num_nodes)  # Reshape to (batch, nodes, colors)

def trajectory_balance_loss(model, trajectory, reward):
    log_probs = []
    for state_tensor, (node, color) in trajectory:
        logits = model(state_tensor.unsqueeze(0))
        probs = torch.softmax(logits.view(-1), dim=0)
        action_idx = node * num_nodes + color
        log_probs.append(torch.log(probs[action_idx]))
    
    return -sum(log_probs) * reward

# Generate a test graph
graph = nx.erdos_renyi_graph(1000, 0.3, seed=42)
env = GraphColoringEnv(graph)
num_nodes = len(graph.nodes)

# Initialize the model and loss function
model = GFlowNet(num_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
#loss_fn = TrajectoryBalance()

# Training loop
for epoch in range(1000):
    env = GraphColoringEnv(graph)  # Reset environment
    state_tensor = torch.zeros(num_nodes * 2)  # Example state representation
    traj = []
    
    while not env.done:
        logits = model(state_tensor.unsqueeze(0))
        mask = env.get_valid_actions()
        
        # Sample an action (node, color) from valid actions
        #logits[~mask] = -float('inf')  # Mask out invalid actions
        logits[:, ~mask] = -float('inf')
        probs = torch.softmax(logits.view(-1), dim=0)
        action_idx = torch.multinomial(probs, 1).item()
        node, color = divmod(action_idx, num_nodes)
        
        env.step(node, color)
        traj.append((state_tensor.clone(), (node, color)))
        
        # Update state representation
        state_tensor[node] = color
    
    # Compute reward and loss
    reward = env.get_reward()
    loss = trajectory_balance_loss(model, traj, reward)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Reward: {reward}, Loss: {loss.item()}")

