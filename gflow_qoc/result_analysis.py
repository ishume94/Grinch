import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import itertools as it
from .utils import *

def draw_labeled_multigraph(G, attr_name, ax=None):
    """
    Draws a MultiGraph with multi-edges (multiple edges between nodes) and labels the edges
    based on an attribute (e.g., 'w' for weight, 'colors'). Each edge has two colors in this example.
    Based on the example https://networkx.org/documentation/stable/auto_examples/drawing/plot_multigraphs.html
    """
    # Define color mapping for two colors per edge
    color_mapping = {0: 'blue', 1: 'red', 2: 'green', 3: 'yellow', 4: 'purple'}
    #Tried to plot each edge with 2 colors but could not do it. It is better to print the label.
    connectionstyle = [f"arc3,rad={r}" for r in it.accumulate([0.15] * 4)]
    # ordered_nodes = sorted(G.nodes())
    # pos = nx.circular_layout(G)  # You can change the layout if needed
    # pos = {n: pos[n] for n in ordered_nodes}
    # Define node layout manually (clockwise order)
    ordered_nodes = sorted(G.nodes())  # You can customize this order
    n = len(ordered_nodes)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)[::-1]  # Clockwise
    pos = {
        node: (np.cos(theta), np.sin(theta))
        for node, theta in zip(ordered_nodes, angles)
    }
    nx.draw_networkx_nodes(G, pos, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
    nx.draw_networkx_edges(
        G, pos, edge_color="grey", connectionstyle=connectionstyle, ax=ax
    )

    # Prepare edge labels
    labels = {
        tuple(edge): f"{edge}, {attr_name}:{attrs[attr_name]}"
        for *edge, attrs in G.edges(keys=True, data=True)
    }
    #labels
    nx.draw_networkx_edge_labels(
        G,
        pos,
        labels,
        connectionstyle=connectionstyle,
        label_pos=0.3,
        font_color="blue",
        bbox={"alpha": 0},
        ax=ax,
    )
def plot_graph(edge_vec):
    G = nx.MultiGraph()
    for (a, b), (c1, c2) in edge_vec:
        G.add_edge(a, b, colors=(c1, c2))
    fig, ax = plt.subplots(figsize=(10, 10))
    draw_labeled_multigraph(G, 'colors', ax=ax) #Plots the colors
    plt.savefig("graph.svg", format='svg', dpi=600)

def histo_fidelity(target_state, ordered_states, num_colors):
    """
    Plots a histogram of reward fidelities for a list of states compared to a target state.
    Args:
        target_state (list): The target state to compare against.
        ordered_states (list): A list of states to compute fidelities for.
    """
    # Step 1: Compute reward fidelities
    fidelities = [reward_fidelity(target_state, s, num_colors) for s in ordered_states]
    # Step 2: Plot histogram
    plt.figure(figsize=(8, 5))
    plt.hist(fidelities, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Reward Fidelity')
    plt.ylabel('Number of States')
    plt.title('Histogram of Reward Fidelities')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("histogram.svg", format='svg', dpi=600)

def plot_loss_curve(figure, losses_A, title=""):
    filename = f"{figure}_loss.svg"
    plt.figure(figsize=(10,5))
    plt.plot(losses_A, color="black")
    plt.savefig(filename, format='svg', dpi=600)

def plot_rewards(rewards):

    # Best-so-far (cumulative max)
    best = []
    cur = None
    for r in rewards:
        cur = r if cur is None or r > cur else cur
        best.append(cur)

    x = list(range(1, len(rewards) + 1))

    plt.figure(figsize=(8, 5))
    plt.plot(x, best, linewidth=1.5)
    plt.xlabel("Iteration")
    plt.ylabel("Best reward")
    plt.savefig("rewards_progress.svg", format="svg", dpi=600, bbox_inches="tight")

