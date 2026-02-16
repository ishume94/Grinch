from .gflow_utils import *
from .utils import *
import os
from tqdm import tqdm
from torch.distributions.categorical import Categorical

def _make_tb_optimizer(model, learning_rate, logZ_lr_mult):
    if logZ_lr_mult <= 0:
        raise ValueError("logZ_lr_mult must be > 0.")

    named_params = list(model.named_parameters())
    logZ_params = [
        param for name, param in named_params
        if name == "logZ" or name.endswith(".logZ")
    ]
    non_logZ_params = [
        param for name, param in named_params
        if not (name == "logZ" or name.endswith(".logZ"))
    ]

    if not logZ_params:
        return torch.optim.Adam(model.parameters(), learning_rate)

    param_groups = []
    if non_logZ_params:
        param_groups.append({"params": non_logZ_params, "lr": learning_rate})
    param_groups.append({"params": logZ_params, "lr": learning_rate * logZ_lr_mult})
    return torch.optim.Adam(param_groups)


def _save_tb_snapshot(snapshot_dir, snapshot_prefix, episode, model_kind, model_kwargs, model):
    snapshot_path = os.path.join(snapshot_dir, f"{snapshot_prefix}_ep{episode:06d}.pth")
    torch.save(
        {
            "episode": int(episode),
            "model_kind": model_kind,
            "model_state_dict": model.state_dict(),
            "model_kwargs": model_kwargs,
        },
        snapshot_path,
    )


def TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    model,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    pruning: bool = True,
    logZ_lr_mult: float = 10.0,
    save_snapshot_history: bool = False,
    snapshot_every: int = 10,
    snapshot_dir: str = "tb_snapshots",
    snapshot_prefix: str = "TB_snapshot",
):
    set_seed(seed)

    if save_snapshot_history and snapshot_every <= 0:
        raise ValueError("snapshot_every must be > 0 when save_snapshot_history=True.")

    # Instantiate model and optimizer
    opt = _make_tb_optimizer(model, learning_rate, logZ_lr_mult)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)

    model_class = model.__class__.__name__.lower()
    if model_class == "tbmodel":
        model_kind = "tb"
        hidden_dim = None
        try:
            hidden_dim = int(model.mlp[0].out_features)
        except Exception:
            pass
        model_kwargs = {"hidden_dim": hidden_dim} if hidden_dim is not None else {}
    elif model_class == "embtbmodel":
        model_kind = "embtb"
        hidden_dim = None
        n_emb = None
        try:
            hidden_dim = int(model.encode_layer[0].out_features)
        except Exception:
            pass
        try:
            n_emb = int(model.emb_layer.embedding_dim)
        except Exception:
            pass
        model_kwargs = {}
        if hidden_dim is not None:
            model_kwargs["hidden_dim"] = hidden_dim
        if n_emb is not None:
            model_kwargs["n_emb"] = n_emb
    else:
        model_kind = model_class
        model_kwargs = {}

    if save_snapshot_history:
        os.makedirs(snapshot_dir, exist_ok=True)
        _save_tb_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
            episode=0,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
            model=model,
        )

    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_states, logZs, rewards = [], [], [], []
    pruned_states = []
    minibatch_loss = 0

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []  # Each episode starts with an empty state.
        P_F_s, P_B_s = model(state_to_tensor(state,FEATURE_KEYS), FEATURE_KEYS)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(max_edges):  # All trajectories are length 2 (not including s0).
            mask = calculate_forward_mask_from_state(state,target_expr, FEATURE_KEYS)#calculate_forward_mask_from_state(state)
            P_F_s = torch.where(mask, P_F_s, -1000)  # Removes invalid forward actions.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state = state + [FEATURE_KEYS[action]] # "Go" to next state.
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            if t == max_edges-1:  # End of trajectory.
                #reward = reward_f(new_state)#torch.tensor(reward(new_state)).float()
                reward, opt_weights = reward_fidelity(target_state, new_state, num_colors, pruning)
                #print("Reward",reward)
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(state_to_tensor(new_state,FEATURE_KEYS), FEATURE_KEYS)
            #print(new_state)
            mask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(mask, P_B_s, -1000)  # Removes invalid backward actions.
            P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        # We're done with the episode, add the face to the list, and if we are at an
        # update episode, take a gradient step.
        sampled_states.append(state)
        rewards.append(reward)
        # Prune states with high fidelity :)
        if pruning and reward is not None and reward >= 0.99 and opt_weights is not None:
            pruned_state, pruned_weights, keep_mask = prune_state_by_weight(
                state, opt_weights,
                weight_eps=1e-2,
                keep_at_least=4
            )

            # recompute fidelity of the pruned state using the pruned weights
            try:
                pruned_fid = compute_fidelity(pruned_weights, pruned_state, target_state, num_colors)
            except Exception:
                pruned_fid = 0.0

            if pruned_fid >= 0.99: #In theseus they use 0.95 as fidelity limit! We can do better
                #print(pruned_state)
                #print("Fidelity after pruning:", pruned_fid)
                pruned_states.append(pruned_state)
            
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            "model_kind": model_kind,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            "model_kwargs": model_kwargs,
            }, "TBmodel.pth")

        if save_snapshot_history and ((episode + 1) % snapshot_every == 0 or episode == n_episodes - 1):
            _save_tb_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
                episode=episode + 1,
                model_kind=model_kind,
                model_kwargs=model_kwargs,
                model=model,
            )
    
    return sampled_states, losses, logZs, rewards, pruned_states

def GIN_TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    n_hid_units,
    pruning: bool = True,
    logZ_lr_mult: float = 10.0,
    save_snapshot_history: bool = False,
    snapshot_every: int = 10,
    snapshot_dir: str = "tb_snapshots",
    snapshot_prefix: str = "GINTB_snapshot",
):
    set_seed(seed)

    if save_snapshot_history and snapshot_every <= 0:
        raise ValueError("snapshot_every must be > 0 when save_snapshot_history=True.")

    # Instantiate model and optimizer
    model = GIN_TBModel(num_nodes, n_hid_units, FEATURE_KEYS)
    opt = _make_tb_optimizer(model, learning_rate, logZ_lr_mult)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)
    model_kind = "gin"
    model_kwargs = {
        "node_feat_dim": num_nodes,
        "hidden_dim": n_hid_units,
    }

    if save_snapshot_history:
        os.makedirs(snapshot_dir, exist_ok=True)
        _save_tb_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
            episode=0,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
            model=model,
        )

    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_states, logZs, rewards = [], [], [], []
    pruned_states = []
    minibatch_loss = 0

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []  # Each episode starts with an empty state.
        graph_data = state_to_data(state, num_nodes, num_colors)
        P_F_s, P_B_s = model(graph_data)
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(max_edges):  # All trajectories are length 2 (not including s0).
            mask = calculate_forward_mask_from_state(state,target_expr, FEATURE_KEYS)#calculate_forward_mask_from_state(state)
            P_F_s = torch.where(mask, P_F_s, -1000)  # Removes invalid forward actions.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state = state + [FEATURE_KEYS[action]] # "Go" to next state.
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            if t == max_edges-1:  # End of trajectory.
                #reward = reward_f(new_state)#torch.tensor(reward(new_state)).float()
                reward, opt_weights = reward_fidelity(target_state, new_state, num_colors, pruning)
                #print("Reward",reward)
            # We recompute P_F and P_B for new_state.
            graph_data = state_to_data(new_state, num_nodes, num_colors)
            P_F_s, P_B_s = model(graph_data)
            #print(new_state)
            mask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(mask, P_B_s, -1000)  # Removes invalid backward actions.
            P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        # We're done with the episode, add the face to the list, and if we are at an
        # update episode, take a gradient step.
        sampled_states.append(state)
        rewards.append(reward)
        # Prune states with high fidelity :)
        if pruning and reward is not None and reward >= 0.99 and opt_weights is not None:
            pruned_state, pruned_weights, keep_mask = prune_state_by_weight(
                state, opt_weights,
                weight_eps=1e-2,
                keep_at_least=4
            )

            # recompute fidelity of the pruned state using the pruned weights
            try:
                pruned_fid = compute_fidelity(pruned_weights, pruned_state, target_state, num_colors)
            except Exception:
                pruned_fid = 0.0

            if pruned_fid >= 0.99: #In theseus they use 0.95 as fidelity limit! We can do better
                #print(pruned_state)
                #print("Fidelity after pruning:", pruned_fid)
                pruned_states.append(pruned_state)
            

        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            "model_kind": model_kind,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            "model_kwargs": model_kwargs,
            }, "GINTBmodel.pth")

        if save_snapshot_history and ((episode + 1) % snapshot_every == 0 or episode == n_episodes - 1):
            _save_tb_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
                episode=episode + 1,
                model_kind=model_kind,
                model_kwargs=model_kwargs,
                model=model,
            )
    
    return sampled_states, losses, logZs, rewards, pruned_states

def GINE_TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    n_hid_units,
    edge_feat_dim,
    pruning: bool = True,
    logZ_lr_mult: float = 10.0,
    save_snapshot_history: bool = False,
    snapshot_every: int = 10,
    snapshot_dir: str = "tb_snapshots",
    snapshot_prefix: str = "GINETB_snapshot",
):
    set_seed(seed)

    if save_snapshot_history and snapshot_every <= 0:
        raise ValueError("snapshot_every must be > 0 when save_snapshot_history=True.")

    # Instantiate model and optimizer
    model = GINE_TBModel(num_nodes, edge_feat_dim,n_hid_units, FEATURE_KEYS)
    opt = _make_tb_optimizer(model, learning_rate, logZ_lr_mult)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)
    model_kind = "gine"
    model_kwargs = {
        "node_feat_dim": num_nodes,
        "edge_feat_dim": edge_feat_dim,
        "hidden_dim": n_hid_units,
    }

    if save_snapshot_history:
        os.makedirs(snapshot_dir, exist_ok=True)
        _save_tb_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
            episode=0,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
            model=model,
        )

    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_states, logZs, rewards = [], [], [], []
    pruned_states = []
    minibatch_loss = 0

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []  # Each episode starts with an empty state.
        graph_data = state_to_data(state, num_nodes, num_colors)
        P_F_s, P_B_s = model(graph_data)
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(max_edges):  # All trajectories are length 2 (not including s0).
            mask = calculate_forward_mask_from_state(state,target_expr, FEATURE_KEYS)#calculate_forward_mask_from_state(state)
            P_F_s = torch.where(mask, P_F_s, -1000)  # Removes invalid forward actions.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state = state + [FEATURE_KEYS[action]] # "Go" to next state.
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            if t == max_edges-1:  # End of trajectory.
                #reward = reward_f(new_state)#torch.tensor(reward(new_state)).float()
                reward, opt_weights = reward_fidelity(target_state, new_state, num_colors, pruning)
                #print("Reward",reward)
            # We recompute P_F and P_B for new_state.
            graph_data = state_to_data(new_state, num_nodes, num_colors)
            P_F_s, P_B_s = model(graph_data)
            #print(new_state)
            mask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(mask, P_B_s, -1000)  # Removes invalid backward actions.
            P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
            total_log_P_B += Categorical(logits=P_B_s).log_prob(action)

            state = new_state  # Continue iterating.

        # We're done with the trajectory, let's compute its loss. Since the reward
        # can sometimes be zero, instead of log(0) we'll clip the log-reward to -20.
        minibatch_loss += trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        # We're done with the episode, add the face to the list, and if we are at an
        # update episode, take a gradient step.
        sampled_states.append(state)
        rewards.append(reward)
        # Prune states with high fidelity :)
        if pruning and reward is not None and reward >= 0.99 and opt_weights is not None:
            pruned_state, pruned_weights, keep_mask = prune_state_by_weight(
                state, opt_weights,
                weight_eps=1e-2,
                keep_at_least=4
            )

            # recompute fidelity of the pruned state using the pruned weights
            try:
                pruned_fid = compute_fidelity(pruned_weights, pruned_state, target_state, num_colors)
            except Exception:
                pruned_fid = 0.0

            if pruned_fid >= 0.99: #In theseus they use 0.95 as fidelity limit! We can do better
                #print(pruned_state)
                #print("Fidelity after pruning:", pruned_fid)
                pruned_states.append(pruned_state)
            
        if episode % update_freq == 0:
            losses.append(minibatch_loss.item())
            logZs.append(model.logZ.item())
            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()
            minibatch_loss = 0
            torch.save({
            'epoch': episode,
            "model_kind": model_kind,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            "model_kwargs": model_kwargs,
            }, "GINETBmodel.pth")

        if save_snapshot_history and ((episode + 1) % snapshot_every == 0 or episode == n_episodes - 1):
            _save_tb_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
                episode=episode + 1,
                model_kind=model_kind,
                model_kwargs=model_kwargs,
                model=model,
            )

    return sampled_states, losses, logZs, rewards, pruned_states

def GAT_TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    n_hid_units,
    edge_feat_dim,
    pruning: bool = True,
    gat_heads: int = 4,
    gat_layers: int = 3,
    dropout: float = 0.0,
    logZ_lr_mult: float = 10.0,
    save_snapshot_history: bool = False,
    snapshot_every: int = 10,
    snapshot_dir: str = "tb_snapshots",
    snapshot_prefix: str = "GATTB_snapshot",
):
    set_seed(seed)

    if save_snapshot_history and snapshot_every <= 0:
        raise ValueError("snapshot_every must be > 0 when save_snapshot_history=True.")

    model = GAT_TBModel(
        node_feat_dim=num_nodes,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=n_hid_units,
        FEATURE_KEYS=FEATURE_KEYS,
        num_layers=gat_layers,
        heads=gat_heads,
        dropout=dropout,
    )
    opt = _make_tb_optimizer(model, learning_rate, logZ_lr_mult)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)
    model_kind = "gat"
    model_kwargs = {
        "node_feat_dim": num_nodes,
        "edge_feat_dim": edge_feat_dim,
        "hidden_dim": n_hid_units,
        "num_layers": gat_layers,
        "heads": gat_heads,
        "dropout": dropout,
    }

    if save_snapshot_history:
        os.makedirs(snapshot_dir, exist_ok=True)
        _save_tb_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
            episode=0,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
            model=model,
        )

    losses, sampled_states, logZs, rewards = [], [], [], []
    pruned_states = []
    minibatch_loss = 0.0

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []
        graph_data = state_to_data(state, num_nodes, num_colors)
        P_F_s, P_B_s = model(graph_data)

        total_log_P_F = 0.0
        total_log_P_B = 0.0

        for t in range(max_edges):
            mask = calculate_forward_mask_from_state(state, target_expr, FEATURE_KEYS)
            P_F_s = torch.where(mask, P_F_s, torch.tensor(-1000.0, device=P_F_s.device, dtype=P_F_s.dtype))
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100.0), P_F_s)

            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            action_idx = int(action.item())

            new_state = state + [FEATURE_KEYS[action_idx]]
            total_log_P_F = total_log_P_F + categorical.log_prob(action)

            if t == max_edges - 1:
                reward, opt_weights = reward_fidelity(target_state, new_state, num_colors, pruning)

            graph_data = state_to_data(new_state, num_nodes, num_colors)
            P_F_s, P_B_s = model(graph_data)

            bmask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(bmask, P_B_s, torch.tensor(-1000.0, device=P_B_s.device, dtype=P_B_s.dtype))
            P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100.0), P_B_s)
            total_log_P_B = total_log_P_B + Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        minibatch_loss = minibatch_loss + trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        sampled_states.append(state)
        rewards.append(reward)
        # Prune states with high fidelity :)
        if pruning and reward is not None and reward >= 0.99 and opt_weights is not None:
            pruned_state, pruned_weights, keep_mask = prune_state_by_weight(
                state, opt_weights,
                weight_eps=1e-2,
                keep_at_least=4
            )

            # recompute fidelity of the pruned state using the pruned weights
            try:
                pruned_fid = compute_fidelity(pruned_weights, pruned_state, target_state, num_colors)
            except Exception:
                pruned_fid = 0.0

            if pruned_fid >= 0.99: #In theseus they use 0.95 as fidelity limit! We can do better
                #print(pruned_state)
                #print("Fidelity after pruning:", pruned_fid)
                pruned_states.append(pruned_state)

        if episode % update_freq == 0:
            losses.append(float(minibatch_loss.item()))
            logZs.append(float(model.logZ.item()))

            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()

            minibatch_loss = 0.0

            torch.save(
                {
                    "epoch": episode,
                    "model_kind": model_kind,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": losses,
                    "model_kwargs": model_kwargs,
                },
                "GATTBmodel.pth",
            )

        if save_snapshot_history and ((episode + 1) % snapshot_every == 0 or episode == n_episodes - 1):
            _save_tb_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
                episode=episode + 1,
                model_kind=model_kind,
                model_kwargs=model_kwargs,
                model=model,
            )
    return sampled_states, losses, logZs, rewards, pruned_states

def Transformer_TB_train(
    num_colors,
    num_nodes,
    FEATURE_KEYS,
    target_expr,
    target_state,
    n_episodes,
    learning_rate,
    decay_rate,
    seed,
    update_freq,
    max_edges,
    n_hid_units,
    edge_feat_dim,
    pruning: bool = True,
    tf_heads: int = 4,
    tf_layers: int = 3,
    dropout: float = 0.0,
    logZ_lr_mult: float = 10.0,
    save_snapshot_history: bool = False,
    snapshot_every: int = 10,
    snapshot_dir: str = "tb_snapshots",
    snapshot_prefix: str = "TransformerTB_snapshot",
):
    set_seed(seed)

    if save_snapshot_history and snapshot_every <= 0:
        raise ValueError("snapshot_every must be > 0 when save_snapshot_history=True.")

    model = Transformer_TBModel(
        node_feat_dim=num_nodes,
        edge_feat_dim=edge_feat_dim,
        hidden_dim=n_hid_units,
        FEATURE_KEYS=FEATURE_KEYS,
        num_layers=tf_layers,
        heads=tf_heads,
        dropout=dropout,
    )
    opt = _make_tb_optimizer(model, learning_rate, logZ_lr_mult)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)
    model_kind = "transformer"
    model_kwargs = {
        "node_feat_dim": num_nodes,
        "edge_feat_dim": edge_feat_dim,
        "hidden_dim": n_hid_units,
        "num_layers": tf_layers,
        "heads": tf_heads,
        "dropout": dropout,
    }

    losses, sampled_states, logZs, rewards = [], [], [], []
    pruned_states = [] 
    minibatch_loss = 0.0

    if save_snapshot_history:
        os.makedirs(snapshot_dir, exist_ok=True)
        _save_tb_snapshot(
            snapshot_dir=snapshot_dir,
            snapshot_prefix=snapshot_prefix,
            episode=0,
            model_kind=model_kind,
            model_kwargs=model_kwargs,
            model=model,
        )

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []
        graph_data = state_to_data(state, num_nodes, num_colors)
        P_F_s, P_B_s = model(graph_data)

        total_log_P_F = 0.0
        total_log_P_B = 0.0

        for t in range(max_edges):
            mask = calculate_forward_mask_from_state(state, target_expr, FEATURE_KEYS)
            P_F_s = torch.where(mask, P_F_s, torch.tensor(-1000.0, device=P_F_s.device, dtype=P_F_s.dtype))
            P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100.0), P_F_s)

            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            action_idx = int(action.item())

            new_state = state + [FEATURE_KEYS[action_idx]]
            total_log_P_F = total_log_P_F + categorical.log_prob(action)

            if t == max_edges - 1:
                reward, opt_weights = reward_fidelity(target_state, new_state, num_colors, pruning)

            graph_data = state_to_data(new_state, num_nodes, num_colors)
            P_F_s, P_B_s = model(graph_data)

            bmask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(bmask, P_B_s, torch.tensor(-1000.0, device=P_B_s.device, dtype=P_B_s.dtype))
            P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100.0), P_B_s)
            total_log_P_B = total_log_P_B + Categorical(logits=P_B_s).log_prob(action)

            state = new_state

        minibatch_loss = minibatch_loss + trajectory_balance_loss(
            model.logZ,
            total_log_P_F,
            total_log_P_B,
            reward,
        )

        sampled_states.append(state)
        rewards.append(reward)
        # Prune states with high fidelity :)
        if pruning and reward is not None and reward >= 0.95 and opt_weights is not None:
            # pruned_state, pruned_weights, keep_mask = prune_state_by_weight(
            #     state, opt_weights,
            #     weight_eps=1e-2,
            #     keep_at_least=4
            # )
            pruned_state, pruned_weights, keep_mask = prune_state_by_logic(
                state,
                target_state,
                num_colors,
                weights=opt_weights,
            order="increasing_abs_weight",
            )

            # recompute fidelity of the pruned state using the pruned weights
            try:
                pruned_fid = compute_fidelity(pruned_weights, pruned_state, target_state, num_colors)
            except Exception:
                pruned_fid = 0.0

            if pruned_fid >= 0.95: #In theseus they use 0.95 as fidelity limit! We can do better
                #print(pruned_state)
                #print("Fidelity after pruning:", pruned_fid)
                pruned_states.append(pruned_state)
                

        if episode % update_freq == 0:
            losses.append(float(minibatch_loss.item()))
            logZs.append(float(model.logZ.item()))

            minibatch_loss.backward()
            opt.step()
            opt.zero_grad()
            scheduler.step()

            minibatch_loss = 0.0

            torch.save(
                {
                    "epoch": episode,
                    "model_kind": model_kind,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "loss": losses,
                    "model_kwargs": model_kwargs,
                },
                "TransformerTBmodel.pth",
            )

        if save_snapshot_history and ((episode + 1) % snapshot_every == 0 or episode == n_episodes - 1):
            _save_tb_snapshot(
                snapshot_dir=snapshot_dir,
                snapshot_prefix=snapshot_prefix,
                episode=episode + 1,
                model_kind=model_kind,
                model_kwargs=model_kwargs,
                model=model,
            )

    return sampled_states, losses, logZs, rewards, pruned_states
