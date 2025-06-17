from .gflow_utils import *
from .utils import *
from tqdm import tqdm
from torch.distributions.categorical import Categorical

def TB_train(num_colors, num_nodes, FEATURE_KEYS, target_expr, target_state, model, n_episodes, learning_rate, decay_rate, seed, update_freq, max_edges):
    set_seed(seed)

    # Instantiate model and optimizer
    opt = torch.optim.Adam(model.parameters(), learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay_rate)

    # To not complicate the code, I'll just accumulate losses here and take a
    # gradient step every `update_freq` episode (at the end of each trajectory).
    losses, sampled_states, logZs = [], [], []
    minibatch_loss = 0

    for episode in tqdm(range(n_episodes), ncols=40):
        state = []  # Each episode starts with an empty state.
        P_F_s, P_B_s = model(state_to_tensor(state,FEATURE_KEYS), FEATURE_KEYS)  # Forward and backward policy
        total_log_P_F, total_log_P_B = 0, 0

        for t in range(max_edges):  # All trajectories are length 2 (not including s0).
            mask = calculate_forward_mask_from_state(state,target_expr, FEATURE_KEYS)#calculate_forward_mask_from_state(state)
            P_F_s = torch.where(mask, P_F_s, -100)  # Removes invalid forward actions.
            # Here P_F is logits, so we use Categorical to compute a softmax.
            #P_F_s = torch.where(torch.isnan(P_F_s), torch.full_like(P_F_s, -100), P_F_s)
            categorical = Categorical(logits=P_F_s)
            action = categorical.sample()
            new_state = state + [FEATURE_KEYS[action]] # "Go" to next state.
            total_log_P_F += categorical.log_prob(action)  # Accumulate the log_P_F sum.

            if t == max_edges-1:  # End of trajectory.
                #reward = reward_f(new_state)#torch.tensor(reward(new_state)).float()
                reward = reward_fidelity(target_state, new_state, num_colors)
                #print("Reward",reward)
            # We recompute P_F and P_B for new_state.
            P_F_s, P_B_s = model(state_to_tensor(new_state,FEATURE_KEYS), FEATURE_KEYS)
            #print(new_state)
            mask = calculate_backward_mask_from_state(new_state, FEATURE_KEYS)
            P_B_s = torch.where(mask, P_B_s, -100)  # Removes invalid backward actions.
            #P_B_s = torch.where(torch.isnan(P_B_s), torch.full_like(P_B_s, -100), P_B_s)
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
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': losses,
            }, "TBmodel.pth")
    
    return sampled_states, losses, logZs
