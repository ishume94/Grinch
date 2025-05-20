import torch
from tqdm import tqdm

from gfn.gflownet import TBGFlowNet
from gfn.gym import HyperGrid  # We use the hyper grid environment
from gfn.preprocessors import KHotPreprocessor
from gfn.modules import DiscretePolicyEstimator
from gfn.samplers import Sampler
from gfn.utils.modules import MLP  # is a simple multi-layer perceptron (MLP)

# 1 - We define the environment.
env = HyperGrid(ndim=4, height=8, R0=0.01)  # Grid of size 8x8x8x8
preprocessor = KHotPreprocessor(ndim=env.ndim, height=env.height)

# 2 - We define the needed modules (neural networks).
module_PF = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions
)  # Neural network for the forward policy, with as many outputs as there are actions

module_PB = MLP(
    input_dim=preprocessor.output_dim,
    output_dim=env.n_actions - 1,
    trunk=module_PF.trunk  # We share all the parameters of P_F and P_B, except for the last layer
)

# 3 - We define the estimators.
pf_estimator = DiscretePolicyEstimator(module_PF, env.n_actions, is_backward=False, preprocessor=preprocessor)
pb_estimator = DiscretePolicyEstimator(module_PB, env.n_actions, is_backward=True, preprocessor=preprocessor)

# 4 - We define the GFlowNet.
gfn = TBGFlowNet(logZ=0., pf=pf_estimator, pb=pb_estimator)  # We initialize logZ to 0

# 5 - We define the sampler and the optimizer.
sampler = Sampler(estimator=pf_estimator)  # We use an on-policy sampler, based on the forward policy

# Different policy parameters can have their own LR.
# Log Z gets dedicated learning rate (typically higher).
optimizer = torch.optim.Adam(gfn.pf_pb_parameters(), lr=1e-3)
optimizer.add_param_group({"params": gfn.logz_parameters(), "lr": 1e-1})

# 6 - We train the GFlowNet for 1000 iterations, with 16 trajectories per iteration
for i in (pbar := tqdm(range(1000))):
    trajectories = sampler.sample_trajectories(env=env, n=16, save_logprobs=True)  # The save_logprobs=True makes on-policy training faster
    optimizer.zero_grad()
    loss = gfn.loss(env, trajectories)
    loss.backward()
    optimizer.step()
    if i % 25 == 0:
        pbar.set_postfix({"loss": loss.item()})
