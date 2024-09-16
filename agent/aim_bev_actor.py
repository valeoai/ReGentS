from jax import numpy as jnp
import numpy as np
import jax
import torch
from torch.func import functional_call
from torch2jax import tree_t2j, torch2jax_with_vjp

from waymax.agents.actor_core import WaymaxActorCore, WaymaxActorOutput
from waymax.datatypes.observation import sdc_observation_from_state
from waymax import datatypes

from .rasterization_jnp import rasterize_observation_jnp

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_SDC_CONTROL_FUNC = lambda state: state.object_metadata.is_sdc

@jax.jit
def rasterize(state, id_sdc):
    obs = sdc_observation_from_state(state)
    gt_speed = jnp.linalg.norm(state.current_sim_trajectory.vel_xy[id_sdc, 0])
    pose2d = datatypes.dynamic_index(obs.pose2d, index=0, axis=0, keepdims=False)
        
    traj = state.log_trajectory
    traj = datatypes.transform_trajectory(traj, pose2d)
    
    target_point = traj.xy[id_sdc, -1]
    img = rasterize_observation_jnp(obs).astype(jnp.float32) / 255.0 * 2. - 1.

    return img, target_point, gt_speed

class AimBEVActor(WaymaxActorCore):
    def __init__(self, torch_model):
        super().__init__()
        self.torch_model = torch_model
        self.torch_model.eval()
        self.params, self.buffers = dict(self.torch_model.named_parameters()), dict(self.torch_model.named_buffers())
        self.params_jax = dict()
        for k, v in self.params.items():
            self.params_jax[k] = tree_t2j(v.cpu().numpy())

        self.buffers_jax = tree_t2j(self.buffers)
        
        def torch_fwd_fn(params, buffers, input):
            buffers = {k: torch.clone(v) for k, v in buffers.items()}
            return functional_call(self.torch_model, (params, buffers), args=input)

        self.torch_forward_function = torch_fwd_fn
        self.jax_fn = torch2jax_with_vjp(self.torch_forward_function, self.params, self.buffers, (torch.zeros((1, 192, 192, 3)).to(device), torch.zeros((1, 2)).to(device)), nondiff_argnums=(0, 1, 2), depth=1)
        
    def init(self, rng, state):
        return

    def name(self):
        return 'AimBevActor'

    @torch.no_grad()
    def select_action(self, params, state, actor_state, rng):
        del rng, actor_state
        id_sdc = params['id_sdc']

        state = jax.lax.stop_gradient(state)

        img, target_point, gt_speed = rasterize(state, id_sdc)
        img = jnp.expand_dims(img, axis=0)
        target_point = jnp.expand_dims(target_point, axis=0)

        pred_waypoints = self.jax_fn(self.params_jax, self.buffers_jax, (img, target_point))
        steer, throttle, brake = self.torch_model.control_pid(pred_waypoints, gt_speed)
        
        throttle = brake * -1.0 * jnp.ones_like(brake) + (1 - brake) * throttle

        action_data = jnp.zeros((state.num_objects, 2))
        action_data = action_data.at[id_sdc].set(jnp.array([throttle.squeeze(), steer.squeeze()]))
        action_valid = jnp.ones((state.num_objects, 1)) == 1

        action = datatypes.Action(data=action_data, valid=action_valid)

        return WaymaxActorOutput(
            actor_state=None,
            action=action,
            is_controlled=_SDC_CONTROL_FUNC(state)
        )