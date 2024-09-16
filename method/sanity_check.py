import jax
from jax import numpy as jnp
import optax
import logging

from waymax.utils import geometry

from simulation import simulate_scenario_with_actions
from cost import calculate_distance_ego_col
from utils import flatten_actions, unflatten_actions, get_traj_from_state_list
from method.utils import FIELDS_5DOF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def opt_sanity_check(actions, env, scenario, solver, cfg):
    ego_idx = cfg.ego_idx
    adv_idx = cfg.adv_idx
    max_steps = cfg.max_steps
    actions_data, actions_valid = flatten_actions(actions)
    opt_state = solver.init(actions_data)
    loss_grad_fn = jax.value_and_grad(loss_chosen_adv, has_aux=True)
    
    for step in range(max_steps):
        logger.info(f'Optim {step=}')
        (_, overlap), grad_actions_data = loss_grad_fn(actions_data, actions_valid, ego_idx, adv_idx, env, scenario)
        
        for t in range(len(grad_actions_data)):
            grad_actions_data[t] = grad_actions_data[t].at[ego_idx].set(0.0)

        updates, opt_state = solver.update(grad_actions_data, opt_state, actions_data)
        actions_data = optax.apply_updates(actions_data, updates)

        if overlap:
            logger.info('************ Collision ************')
            break

    actions = unflatten_actions(actions_data, actions_valid)
    return actions

def loss_chosen_adv(actions_data, actions_valid, ego_idx, adv_idx, env, scenario):
    actions = unflatten_actions(actions_data, actions_valid)
    states = simulate_scenario_with_actions(env, scenario, actions)

    trajs = get_traj_from_state_list(states, [ego_idx, adv_idx], FIELDS_5DOF)

    ego_traj = trajs[0]
    adv_traj = trajs[1, None]

    pairwise_overlap = geometry.compute_pairwise_overlaps(jnp.swapaxes(trajs, 0, 1))
    collided = jnp.any(pairwise_overlap)

    return calculate_distance_ego_col(ego_traj, adv_traj).sum(), collided
