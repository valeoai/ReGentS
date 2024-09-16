import jax
from jax import numpy as jnp
import optax
import logging

from simulation import simulate_scenario_with_aim_bev_and_actions
from cost import calculate_distance_ego_col, calculate_distance_adv_col, calculate_potential_adv_dev
from utils import flatten_actions, unflatten_actions, get_traj_from_state_list, debug_info 
from method.utils import FIELDS_5DOF, overlap_with_ego, get_drivable_area_map, wrap_to_pi

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def opt(actions, env, torch_model, scenario, solver, cfg):
    obj_idx = jnp.arange(scenario.num_objects)
    ego_idx = jnp.where(scenario.object_metadata.is_sdc)[0][0]
    static_vehicle_idx = jnp.array([], dtype=int)
    convergent_idx = jnp.array([], dtype=int)
    invalid_idx = jnp.where(jnp.mean(scenario.log_trajectory.valid, axis=-1) < 0.5)[0]
    non_vehicle_idx = jnp.where(scenario.object_metadata.object_types != 1)[0]

    if cfg.type == 'regents':
        dist_vehicle = jnp.linalg.norm(scenario.log_trajectory.xy[:, -1] - scenario.log_trajectory.xy[:, 0], axis=-1)
        static_vehicle_idx = jnp.where(dist_vehicle < 0.2)[0]

        displacement = scenario.log_trajectory.xy - scenario.log_trajectory.xy[ego_idx, jnp.newaxis]

        initial_angle = jnp.arctan2(displacement[..., 1], displacement[..., 0])
        delta_angle_convergent = wrap_to_pi(initial_angle - scenario.log_trajectory.yaw[ego_idx, jnp.newaxis])
        
        convergent = jnp.mean(((delta_angle_convergent > 7*jnp.pi/8) | (delta_angle_convergent < -7 * jnp.pi/8)), axis=-1) > 0.8
        convergent_idx = jnp.where(convergent)[0]

    non_controlled_idx = jnp.concatenate((jnp.expand_dims(ego_idx, axis=0), non_vehicle_idx, static_vehicle_idx, convergent_idx, invalid_idx))
    adv_idx = jnp.delete(obj_idx, non_controlled_idx)
    logger.info(f'{adv_idx=}, {convergent_idx=}')
    
    max_steps = cfg.max_steps

    actions_data, actions_valid = flatten_actions(actions)
    for i, action_data in enumerate(actions_data):
        actions_data[i] = jnp.where(action_data == 0.0, 1e-6, action_data)
    drivable_area_map = get_drivable_area_map(scenario.roadgraph_points, resolution=cfg.drivable_area_map_resolution)

    opt_state = solver.init(actions_data)
    
    for step in range(max_steps):
        logger.info(f'Optim {step=}')
        if step % 1 == 0:
            loss_grad_fn = jax.value_and_grad(loss_king, has_aux=True)
            (_, (sim_traj, overlap)), grad_actions_data = loss_grad_fn(actions_data, actions_valid, ego_idx, adv_idx, env, scenario, drivable_area_map, torch_model, cfg.loss)
            
            if cfg.type == 'regents':
                delta_yaw = wrap_to_pi(sim_traj.yaw - sim_traj.yaw[ego_idx, jnp.newaxis])
                displacements = sim_traj.xy - sim_traj.xy[ego_idx, jnp.newaxis]
                angles = jnp.arctan2(displacements[..., 1], displacements[..., 0])
                delta_angle_divergent = wrap_to_pi(angles - sim_traj.yaw[ego_idx, jnp.newaxis])
                divergent = jnp.mean(((delta_angle_divergent / delta_yaw) < 1) & ((delta_angle_divergent / delta_yaw) > 0) & ((delta_angle_divergent < jnp.pi/8) & (delta_angle_divergent > -jnp.pi/8)) & ((delta_yaw < jnp.pi/2) & (delta_yaw > -jnp.pi/2)), axis=-1) > 0.5
                logger.info(f'{divergent=}')

            if any(jnp.isnan(grad).any() or jnp.isinf(grad).any() for grad in grad_actions_data):
                logger.error('NaN or infinite gradient detected, stopping optimization.')
                break
        
        if overlap:
            logger.info('************ Collision ************')
            break
            
        updates, opt_state = solver.update(grad_actions_data, opt_state, actions_data)

        for t in range(len(updates)):
            updates[t] = updates[t].at[ego_idx].set(0.0)
            if cfg.type == 'regents':
                new_grad = jnp.where(divergent, 0.0 * updates[t][:, 1], 0.5 * updates[t][:, 1])
                updates[t] = updates[t].at[:, 1].set(new_grad)

        actions_data = optax.apply_updates(actions_data, updates)
        
    actions = unflatten_actions(actions_data, actions_valid)
    return actions

def loss_king_traj(states, ego_idx, adv_idx, drivable_area_map, cfg):
    ego_traj,  valid_ego_traj  = get_traj_from_state_list(states, ego_idx, FIELDS_5DOF, keepdim=False)
    adv_trajs, valid_adv_trajs = get_traj_from_state_list(states, adv_idx, FIELDS_5DOF)
    loss_ego_col = calculate_distance_ego_col(ego_traj, adv_trajs, valid_ego_traj, valid_adv_trajs)
    loss_adv_col = -calculate_distance_adv_col(adv_trajs, valid_adv_trajs, cfg.adv_col.threshold) if 'adv_col' in cfg else 0.0
    loss_adv_dev = calculate_potential_adv_dev(adv_trajs, valid_adv_trajs, drivable_area_map, cfg.adv_dev.cropsize) if 'adv_dev' in cfg else 0.0
    return loss_ego_col, loss_adv_col, loss_adv_dev

def loss_king(actions_data, actions_valid, ego_idx, adv_idx, env, scenario, drivable_area_map, torch_model, cfg):
    actions = unflatten_actions(actions_data, actions_valid)

    states = simulate_scenario_with_aim_bev_and_actions(env, torch_model, scenario, actions)
    loss_ego_col, loss_adv_col, loss_adv_dev = loss_king_traj(states, ego_idx, adv_idx, drivable_area_map, cfg)
    # debug_info(logger, 'loss_ego_col: {}, loss_adv_col: {}, loss_adv_dev: {}', loss_ego_col,  loss_adv_col, loss_adv_dev)
    loss_total = loss_ego_col + cfg.adv_col.coeff * loss_adv_col + cfg.adv_dev.coeff * loss_adv_dev

    collided = jnp.any(overlap_with_ego(states, ego_idx, adv_idx))
    
    return loss_total, (states[-1].sim_trajectory, collided)

