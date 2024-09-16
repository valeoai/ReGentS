import jax
from jax import numpy as jnp
import optax
import logging

from waymax.utils import geometry
from waymax import datatypes

from simulation import simulate_scenario_with_actions
from utils import flatten_actions, unflatten_actions, mask_invalid_traj, get_traj_from_state_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIELDS_5DOF = ['x', 'y', 'length', 'width', 'yaw']
FIELDS_3DOF = ['x', 'y', 'yaw']

def wrap_to_pi(angles):
    return (angles + jnp.pi) % (2 * jnp.pi) - jnp.pi

def regularize_actions(env, scenario, actions, solver, max_steps=4000):
    actions_data, actions_valid = flatten_actions(actions)
    opt_state = solver.init(actions_data)
    loss_grad_fn = jax.value_and_grad(loss_traj_regularize)

    for step in range(max_steps):
        loss_val, grad_actions_data = loss_grad_fn(actions_data, actions_valid, env, scenario)
        updates, opt_state = solver.update(grad_actions_data, opt_state, actions_data)
        actions_data = optax.apply_updates(actions_data, updates)

        if step % 10 == 0:
            logger.info(f'{step=}, {loss_val=}')
    actions = unflatten_actions(actions_data, actions_valid)
    return actions

def loss_traj_regularize(actions_data, actions_valid, env, scenario):
    actions = unflatten_actions(actions_data, actions_valid)
    states = simulate_scenario_with_actions(env, scenario, actions)
    log_trajectory = states[0].log_trajectory

    targ_traj = log_trajectory.xy
    pred_traj = jnp.concat([state.current_sim_trajectory.xy for state in states], axis=-2)

    diff_traj = (pred_traj - targ_traj) * mask_invalid_traj(targ_traj)
    return jnp.sum(diff_traj ** 2)

@jax.jit
def overlap_with_ego(states, ego_idx, adv_idx):
    ego_traj, _ = get_traj_from_state_list(states, ego_idx, FIELDS_5DOF, keepdim=False)
    adv_trajs, _ = get_traj_from_state_list(states, adv_idx, FIELDS_5DOF)
    
    def unbatched_overlap_with_ego(ego_traj, adv_traj):
        trajs = jnp.stack([ego_traj, adv_traj], axis=0)
        trajs = jnp.swapaxes(trajs, 0, 1)
        pairwise_overlap = geometry.compute_pairwise_overlaps(trajs)
        return jnp.any(pairwise_overlap)

    batched_fn = jax.vmap(unbatched_overlap_with_ego, [None, 0])
    return batched_fn(ego_traj, adv_trajs)

def get_drivable_area_map(roadgraph_points: datatypes.RoadgraphPoints, resolution=1.0):
    is_road_edge = datatypes.is_road_edge(roadgraph_points.types)
    valid_road_edge = jnp.logical_and(roadgraph_points.valid, is_road_edge)
    xy = roadgraph_points.xy[jnp.logical_and(is_road_edge, roadgraph_points.valid)]
    
    road_edge_xys = roadgraph_points.xy[valid_road_edge]
    road_edge_dir_xys = roadgraph_points.dir_xy[valid_road_edge]
    road_edge_ids = roadgraph_points.ids[valid_road_edge]
    
    def compute_signed_distance_to_nearest_road_edge_point(query_point):
        differences = road_edge_xys - jnp.expand_dims(query_point, axis=-2)
        square_distances = jnp.sum(differences ** 2, axis=-1)
        nearest_indices = jnp.argmin(square_distances, axis=-1)

        prior_indices = jnp.maximum(
            jnp.zeros_like(nearest_indices), nearest_indices - 1
        )

        nearest_xys = road_edge_xys[nearest_indices]
        nearest_dir_xys = road_edge_dir_xys[nearest_indices]

        prev_nearest_dir_xys = road_edge_dir_xys[prior_indices]
        points_to_edge = query_point - nearest_xys

        cross_product = jnp.cross(points_to_edge, nearest_dir_xys)
        cross_product_prior = jnp.cross(points_to_edge, prev_nearest_dir_xys)
        prior_point_in_same_curve = jnp.equal(
            road_edge_ids[nearest_indices], road_edge_ids[prior_indices]
        )

        offroad_sign = jnp.sign(
            jnp.where(
                jnp.logical_and(prior_point_in_same_curve, cross_product_prior < cross_product),
                cross_product_prior,
                cross_product,
            )
        )

        return (
            jnp.sum(points_to_edge ** 2, axis=-1) * offroad_sign
        )

    x_range = jnp.arange(xy[:, 0].min(), xy[:, 0].max(), resolution)
    y_range = jnp.arange(xy[:, 1].min(), xy[:, 1].max(), resolution)

    grid = jnp.stack(jnp.meshgrid(x_range, y_range), axis=-1)
    distances = jax.lax.map(compute_signed_distance_to_nearest_road_edge_point, grid)
    return (distances > 0.0).astype(jnp.float32), (x_range, y_range, grid)