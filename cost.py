import jax
from jax import numpy as jnp

from functools import partial

def calculate_state_distance_center_points(state_1, state_2):
    return ((state_1[:2] - state_2[:2]) ** 2).sum()

def calculate_box_corner_corrdinates(state):
    yaw = state[..., 4]
    c = jnp.cos(yaw) # (N,)
    s = jnp.sin(yaw) # (N,)
    pt = state[..., 0:2]  # (N, 2)
    length, width = state[..., 2], state[..., 3]
    length_half, width_half = length / 2, width / 2

    u = jnp.stack((c, -s), axis=-1) # (N, 2)
    ut = jnp.stack((s, c), axis=-1)
    rot = jnp.stack((u, ut), axis=-2)

    # # Compute box corner coordinates.
    tl = pt + jnp.einsum('...ij, ...j -> ...i', rot, jnp.stack(( length_half,  width_half), axis=-1))
    tr = pt + jnp.einsum('...ij, ...j -> ...i', rot, jnp.stack(( length_half, -width_half), axis=-1))
    br = pt + jnp.einsum('...ij, ...j -> ...i', rot, jnp.stack((-length_half, -width_half), axis=-1))
    bl = pt + jnp.einsum('...ij, ...j -> ...i', rot, jnp.stack((-length_half,  width_half), axis=-1))
    
    return jnp.stack((tl, tr, br, bl), axis=-2)


def calculate_distance_bounding_boxes(state_1, state_2):
    pass

_FUNC_DICT = {
    'center': calculate_state_distance_center_points,
}

def _get_distance_fn(option):
    distance_fn = _FUNC_DICT.get(option, None)
    if distance_fn == None:
        raise NotImplementedError(f'Distance function \'{option}\' is not implemented.')
    return distance_fn

def calculate_state_distance(state_1, state_2, valid_1, valid_2, distance_fn):
    return jnp.where(valid_1 & valid_2, distance_fn(state_1, state_2), jnp.nan)

def _reduction(res, reduction='none'):
    if reduction == 'mean':
        return jnp.nanmean(res)
    elif reduction == 'sum':
        return jnp.nansum(res)
    elif reduction == 'min':
        return jnp.nanmin(res)
    else:
        return res

def calculate_traj_distance(traj_1, traj_2, valid_traj_1, valid_traj_2, distance_fn, reduction):
    '''
    The function is equivalent to:

    dist = []
    for t in range(T):  -> jax.vmap(unbatched_fn, [0, 0, 0, 0])
        dist.append(calculate_state_distance(traj_1[t], traj_2[t], valid_traj_1[t], valid_traj_2[t])) -> unbatched_fn
    return _reduction(jnp.stack(dist, axis=0), reduction)
    '''
    unbatched_fn = partial(calculate_state_distance, distance_fn=distance_fn)
    batched_fn = jax.vmap(unbatched_fn, [0, 0, 0, 0])
    res = batched_fn(traj_1, traj_2, valid_traj_1, valid_traj_2)
    return _reduction(res, reduction=reduction)

@jax.jit
def calculate_distance_ego_col(ego_traj, adv_trajs, valid_ego_traj, valid_adv_trajs, option='center'):
    unbatched_fn = partial(calculate_traj_distance, distance_fn=_get_distance_fn(option), reduction='mean')
    batched_fn = jax.vmap(unbatched_fn, [None, 0, None, 0])
    res = batched_fn(ego_traj, adv_trajs, valid_ego_traj, valid_adv_trajs)
    return _reduction(res, reduction='min')

@jax.jit
def calculate_distance_adv_col(adv_trajs, valid_adv_trajs, threshold, option='center'):
    unbatched_fn = partial(calculate_state_distance, distance_fn=_get_distance_fn(option))
    batched_fn = jax.vmap(unbatched_fn, [0, 0, 0, 0], 0)       # Loop for time
    batched_fn = jax.vmap(batched_fn, [None, 0, None, 0])      # Loop for state_2
    batched_fn = jax.vmap(batched_fn, [0, None, 0, None])      # Loop for state_1
    res = batched_fn(adv_trajs, adv_trajs, valid_adv_trajs, valid_adv_trajs)
    for i in range(adv_trajs.shape[0]):
        res = res.at[i, i:].set(jnp.nan)
    return jnp.clip(_reduction(res, reduction='min'), max=threshold ** 2)


def calculate_potential_adv_dev(adv_trajs, valid_adv_trajs, drivable_area_map, cropsize=32):
    binary_map, (x_range, y_range, coord_xys) = drivable_area_map

    def calculate_potential(xy, sigma=0.5):
        x_center = jnp.abs(x_range - xy[0]).argmin()
        y_center = jnp.abs(y_range - xy[1]).argmin()

        cropped_binary_map = jax.lax.dynamic_slice(binary_map, (jnp.minimum(x_center-cropsize//2, 0), jnp.minimum(y_center-cropsize//2, 0)), (cropsize, cropsize))
        cropped_coord_xys = jax.lax.dynamic_slice(coord_xys, (jnp.minimum(x_center-cropsize//2, 0), jnp.minimum(y_center-cropsize//2, 0), 0), (cropsize, cropsize, 2))
        
        gs = 1 / (sigma * jnp.sqrt(2 * jnp.pi)) * jnp.exp(-0.5 * jnp.sum((cropped_coord_xys - xy) ** 2, axis=-1) / sigma ** 2)
        return jnp.sum(gs * cropped_binary_map)
        
    adv_trajs_corners = calculate_box_corner_corrdinates(adv_trajs)

    compute_fn = calculate_potential
    for _ in range(len(adv_trajs_corners.shape[:-1])):
        compute_fn = jax.vmap(compute_fn)

    res = compute_fn(adv_trajs_corners)
    res = res[valid_adv_trajs]
    return _reduction(res, reduction='sum')
