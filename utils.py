from typing import List
from jax import numpy as jnp
import jax
import optax

from waymax import datatypes

def mask_invalid_traj(log_xy):
    mask_a = (log_xy != -1.).astype(jnp.float32)
    mask_b = (log_xy[:, 0, 0, None, None] != -1.).astype(jnp.float32)

    return mask_a * mask_b

def flatten_actions(actions):
    actions_data = [action.data for action in actions]
    actions_valid = [action.valid for action in actions]
    return actions_data, actions_valid

def unflatten_actions(actions_data, actions_valid):
    return [datatypes.Action(data=data, valid=valid) for data, valid in zip(actions_data, actions_valid)]

def get_traj_from_state_list(states, idx='all', fields=['x', 'y'], keepdim=True):
    assert (isinstance(idx, jax.Array) and jax.numpy.isdtype(idx.dtype, 'integral')) \
        or isinstance(idx, int) \
        or (isinstance(idx, List) and all(isinstance(e, int) for e in idx)) \
        or idx == 'all', \
            'idx should be an integer, a list of integers, an Array of integers, or \'all\'.'
    
    if isinstance(idx, int) and keepdim:
        idx = [idx]

    idx = jnp.array(idx)

    if idx == 'all':
        return jnp.concat([state.current_sim_trajectory.stack_fields(fields) for state in states], axis=-2), \
                states[0].log_trajectory.valid
    elif isinstance(idx, jax.Array):
        return jnp.concat([state.current_sim_trajectory.stack_fields(fields)[idx] for state in states], axis=-2), \
                states[0].log_trajectory.valid[idx]

def zero_grads():
    # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
    def init_fn(_): 
        return ()
    def update_fn(updates, state, params=None):
        return jax.tree_map(jnp.zeros_like, updates), ()
    return optax.GradientTransformation(init_fn, update_fn)

def debug_info(logger, fmt, *args, **kwargs):
    jax.debug.callback(
        lambda *args, **kwargs: logger.info(fmt.format(*args, **kwargs)),
        *args, **kwargs)