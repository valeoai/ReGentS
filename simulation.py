from waymax import agents
import torch
import jax.numpy as jnp
from agent import AimBEVActor, ActionActor

_NON_SDC_CONTROL_FUNC = lambda state: ~state.object_metadata.is_sdc

def simulate_scenario(env, scenario, dynamics_model):
    obj_idx = jnp.arange(scenario.num_objects)
    logged_actor = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=lambda state: (obj_idx >= 0),
    )
    actors = [logged_actor]
    states, actions = [env.reset(scenario)], []     # Create lists storing states and actions; store the initial states
    timesteps = states[0].remaining_timesteps       # Number of remaining states
    for _ in range(timesteps):
        curr_state = states[-1]                     # s[t-1]
        outputs = [actor.select_action({}, curr_state, None, None) for actor in actors]
        action = agents.merge_actions(outputs)
        next_state = env.step(curr_state, action)
        states.append(next_state)
        actions.append(action)
    return states, actions

def simulate_scenario_with_actions(env, scenario, actions):
    states = [env.reset(scenario)] 
    timesteps = states[0].remaining_timesteps
    for t in range(timesteps):
        curr_state = states[-1]
        next_state = env.step(curr_state, actions[t])
        states.append(next_state)
    return states

@torch.no_grad()
def simulate_scenario_aim_bev(env, torch_model, scenario, dynamics_model):
    torch_model.reset_controllers()
    ego_actor = AimBEVActor(torch_model)
    adv_actor = agents.create_expert_actor(
        dynamics_model=dynamics_model,
        is_controlled_func=_NON_SDC_CONTROL_FUNC,
    )
    id_sdc = jnp.where(scenario.object_metadata.is_sdc)[0][0]

    select_action_fns = [ego_actor.select_action, adv_actor.select_action]
    states, actions = [env.reset(scenario)], []     # Create lists storing states and actions; store the initial states
    timesteps = states[0].remaining_timesteps       # Number of remaining states
    for _ in range(timesteps):
        curr_state = states[-1]                     # s[t-1]
        outputs = [select_action_fn({'id_sdc': id_sdc}, curr_state, None, None) for select_action_fn in select_action_fns]
        action = agents.merge_actions(outputs)
        next_state = env.step(curr_state, action)
        states.append(next_state)
        actions.append(action)
    return states, actions

@torch.no_grad()
def simulate_scenario_with_aim_bev_and_actions(env, torch_model, scenario, actions):
    torch_model.reset_controllers()
    ego_actor = AimBEVActor(torch_model)
    adv_actor = ActionActor(actions)
    select_action_fns = [ego_actor.select_action, adv_actor.select_action]
    states = [env.reset(scenario)] 
    timesteps = states[0].remaining_timesteps
    id_sdc = jnp.where(scenario.object_metadata.is_sdc)[0][0]
    for t in range(timesteps):
        curr_state = states[-1]
        outputs = [select_action_fn({'id_sdc': id_sdc}, curr_state, None, None) for select_action_fn in select_action_fns]
        action = agents.merge_actions(outputs)
        next_state = env.step(curr_state, action)
        states.append(next_state)
    return states