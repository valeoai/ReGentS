import torch

from waymax.agents.actor_core import WaymaxActorCore, WaymaxActorOutput

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_NON_SDC_CONTROL_FUNC = lambda state: ~state.object_metadata.is_sdc

class ActionActor(WaymaxActorCore):
    def __init__(self, actions):
        super().__init__()
        self.actions = actions

    def init(self, rng, state):
        return

    def name(self):
        return 'ActionActor'

    def select_action(self, params, state, actor_state, rng):
        del params, rng, actor_state
        curr_timestep = state.timestep
        action = self.actions[curr_timestep]

        return WaymaxActorOutput(
            actor_state=None,
            action=action,
            is_controlled=_NON_SDC_CONTROL_FUNC(state)
        )