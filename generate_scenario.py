import dataclasses
import optax
import logging
from pathlib import Path
import pickle
import copy
import torch
import os

from waymax import config as _config
from waymax import dataloader
from waymax import dynamics
from waymax import env as _env

import hydra
from omegaconf import DictConfig

from simulation import simulate_scenario_aim_bev, simulate_scenario_with_aim_bev_and_actions
from visualization import generate_video
import method
from agent.model import AimBev

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false' 
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

@torch.no_grad()
@hydra.main(version_base=None, config_path="conf", config_name="config_scenario_opt")
def main(cfg: DictConfig):
    output_path = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    
    dataset_config = _config.DatasetConfig(
        path=cfg.dataset.path,
        max_num_rg_points=cfg.dataset.max_num_rg_points,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=cfg.dataset.max_num_objects,
    )

    data_iter = dataloader.simulator_state_generator(config=dataset_config)

    dynamics_model = dynamics.InvertibleBicycleModel()

    env = _env.MultiAgentEnvironment(
        dynamics_model=dynamics_model,
        config=dataclasses.replace(
            _config.EnvironmentConfig(),
            init_steps=cfg.env.init_steps,
            max_num_objects=cfg.dataset.max_num_objects,
            controlled_object=_config.ObjectType.VALID,
        ),
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch_model = AimBev(pred_len=8).to(device)
    checkpoint = torch.load(cfg.planner.checkpoint, map_location=device)

    torch_model.load_state_dict(checkpoint)

    scenario_path = output_path / 'scenario'
    video_path = output_path / 'video'

    scenario_path.mkdir(parents=True, exist_ok=True)
    video_path.mkdir(parents=True, exist_ok=True)

    for i, scenario in enumerate(data_iter):
        if i < cfg.method.scenario_id.min:
            continue
        if i >= cfg.method.scenario_id.max:
            return
        
        states_orig, actions = simulate_scenario_aim_bev(env, torch_model, scenario, dynamics_model)
        logger.info(f'Actions loaded for scenario {i}.')

        solver = optax.adam(learning_rate=cfg.optimizer.main.lr)

        actions = method.opt(actions, env, torch_model, scenario, solver, cfg.method)
        states = simulate_scenario_with_aim_bev_and_actions(env, torch_model, scenario, actions)

        generate_video(states, video_path / f'new_scenario_{i:05d}.gif')

        new_traj = states[-1].sim_trajectory
        new_scenario = copy.deepcopy(states[0])
        new_scenario.log_trajectory = new_traj

        orig_traj = states_orig[-1].sim_trajectory
        orig_scenario = copy.deepcopy(states_orig[0])
        orig_scenario.log_trajectory = orig_traj

        with open(scenario_path / f'new_scenario_{i:05d}.pkl', "wb") as f:
            pickle.dump(new_scenario, f)

        with open(scenario_path / f'orig_scenario_{i:05d}.pkl', "wb") as f:
            pickle.dump(orig_scenario, f)


if __name__ == '__main__':
    main()