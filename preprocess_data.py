import logging
import shelve

import numpy as np
import torch
import hydra
import jax
torch.multiprocessing.set_sharing_strategy('file_system')

from waymax import dataloader
from waymax import config as _config

from waymax import datatypes
from waymax.datatypes.observation import sdc_observation_from_state

from agent.rasterization_jnp import rasterize_observation_jnp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@jax.jit
def rasterization(scenario):
    init_obs = sdc_observation_from_state(scenario)
    pose2d = datatypes.dynamic_index(init_obs.pose2d, 0, axis=0, keepdims=False)
    traj = scenario.log_trajectory
    traj = datatypes.transform_trajectory(traj, pose2d)

    obs = sdc_observation_from_state(scenario)
    bev = rasterize_observation_jnp(obs)
    return bev, traj

@hydra.main(version_base=None, config_path="conf", config_name="config_aim_bev")
def main(cfg):
    dataset_config = _config.DatasetConfig(
        path=cfg.dataset.train.path,
        repeat=1,
        max_num_rg_points=cfg.dataset.max_num_rg_points,
        data_format=_config.DataFormat.TFRECORD,
        max_num_objects=cfg.dataset.max_num_objects,
    )

    data_iter = dataloader.simulator_state_generator(config=dataset_config)

    buffer = shelve.open(dataset_config.train.buffer)

    for idx, scenario in enumerate(data_iter):
        if idx >= 100_000:
            break
         
        if buffer.get(f'{idx}', None) != None:
            print(f'{idx} exist')
            continue

        print(f'Generate {idx}')

        scenario = jax.lax.stop_gradient(scenario)
        bev, traj = rasterization(scenario)

        id_sdc = np.where(scenario.object_metadata.is_sdc)[0][0]
        waypoints = traj.xy[id_sdc]

        data = dict()
        data['bev'] = np.array(bev)
        data['waypoints'] = np.array(waypoints)
        data['target_point'] = np.array(waypoints[-1])

        buffer[f'{idx}'] = data


if __name__ == "__main__":
    main()