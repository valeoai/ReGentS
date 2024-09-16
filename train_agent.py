import json
import os
from tqdm import tqdm
from pathlib import Path
import random
import logging

from typing import List

import numpy as np
import torch
import jax 
import jax.numpy as jnp
import hydra
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
torch.multiprocessing.set_sharing_strategy('file_system')
import time
import pickle

from waymax import config as _config
from waymax import dataloader
from waymax import dynamics, agents, datatypes, visualization
from waymax import env as _env
from waymax.utils import geometry
from waymax.metrics.roadgraph import is_offroad
import dataclasses

from agent.data import WaymaxRasterDatasetFromBuffer
from agent.model import AimBev
from agent.aim_bev_actor import AimBEVActor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

_NON_SDC_CONTROL_FUNC = lambda state: ~state.object_metadata.is_sdc


def draw_centered_rectangle(image, rect_width, rect_height, rect_color):
    # Calculate the position to center the rectangle
    center_x = 96
    center_y = 96
    top_left_x = center_x - rect_width // 2
    top_left_y = center_y - rect_height // 2

    # Draw the rectangle
    image[:, top_left_y:top_left_y + rect_height, top_left_x:top_left_x + rect_width] = rect_color

    return image

def unpickle_files_in_directory(directory_path):
    # List all files in the directory
    files = [f for f in sorted(os.listdir(directory_path)) if os.path.isfile(os.path.join(directory_path, f))]
    
    # Define a generator function to unpickle files one by one
    def file_iterator():
        for file in files:
            with open(os.path.join(directory_path, file), 'rb') as f:
                yield pickle.load(f)
    
    # Create and return the iterator
    return iter(file_iterator())

class Trainer:
    def __init__(self, model, optimizer, dataloader_train, cfg):
        self.model = model
        self.pred_len = model.pred_len
        self.optimizer = optimizer
        self.dataloader_train = dataloader_train
        self.cur_epoch = 0
        self.cur_iter = 0
        self.bestval_epoch = 0
        self.train_loss = []
        self.val_loss = []
        self.bestval = -1e5
        self.cfg = cfg
        self.log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
        
    def train(self):
        self.model.train()
        for epoch in range(self.cfg.trainer.max_epochs):
            self.train_epoch()
            logger.info(f'Training loss at {self.cur_epoch}: {self.train_loss[-1]}')
            self.save_recent()
            if epoch % self.cfg.trainer.eval_every == 0 and not epoch == 0:
                self.run_eval(self.cfg.dataset)
                logger.info(f'Training loss at {self.cur_epoch}: {self.train_loss[-1]}')
                self.save_best()

    def train_epoch(self):
        wp_epoch = 0.
        num_batches = 0

        for data in tqdm(self.dataloader_train):
            gt_waypoints = data['waypoints'][:, 1:self.pred_len+1].to(device)
            bev = data['bev'].to(torch.float).to(device) / 255.0 * 2. - 1.

            target_point = data['target_point'].to(device)
            pred_waypoints = self.model(bev, target_point)

            loss = F.l1_loss(pred_waypoints, gt_waypoints)
            wp_epoch += loss.item()
            
            loss.backward()

            num_batches += 1
            self.optimizer.step()
            self.optimizer.zero_grad(True)

            self.cur_iter += 1

        loss_epoch = wp_epoch / num_batches

        self.train_loss.append(loss_epoch)
        self.cur_epoch += 1

    @torch.no_grad()
    def run_eval(self, cfg):
        eval = Evaluator(self.model, cfg)

        if 'val' in cfg:
            val_dataset_config = _config.DatasetConfig(
                path=cfg.val.path,
                repeat=1,
                max_num_rg_points=cfg.max_num_rg_points,
                data_format=_config.DataFormat.TFRECORD,
                max_num_objects=cfg.max_num_objects,
            )

            val_orig_data_iter = dataloader.simulator_state_generator(config=val_dataset_config)
        if 'val_king' in cfg:
            val_data_iter = unpickle_files_in_directory(cfg.val_king.path)

        if 'val_king' in cfg:
            metrics = eval.evaluate_generated(val_data_iter, val_orig_data_iter)
        else:
            metrics = eval.evaluate(val_orig_data_iter)
        self.val_loss.append(metrics['cr'])
        return metrics

    def save_best(self):
        save_best = False
        if self.val_loss[-1] < self.bestval:
            self.bestval = self.val_loss[-1]
            self.bestval_epoch = self.cur_epoch
            save_best = True

        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        if save_best:
            torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'best_model.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'best_optim.pth'))
            print('====== Overwrote best model ======>')
            with open(os.path.join(self.log_dir, 'best.log'), 'w') as f:
                f.write(json.dumps(log_table))

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, f'model_epoch{self.cur_epoch}.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, f'optim_epoch{self.cur_epoch}.pth'))
        print('====== Save model by epoch ======>')
        with open(os.path.join(self.log_dir, f'epoch{self.cur_epoch}.log'), 'w') as f:
            f.write(json.dumps(log_table))

    def save_recent(self):
        # Create a dictionary of all data to save
        log_table = {
            'epoch': self.cur_epoch,
            'iter': self.cur_iter,
            'bestval': self.bestval,
            'bestval_epoch': self.bestval_epoch,
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
        }

        torch.save(self.model.state_dict(), os.path.join(self.log_dir, 'model.pth'))
        torch.save(self.optimizer.state_dict(), os.path.join(self.log_dir, 'recent_optim.pth'))

        # Log other data corresponding to the recent model
        with open(os.path.join(self.log_dir, 'recent.log'), 'w') as f:
            f.write(json.dumps(log_table))

        print('====== Saved recent model ======>')

class Evaluator:
    def __init__(self, model, cfg):
        self.cfg = cfg
        self.model = model

    def evaluate(self, val_data_iter):
        self.model.eval()

        dynamics_model = dynamics.InvertibleBicycleModel()

        # Expect users to control all valid object in the scene.
        env = _env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                _config.EnvironmentConfig(),
                init_steps=self.cfg.init_steps,
                max_num_objects=self.cfg.max_num_objects,
                controlled_object=_config.ObjectType.VALID,
            ),
        )

        # Build Agent
        ego_actor = AimBEVActor(self.model)

        adv_actor = agents.create_expert_actor(
            dynamics_model=dynamics_model,
            is_controlled_func=_NON_SDC_CONTROL_FUNC,
        )

        select_action_fns = [ego_actor.select_action, jax.jit(adv_actor.select_action)]

        rc_list, is_list, ds_list, cr_list, im_list = (list() for _ in range(5))

        for s_idx, scenario in enumerate(val_data_iter):
            if 'val' in self.cfg and s_idx >= self.cfg.val.max_num_scenario:
                break
            t_start = time.time()
            id_sdc = jnp.where(scenario.object_metadata.is_sdc)[0][0]
            
            states = [env.reset(scenario)]
            for t in range(states[0].remaining_timesteps):
                curr_state = states[-1]
                outputs = [
                    select_action_fn({'id_sdc': id_sdc}, curr_state, None, None) for select_action_fn in select_action_fns
                ]
                action = agents.merge_actions(outputs)
                next_state = env.step(curr_state, action)
                states.append(next_state)

            # imgs = list()
            # imgs_org = list()
            # for state in states:
            #     imgs.append(visualization.plot_simulator_state(state, use_log_traj=False))
            #     imgs_org.append(visualization.plot_simulator_state(state, use_log_traj=True))

            # mediapy.write_video(f'vid_{s_idx}.gif', imgs, fps=10, codec='gif')
            # mediapy.write_video(f'vid_{s_idx}_org.gif', imgs_org, fps=10, codec='gif')
            _is, is_coll, coll_t = self.infraction_score(id_sdc, states)
            _rc = float(self.route_completion_state(id_sdc, states[coll_t]).squeeze())
            _is = float(_is.squeeze())
            _ds = _rc * _is
            _cr = 1.0 if is_coll else 0.0
            log_traj_xy = states[0].log_trajectory.xy
            sim_traj_xy = jnp.concatenate([state.current_sim_trajectory.xy for state in states], axis=1)
            _im = float(jnp.abs(sim_traj_xy[id_sdc] - log_traj_xy[id_sdc]).mean())
            print(time.time() - t_start)

            logger.info(f'{_rc=:.4}, {_is=:.4}, {_ds=:.4}, {_cr=}, {_im=}')

            rc_list.append(_rc)
            is_list.append(_is)
            ds_list.append(_ds)
            cr_list.append(_cr)
            im_list.append(_im)

        return {
            'rc': np.array(rc_list).mean(),
            'is': np.array(is_list).mean(),
            'ds': np.array(ds_list).mean(),
            'cr': np.array(cr_list).mean(),
            'im': np.array(im_list).mean(),
        }
    
    def evaluate_generated(self, val_data_iter, val_data_orig_iter):
        self.model.eval()

        dynamics_model = dynamics.InvertibleBicycleModel()

        # Expect users to control all valid object in the scene.
        env = _env.MultiAgentEnvironment(
            dynamics_model=dynamics_model,
            config=dataclasses.replace(
                _config.EnvironmentConfig(),
                init_steps=self.cfg.init_steps,
                max_num_objects=self.cfg.max_num_objects,
                controlled_object=_config.ObjectType.VALID,
            ),
        )

        rc_list, is_list, ds_list, cr_list, im_list = (list() for _ in range(5))

        for s_idx, (scenario, orig_scenario) in enumerate(zip(val_data_iter, val_data_orig_iter)):
            if s_idx >= self.cfg.val.max_num_scenario:
                break
            t_start = time.time()
            id_sdc = jnp.where(scenario.object_metadata.is_sdc)[0][0]
            
            states = [env.reset(scenario)]
            for t in range(states[0].remaining_timesteps):
                curr_state = states[-1]
                next_state = datatypes.update_state_by_log(curr_state, num_steps=1)
                states.append(next_state)

            _is, is_coll, coll_t = self.infraction_score(id_sdc, states)
            _rc = float(self.route_completion(id_sdc, states[coll_t], orig_scenario).squeeze())
            _is = float(_is.squeeze())
            _ds = _rc * _is
            _cr = 1.0 if is_coll else 0.0
            log_traj_xy = orig_scenario.log_trajectory.xy
            sim_traj_xy = jnp.concatenate([state.current_sim_trajectory.xy for state in states], axis=1)
            _im = float(jnp.abs(sim_traj_xy[id_sdc] - log_traj_xy[id_sdc]).mean())
            print(time.time() - t_start)

            logger.info(f'{_rc=:.4}, {_is=:.4}, {_ds=:.4}, {_cr=}, {_im=}')

            rc_list.append(_rc)
            is_list.append(_is)
            ds_list.append(_ds)
            cr_list.append(_cr)
            im_list.append(_im)

        return {
            'rc': np.array(rc_list).mean(),
            'is': np.array(is_list).mean(),
            'ds': np.array(ds_list).mean(),
            'cr': np.array(cr_list).mean(),
            'im': np.array(im_list).mean(),
        }

    def route_completion(self, id_sdc, state, orig_state=None):

        obj_xy_curr = datatypes.dynamic_slice(
            state.sim_trajectory.xy,
            state.timestep,
            1,
            axis=-2,
        )
        sdc_xy_curr = obj_xy_curr[..., id_sdc, 0, :]

        obj_path = state.log_trajectory if orig_state is None else orig_state.log_trajectory
        sdc_path = datatypes.dynamic_index(obj_path, id_sdc, axis=0)

        ego_xy = sdc_path.xy
        arc_length_dt = jnp.linalg.norm(ego_xy[..., 1:, :] - ego_xy[..., :-1, :], axis=-1, keepdims=False)
        arc_length = jnp.cumsum(arc_length_dt, axis=-1)
        arc_length = jnp.concat([arc_length[..., 0, jnp.newaxis] * 0.0, arc_length], axis=-1)

        dist_raw = jnp.linalg.norm(
            sdc_xy_curr[..., jnp.newaxis, :] - sdc_path.xy, axis=-1, keepdims=False
        )
        dist = jnp.where(sdc_path.valid, dist_raw, jnp.inf)

        idx = jnp.argmin(dist, axis=-1, keepdims=True)

        curr_arc_length = jnp.take_along_axis(arc_length, indices=idx, axis=-1)[..., 0]
        total_length = arc_length[..., -1]
        if total_length < 1:
            return jnp.array(1.0)

        return curr_arc_length / total_length

    def offroad(self, id_sdc, state):
        sdc_traj = datatypes.dynamic_index(state.sim_trajectory, id_sdc, axis=0)
        offroad = 0.
        for t in range(sdc_traj.shape[-1]):
            sdc_traj_t = datatypes.dynamic_index(sdc_traj, t, axis=1)
            offroad_t = is_offroad(sdc_traj_t, state.roadgraph_points)
            offroad = offroad + offroad_t.astype(jnp.float32).squeeze()
        return offroad / sdc_traj.shape[-1]
    
    def infraction_score(self, id_sdc, states: List[datatypes.SimulatorState]):
        is_coll_with_vehicle = False
        is_coll_with_pedestrian = False
        is_coll_with_cyclist = False
        is_coll = False
        for t, state in enumerate(states):
            current_traj = state.current_sim_trajectory
            traj_5dof = current_traj.stack_fields(['x', 'y', 'length', 'width', 'yaw'])
            # Shape: (..., num_objects, num_objects)
            pairwise_overlap = geometry.compute_pairwise_overlaps(traj_5dof[..., 0, :])

            is_coll = is_coll | jnp.any(pairwise_overlap[..., id_sdc])

            is_vehicle      = state.object_metadata.object_types == 1
            is_pedestrian   = state.object_metadata.object_types == 2
            is_cyclist      = state.object_metadata.object_types == 3

            is_coll_with_vehicle = is_coll_with_vehicle | jnp.any(pairwise_overlap[..., id_sdc, is_vehicle])
            is_coll_with_pedestrian = is_coll_with_pedestrian | jnp.any(pairwise_overlap[..., id_sdc, is_pedestrian])
            is_coll_with_cyclist = is_coll_with_cyclist | jnp.any(pairwise_overlap[..., id_sdc, is_cyclist])

            if is_coll:
                break

        offroad = self.offroad(id_sdc, states[-1])
        return ((1 - 0.5 * is_coll_with_pedestrian) * \
            (1 - 0.4 * is_coll_with_vehicle) * \
            (1 - 0.4 * is_coll_with_cyclist) * \
            (1 - offroad), is_coll, t)



@hydra.main(version_base=None, config_path="conf", config_name="config_aim_bev")
def main(cfg):
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = AimBev(pred_len=8).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

    if 'checkpoint' in cfg:
        checkpoint_model = torch.load(cfg.checkpoint.model_path, map_location=device)
        checkpoint_optimizer = torch.load(cfg.checkpoint.optimizer_path, map_location=device)
        model.load_state_dict(checkpoint_model)
        optimizer.load_state_dict(checkpoint_optimizer)

    dataset_train = WaymaxRasterDatasetFromBuffer(cfg.dataset.train.buffer)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.dataloader.batch_size, shuffle=True, drop_last=True)

    trainer = Trainer(model, optimizer, dataloader_train, cfg)
    if 'checkpoint' in cfg and cfg.checkpoint.eval_only:
        trainer.run_eval(cfg.dataset)
    else:
        trainer.train()


if __name__ == "__main__":
    main()