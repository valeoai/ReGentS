from waymax import visualization
import mediapy

viz_config = {
    'front_x': 50.0,
    'front_y': 50.0,
    'back_x': 50.0,
    'back_y': 50.0,
}

def generate_video(states, path):
    imgs = list()
    for state in states:
        imgs.append(visualization.plot_simulator_state(state, use_log_traj=False, viz_config=viz_config))
    mediapy.write_video(path, imgs, fps=10, codec='gif')