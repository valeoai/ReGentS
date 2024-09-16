import torch
from torch import nn
from torchvision import models
import jax.numpy as jnp
import jax

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AimBev(nn.Module):
    def __init__(self, pred_len, batch_size=1, fps=10, lat_pid_params=None, lon_pid_params=None, **kwargs):
        super(AimBev, self).__init__()
        self.pred_len = pred_len
        self.input_size = 3

        if not lat_pid_params:
            lat_pid_params = {"K_P": 1.0, "K_I": 0.1,
                              "K_D": 0.1, "window_size": 8}
        if not lon_pid_params:
            lon_pid_params = {"K_P": 1.0, "K_I": 0.1,    
                              "K_D": 0.1, "window_size": 8}
        self.turn_controller = PIDController(
            **lat_pid_params, batch_size=batch_size)
        self.speed_controller = PIDController(
            **lon_pid_params, batch_size=batch_size)
        self.stop_slope = 10.
        self.decel_slope = 1.
        self.emergency_thresh = 0.6
        self.delta_slope = 0.5
        self.dt = 1. / fps

        self.image_encoder = ImageCNN(512, self.input_size, use_linear=False)
        self.join = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
        )

        self.decoder = nn.GRUCell(input_size=6, hidden_size=64)

        self.output = nn.Linear(64, 2)

    def reset_controllers(self):
        self.turn_controller.reset()
        self.speed_controller.reset()

    def forward(self, bev: torch.Tensor, target_point):
        target_point = target_point
        bev = bev.permute(0, 3, 1, 2)
        feat = self.image_encoder(bev)
        z = self.join(feat)

        output_wp = list()

        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype, device=z.device)
        
        for _ in range(self.pred_len):
            x_in = torch.concat((x, target_point, target_point - x), dim=-1)
            z = self.decoder(x_in, z)
            dx = self.output(z)
            x = dx + x
            output_wp.append(x)

        pred_wp = torch.stack(output_wp, dim=1)

        return pred_wp

    def control_pid(self, waypoints, speed):
        desired_speed = jnp.linalg.norm(
            waypoints[:, 1, None] - waypoints[:, 0, None], axis=-1) / (self.dt)

        stop_probability = jax.nn.sigmoid(-self.stop_slope *
                                         (desired_speed - self.emergency_thresh))
        speed_violation = jax.nn.relu(speed - desired_speed)
        decel_probability = jax.nn.tanh(self.decel_slope * (speed_violation))

        if desired_speed < self.emergency_thresh:
            brake_probability = stop_probability
        else:
            brake_probability = decel_probability
        brake_flag = jax.lax.gt(
            brake_probability, 0.1)

        delta = jax.nn.relu(desired_speed - speed)
        throttle = self.speed_controller.step(delta)
        throttle = jax.nn.relu(throttle)
        throttle = throttle * ~brake_flag

        aim = (waypoints[:, 1] + waypoints[:, 0]) / 2.0

        angle = jnp.atan2(aim[:, 1], aim[:, 0])

        angle = jnp.expand_dims(angle, axis=-1) * ~brake_flag

        steer = self.turn_controller.step(angle)

        return steer, throttle, brake_probability


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, fps=10, window_size=20, batch_size=1):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D
        self.batch_size = batch_size
        self.window_size = window_size
        self._dt = 1.0 / fps

        self._window = jnp.zeros((batch_size, window_size))

    def step(self, error):
        self._window = jnp.concat((self._window[:, 1:], error * self._dt), axis=1)

        if self._window.shape[1] >= 2:
            integral = jnp.sum(self._window, axis=1, keepdims=True)
            derivative = (self._window[:, -1, None] -
                          self._window[:, -2, None]) / (self._dt ** 2)
        else:
            integral = jnp.zeros((self.batch_size, 1))
            derivative = jnp.zeros((self.batch_size, 1))

        return self._K_P * error + self._K_I * integral + self._K_D * derivative

    def reset(self):
        self._window = jnp.zeros(
            (self.batch_size, self.window_size))


class ImageCNN(nn.Module):
    """ Encoder network for image input list.
    Args:
        c_dim (int): output dimension of the latent embedding
        normalize (bool): whether the input images should be normalized
        use_linear (bool): whether a final linear layer should be used
    """

    def __init__(self, c_dim, input_dim=3, use_linear=False, **kwargs):
        super().__init__()
        self.use_linear = use_linear

        self.features = models.mobilenet_v3_small(weights='IMAGENET1K_V1')
        self.features.classifier._modules['3'] = nn.Linear(1024, 512)
        self.features.features._modules['0']._modules['0'] = nn.Conv2d(
            input_dim, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)

        if use_linear:
            self.fc = nn.Linear(512, c_dim)
        elif c_dim == 512:
            self.fc = nn.Sequential()
        else:
            raise ValueError('c_dim must be 512 if use_linear is False')

    def forward(self, inputs):
        net = self.features(inputs)
        return self.fc(net)

    def load_state_dict(self, state_dict):
        errors = super().load_state_dict(state_dict, strict=False)
        print(errors)
