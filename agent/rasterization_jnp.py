import numpy as np
import jax.numpy as jnp

from waymax import config as waymax_config
from waymax import datatypes
from waymax.visualization import color
from waymax.visualization.viz import _index_pytree, plot_traffic_light_signals_as_points, plot_roadgraph_points
from waymax.visualization import utils as viz_utils

import jax

_RoadGraphShown = (1, 2, 3, 15, 16, 17, 18, 19)
_RoadGraphDefaultColor = (230, 230, 230)

TRAFFIC_LIGHT_COLORS = {
    # Unknown = 0, Arrow_Stop = 1, Arrow_Caution = 2, Arrow_Go = 3, Stop = 4,
    # Caution = 5, Go = 6, Flashing_Stop = 7, Flashing_Caution = 8
    # third_party/waymo_open_dataset/protos/map.proto
    0: jnp.array([0.75, 0.75, 0.75]),
    1: jnp.array([1.0, 0.0, 0.0]),
    2: jnp.array([1.0, 1.0, 0.0]),
    3: jnp.array([0.0, 1.0, 0.0]),
    4: jnp.array([1.0, 0.0, 0.0]),
    5: jnp.array([1.0, 1.0, 0.0]),
    6: jnp.array([0.0, 1.0, 0.0]),
    7: jnp.array([1.0, 1.0, 0.0]),
    8: jnp.array([1.0, 1.0, 0.0]),
}

def apply_color_mask(image, mask, color):
    """
    Apply a color to the areas of the image indicated by the mask.

    Parameters:
    - image: Input image as a NumPy array of shape (height, width, 3).
    - mask: Boolean mask as a NumPy array of shape (height, width), where True indicates the areas to be colored.
    - color: Tuple (R, G, B) representing the color to apply.

    Returns:
    - Colored image as a NumPy array.
    """
    # Create a color mask
    color_mask = jnp.zeros_like(image)
    color_mask = jnp.where(mask[..., jnp.newaxis], jnp.array(color), color_mask)
    colored_image = jnp.where(mask[..., jnp.newaxis], color_mask, image)

    return colored_image

def distance_to_points(image_shape, points):
    """
    Compute the distance from each pixel in the image to the nearest point in the set.

    Parameters:
    - image_shape: Tuple (height, width) specifying the size of the image.
    - points: Array of shape (N, 2), where N is the number of points, and each point is represented by (x, y) coordinates.

    Returns:
    - distances: The distance for each pixel in the image to the nearest point.
    """
    height, width = image_shape
    y_indices, x_indices = jnp.indices((height, width))

    x_indices = x_indices[..., jnp.newaxis]
    y_indices = y_indices[..., jnp.newaxis]

    # Extract point coordinates
    px = points[:, 0]
    py = points[:, 1]

    # Compute the squared distances for efficiency
    distances_squared = (x_indices - px) ** 2 + (y_indices - py) ** 2
    
    # Compute the minimum distances
    min_distances_squared = jnp.min(distances_squared, axis=-1)
    
    # Take the square root to get the Euclidean distance
    min_distances = jnp.sqrt(min_distances_squared)
    
    return min_distances

def draw_lines_jax(image, start, end, color):
    """Draw multiple line segments on the image by setting pixels on the segments to 255 using vectorized operations."""
    height, width = image.shape[:-1]

    # Generate a grid of coordinates
    y_indices, x_indices = jnp.indices((height, width))

    x_indices = x_indices[..., jnp.newaxis]
    y_indices = y_indices[..., jnp.newaxis]

    # Separate the lines into starting and ending points
    x0, y0, x1, y1 = start[0, :], start[1, :], end[0, :], end[1, :]

    # Calculate differences and distances
    dx = x1 - x0
    dy = y1 - y0

    px = x_indices - x0
    py = y_indices - y0

    segment_length_squared = dx ** 2 + dy ** 2
    t = (px * dx + py * dy) / segment_length_squared
    t_clamped = jnp.clip(t, 0, 1)

    nearest_x = x0 + t_clamped * dx
    nearest_y = y0 + t_clamped * dy

    distance = jnp.sqrt((x_indices - nearest_x) ** 2 + (y_indices - nearest_y) ** 2)
    on_line = distance < 0.5
    on_line = jnp.any(on_line, axis=-1)

    return apply_color_mask(image, on_line, color=color)


def signed_distance_to_rectangle(x, y, P1, P2, P3, P4):
    # Define edge vectors and normal vectors
    edges = jnp.stack([
        P2 - P1,
        P3 - P2,
        P4 - P3,
        P1 - P4
    ], axis=-1)

    normals = jnp.stack([-edges[1], edges[0]], axis=0)
    # Line equation parameters ax + by + c = 0 for each edge
    a = normals[0]
    b = normals[1]
    c = -jnp.sum(normals * jnp.stack([P1, P2, P3, P4], axis=-1), axis=0)  # Shape (4, height, width)
    
    a = a[..., jnp.newaxis, jnp.newaxis]
    b = b[..., jnp.newaxis, jnp.newaxis]
    c = c[..., jnp.newaxis, jnp.newaxis]
    
    # Signed distance from each pixel to each edge
    distances = (a * x + b * y + c) / jnp.sqrt(a**2 + b**2)
    # Signed distance function is the minimum of distances to all edges
    signed_distances = jnp.min(distances, axis=1)

    return signed_distances

def draw_rectangles_jax(image, P1, P2, P3, P4, color):
    height, width = image.shape[:-1]
    
    # Create a grid of coordinates
    xx, yy = jnp.meshgrid(jnp.arange(width), jnp.arange(height))
    
    # Calculate signed distances for each pixel
    signed_distances = signed_distance_to_rectangle(xx, yy, P1, P2, P3, P4)
    # Threshold the signed distances to determine inside/outside of the rectangle
    inside_rectangle = signed_distances >= 0
    inside_rectangles = jnp.any(inside_rectangle, axis=0)

    return apply_color_mask(image, inside_rectangles, color=color)

def draw_circles(image, centers, radius, color):
    """
    Draw circles on the image at the specified centers with the given radius and color.

    Parameters:
    - image: Input image as a NumPy array of shape (height, width, 3).
    - centers: Array of shape (N, 2) where each row represents the (x, y) coordinates of a circle center.
    - radius: Radius of the circles to be drawn.
    - color: Tuple (R, G, B) representing the color of the circles.

    Returns:
    - Image with circles drawn on it.
    """
    height, width, _ = image.shape
    
    # Compute the distance from each pixel to the nearest center
    distances = distance_to_points((height, width), centers)
    
    mask = distances <= radius
    return apply_color_mask(image, mask, color)

def plot_jnp_bounding_boxes(
    img: jnp.ndarray,
    bboxes: jnp.ndarray,
    color: jnp.ndarray,
) -> None:
    c = jnp.cos(bboxes[:, 4])
    s = jnp.sin(bboxes[:, 4])
    pt = jnp.array((bboxes[:, 0], bboxes[:, 1]))  # (2, N)
    length, width = bboxes[:, 2], bboxes[:, 3]
    u = jnp.array((c, s))
    ut = jnp.array((s, -c))

    # Compute box corner coordinates.
    tl = pt + length / 2 * u - width / 2 * ut
    tr = pt + length / 2 * u + width / 2 * ut
    br = pt - length / 2 * u + width / 2 * ut
    bl = pt - length / 2 * u - width / 2 * ut

    # Compute heading arrow using center left/right/front.
    cl = pt - width / 2 * ut
    cr = pt + width / 2 * ut
    cf = pt + length / 2 * u
    
    start_points = jnp.concat([cl, cr, cf], axis=-1)
    end_points = jnp.concat([cr, cf, cl], axis=-1)
    # Draw heading arrow.
    img = draw_rectangles_jax(img, tl, bl, br, tr, color=color)
    img = draw_lines_jax(img, start_points, end_points, color=(255, 0, 0))

    return img

def plot_roadgraph_points_jnp(
    img, 
    rg_pts,
) -> None:
    if len(rg_pts.shape) != 1:
        raise ValueError(f'Roadgraph should be rank 1, got {len(rg_pts.shape)}')

    xy = jnp.where(rg_pts.valid[..., jnp.newaxis], rg_pts.xy * 5 + 92, -1000)
    rg_type = jnp.where(rg_pts.valid, rg_pts.types, 0)
    for curr_type in _RoadGraphShown:
        # p1 = xy[rg_type == curr_type]
        p1 = jnp.where((rg_type == curr_type)[..., jnp.newaxis], xy, -1000)
        rg_color = (color.ROAD_GRAPH_COLORS.get(int(curr_type), _RoadGraphDefaultColor) * 255).astype(jnp.uint8)
        img = draw_circles(img, p1, radius=1, color=rg_color)

    return img

def plot_traffic_light_signals_as_points_jnp(
    img, 
    tls: datatypes.TrafficLights,
    timestep: int = 0,
) -> None:
    valid = tls.valid[:, timestep]

    # tls_xy = tls.xy[:, timestep][valid] * 5 + 92
    tls_xy = jnp.where(valid[..., jnp.newaxis], tls.xy[:, timestep] * 5 + 92, -1000)
    # tls_state = tls.state[:, timestep][valid]
    tls_state = jnp.where(valid, tls.state[:, timestep], 0)

    for curr_type, curr_color in TRAFFIC_LIGHT_COLORS.items():
        p1 = jnp.where((tls_state == curr_type)[..., jnp.newaxis], tls_xy, -1000)
        tl_color = (jnp.array(curr_color) * 255).astype(jnp.uint8)
        img = draw_circles(img, p1, radius=3, color=tl_color)


    # for xy, state in zip(tls_xy, tls_state):

    #     tl_color = (jnp.array(color.TRAFFIC_LIGHT_COLORS[state]) * 255).astype(jnp.uint8)
    #     img = draw_circles(img, xy[jnp.newaxis], radius=3, color=tl_color)

    return img

def plot_trajectory_jnp(
    img,
    traj: datatypes.Trajectory,
    is_controlled: jnp.ndarray,
    time_idx=None,
    with_ego: bool = False,
) -> None:
    traj_xy = traj.stack_fields(['x', 'y']) * 5 + 96
    traj_lw = traj.stack_fields(['length', 'width']) * 5
    traj_yaw = traj.stack_fields(['yaw'])
    traj_5dof = jnp.concat([traj_xy, traj_lw, traj_yaw], axis=-1)
    
    valid_controlled = is_controlled[:, jnp.newaxis] & traj.valid
    valid_context = ~is_controlled[:, jnp.newaxis] & traj.valid

    num_obj = traj_5dof.shape[0]
    time_indices = jnp.tile(
        jnp.arange(traj_5dof.shape[1])[jnp.newaxis, :], (num_obj, 1)
    )

    if with_ego:
        traj_filtered = jnp.where(((time_indices == time_idx) & valid_controlled)[..., jnp.newaxis], traj_5dof, -1000)
        img = plot_jnp_bounding_boxes(
            img,
            # bboxes=traj_5dof[(time_indices == time_idx) & valid_controlled],
            bboxes=traj_filtered[:, 0],
            color=(0, 255, 0)
        )

    traj_filtered = jnp.where(((time_indices == time_idx) & valid_context)[..., jnp.newaxis], traj_5dof, -1000)
    img = plot_jnp_bounding_boxes(
        img,
        # bboxes=traj_5dof[(time_indices == time_idx) & valid_context],
        bboxes=traj_filtered[:, 0],
        color=(0, 0, 255)
    )
    return img

@jax.jit
def rasterize_observation_jnp(
    obs: datatypes.Observation,
) -> jnp.ndarray:
    img = jnp.zeros((192, 192, 3), dtype=jnp.uint8)
    px_per_meter = 5

    # 1. Plots trajectory.
    traj = datatypes.dynamic_index(obs.trajectory, 0, axis=0, keepdims=False)
    roadpoints = datatypes.dynamic_index(obs.roadgraph_static_points, 0, axis=0, keepdims=False)
    traffic_lights = datatypes.dynamic_index(obs.traffic_lights, 0, axis=0, keepdims=False)
    is_controlled = obs.is_ego[0]

    img = plot_roadgraph_points_jnp(img, roadpoints)
    img = plot_traffic_light_signals_as_points_jnp(img, traffic_lights, 0)
    img = plot_trajectory_jnp(img, traj, is_controlled, time_idx=0)  # pytype: disable=wrong-arg-types  # jax-ndarray

    return jnp.flipud(img)
