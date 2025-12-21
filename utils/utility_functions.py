import jax
from functools import partial
import jax.numpy as jnp
cpu = jax.devices("cpu")[0]
try:
    gpu = jax.devices("cuda")[0]
except:
    gpu = cpu

jnp.set_printoptions(precision=4)

jax.config.update("jax_enable_x64", False)



# Obstacle map generator
def generate_obstacle_map(
    num_obstacles: int, 
    map_size: tuple,
    obstacle_radius_range: tuple,
    map_resolution: float = 0.2,
    seed: int = 0
):
    """
    map_size: (size_x, size_y) in meters
    map_resolution: voxel size in meters
    World frame is assumed to be centered at (0, 0), so map spans
        x in [-map_size[0]/2, map_size[0]/2]
        y in [-map_size[1]/2, map_size[1]/2]
    """

    rng = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(rng)

    # Grid dimensions (rows: y, cols: x)
    nx = int(map_size[0] / map_resolution)  # number of cells along x
    ny = int(map_size[1] / map_resolution)  # number of cells along y

    # obstacle_map[row, col] = occupied?
    obstacle_map = jnp.zeros((ny, nx), dtype=bool)
    obstacles = []

    # Sample random circular obstacles (in world coordinates)
    for _ in range(num_obstacles):
        key, subkey = jax.random.split(key)
        x = jax.random.uniform(subkey, minval=-map_size[0] / 2, maxval=map_size[0] / 2)
        key, subkey = jax.random.split(key)
        y = jax.random.uniform(subkey, minval=-map_size[1] / 2, maxval=map_size[1] / 2)
        key, subkey = jax.random.split(key)
        radius = jax.random.uniform(subkey, minval=obstacle_radius_range[0], maxval=obstacle_radius_range[1])
        obstacles.append((x, y, radius))

    # Rasterize circles into the grid
    for (x, y, radius) in obstacles:
        cx = int((x + map_size[0] / 2) / map_resolution)  # col index
        cy = int((y + map_size[1] / 2) / map_resolution)  # row index
        r = int(radius / map_resolution)

        # grid indices relative to the center cell
        y_grid, x_grid = jnp.ogrid[
            -cy:ny-cy,
            -cx:nx-cx
        ]
        mask = x_grid**2 + y_grid**2 <= r**2
        obstacle_map = jnp.where(mask, True, obstacle_map)

    # ---- Vectorized perimeter extraction (4-neighborhood) ----
    occ = obstacle_map
    inner = occ[1:-1, 1:-1]

    # cell is interior if all 4 neighbors are occupied
    all_neighbors_occupied = (
        inner &
        occ[:-2, 1:-1] &  # up
        occ[2:, 1:-1] &   # down
        occ[1:-1, :-2] &  # left
        occ[1:-1, 2:]     # right
    )

    perimeter_map_full = jnp.zeros_like(occ, dtype=bool)

    # perimeter = occupied but not fully surrounded
    perimeter_inner = inner & ~all_neighbors_occupied

    # insert back into full-size map
    perimeter_map_full = jnp.zeros_like(occ, dtype=bool)
    perimeter_map_full = perimeter_map_full.at[1:-1, 1:-1].set(perimeter_inner)

    # indices of boundary cells (row, col)
    boundary_indices = jnp.argwhere(perimeter_map_full)
    # ---- Convert indices to world (x, y) coordinates ----
    if boundary_indices.size == 0:
        boundary_points = jnp.empty((0, 2), dtype=float)
    else:
        rows = boundary_indices[:, 0]  # y indices
        cols = boundary_indices[:, 1]  # x indices

        # cell centers in world coordinates
        xs = (cols + 0.5) * map_resolution - map_size[0] / 2.0
        ys = (rows + 0.5) * map_resolution - map_size[1] / 2.0

        boundary_points = jnp.stack([xs, ys], axis=-1)  # shape (N, 2)

    # Obstacle points in world coordinates
    obstacle_rows, obstacle_cols = jnp.where(obstacle_map)
    obs_xs = (obstacle_cols + 0.5) * map_resolution - map_size[0] / 2.0
    obs_ys = (obstacle_rows + 0.5) * map_resolution - map_size[1] / 2.0

    obstacle_points = jnp.stack([obs_xs, obs_ys], axis=-1)  # shape (N, 2)
    return obstacle_map, obstacle_points, obstacles, boundary_indices, boundary_points
    