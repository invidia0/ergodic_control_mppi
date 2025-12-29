import jax.numpy as jnp


class Bicycle():
    def __init__(
            self,
            delta_t: float = 0.01,
            max_steer_abs: float = 0.523,
            max_accel_abs: float = 2.0,
            wheelbase: float = 2.5,
            w_accel: float = 1.0,
            w_steer: float = 1.0,
            ):
        self.delta_t = delta_t
        self.max_steer_abs = max_steer_abs
        self.max_accel_abs = max_accel_abs
        self.wheelbase = wheelbase

        # control cost weights for mppi
        self.w_accel = w_accel
        self.w_steer = w_steer

    def step(self, x, u):
        x_next = self._F(x, self._g(u))
        return x_next, x_next
    
    def _g(self, v: jnp.ndarray) -> jnp.ndarray:
        """clamp input: u = [accel, steer] - JAX-compatible version"""
        v_clamped = jnp.array([
            jnp.clip(v[0], -self.max_accel_abs, self.max_accel_abs),
            jnp.clip(v[1], -self.max_steer_abs, self.max_steer_abs)
        ])
        return v_clamped
    
    def _F(self, x_t: jnp.ndarray, v_t: jnp.ndarray) -> jnp.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        x, y, yaw, v = x_t
        accel, steer = v_t

        # prepare params
        l = self.wheelbase
        dt = self.delta_t

        # update state variables
        new_x = x + v * jnp.cos(yaw) * dt
        new_y = y + v * jnp.sin(yaw) * dt
        new_yaw = yaw + v / l * jnp.tan(steer) * dt
        new_v = v + accel * dt

        # return updated state
        x_t_plus_1 = jnp.array([new_x, new_y, new_yaw, new_v])
        return x_t_plus_1
    
    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        accel, steer = u
        cost_accel = self.w_accel * (accel ** 2)
        cost_steer = self.w_steer * (steer ** 2)
        return cost_accel + cost_steer
    


class DoubleIntegrator():
    def __init__(
            self,
            delta_t: float = 0.01,
            max_vel_abs: float = 5.0,
            max_accel_lin_abs: float = 10.0,
            max_accel_ang_abs: float = 10.0,
            w_accel_lin: float = 10.0,
            w_accel_ang: float = 20.0,
            ):
        self.delta_t = delta_t
        self.max_vel_abs = max_vel_abs
        self.max_accel_lin_abs = max_accel_lin_abs
        self.max_accel_ang_abs = max_accel_ang_abs
        # control cost weights for mppi
        self.w_accel_lin = w_accel_lin
        self.w_accel_ang = w_accel_ang

    def step(self, x, u):
        x_next = self._F(x, u)
        return x_next, x_next

    def _F(self, x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        px, py, vx, vy, yaw, yaw_rate = x_t
        ax, ay, w = u_t

        # prepare params
        dt = self.delta_t

        # update state variables
        new_px = px + vx * dt + 0.5 * ax * dt**2
        new_py = py + vy * dt + 0.5 * ay * dt**2
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt
        new_yaw = yaw + yaw_rate * dt + 0.5 * w * dt**2
        new_yaw_rate = yaw_rate + w * dt

        # return updated state
        x_t_plus_1 = jnp.array([new_px, new_py, new_vx, new_vy, new_yaw, new_yaw_rate])
        return x_t_plus_1
    
    def _g(self, u: jnp.ndarray) -> jnp.ndarray:
        """clamp input: u = [acc_x, acc_y] - JAX-compatible version"""
        v_clamped = jnp.array([
            jnp.clip(u[0], -self.max_accel_lin_abs, self.max_accel_lin_abs),
            jnp.clip(u[1], -self.max_accel_lin_abs, self.max_accel_lin_abs),
            jnp.clip(u[2], -self.max_accel_ang_abs, self.max_accel_ang_abs),
        ])
        return v_clamped
    
    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Example stage cost function."""
        accel_lin_x, accel_lin_y, accel_ang = u
        cost_accel_lin = self.w_accel_lin * (accel_lin_x ** 2 + accel_lin_y ** 2)
        cost_accel_ang = self.w_accel_ang * (accel_ang ** 2)
        return cost_accel_lin + cost_accel_ang


class DoubleIntegratorNoYaw():
    def __init__(
            self,
            delta_t: float = 0.01,
            max_vel_abs: float = 5.0,
            max_accel_lin_abs: float = 10.0,
            w_accel_lin: float = 50.0,
            ):
        self.delta_t = delta_t
        self.max_vel_abs = max_vel_abs
        self.max_accel_lin_abs = max_accel_lin_abs
        # control cost weights for mppi
        self.w_accel_lin = w_accel_lin

    def step(self, x, u):
        x_next = self._F(x, u)
        return x_next, x_next

    def _F(self, x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        px, py, vx, vy = x_t
        ax, ay = u_t

        # prepare params
        dt = self.delta_t

        # update state variables
        new_px = px + vx * dt #+ 0.5 * ax * dt**2
        new_py = py + vy * dt #+ 0.5 * ay * dt**2
        new_vx = vx + ax * dt
        new_vy = vy + ay * dt

        # return updated state
        x_t_plus_1 = jnp.array([new_px, new_py, new_vx, new_vy])
        return x_t_plus_1
    
    def _g(self, u: jnp.ndarray) -> jnp.ndarray:
        """clamp input: u = [acc_x, acc_y] - JAX-compatible version"""
        v_clamped = jnp.array([
            jnp.clip(u[0], -self.max_accel_lin_abs, self.max_accel_lin_abs),
            jnp.clip(u[1], -self.max_accel_lin_abs, self.max_accel_lin_abs),
        ])
        return v_clamped
    
    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Example stage cost function."""
        accel_lin_x, accel_lin_y = u
        cost_accel_lin = self.w_accel_lin * (accel_lin_x ** 2 + accel_lin_y ** 2)
        return cost_accel_lin
    
class SingleIntegrator():
    def __init__(
            self,
            delta_t: float = 0.01,
            max_vel_abs: float = 5.0,
            w_vel: float = 10.0,
            ):
        self.delta_t = delta_t
        self.max_vel_abs = max_vel_abs
        # control cost weights for mppi
        self.w_vel = w_vel

    def step(self, x, u):
        x_next = self._F(x, u)
        return x_next, x_next

    def _F(self, x_t: jnp.ndarray, u_t: jnp.ndarray) -> jnp.ndarray:
        """calculate next state of the vehicle"""
        # get previous state variables
        px, py = x_t
        vx, vy = u_t

        # prepare params
        dt = self.delta_t

        # update state variables
        new_px = px + vx * dt
        new_py = py + vy * dt

        # return updated state
        x_t_plus_1 = jnp.array([new_px, new_py])
        return x_t_plus_1
    
    def _g(self, u: jnp.ndarray) -> jnp.ndarray:
        """clamp input: u = [vel_x, vel_y] - JAX-compatible version"""
        v_clamped = jnp.array([
            jnp.clip(u[0], -self.max_vel_abs, self.max_vel_abs),
            jnp.clip(u[1], -self.max_vel_abs, self.max_vel_abs),
        ])
        return v_clamped
    
    def stage_cost(self, x: jnp.ndarray, u: jnp.ndarray) -> float:
        """Example stage cost function."""
        vel_x, vel_y = u
        cost_vel = self.w_vel * (vel_x ** 2 + vel_y ** 2)
        return cost_vel