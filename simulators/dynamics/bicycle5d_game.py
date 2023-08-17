from typing import Tuple, Any
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp
from jax import random

from .base_dynamics_game import BaseDynamics


class Bicycle5DGame(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray, disturbance_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.
    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    super().__init__(config, action_space, disturbance_space)
    self.dim_x = 5  # [x, y, v, psi, delta].

    # load parameters
    self.wheelbase: float = config.WHEELBASE  # vehicle chassis length
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.v_min = 0
    self.v_max = config.V_MAX

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray, disturb: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    """Clips the control and computes one-step time evolution of the system.
    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].
    Returns:
        DeviceArray: next state.
        DeviceArray: clipped control.
    """
    # Clips the controller values between min and max accel and steer values.
    ctrl_clip = jnp.clip(control, self.ctrl_space[:, 0], self.ctrl_space[:, 1])
    disturb_clip = jnp.clip(disturb, self.dstb_space[:, 0], self.dstb_space[:, 1])

    state_nxt = self._integrate_forward(state, ctrl_clip, disturb_clip)

    return state_nxt, ctrl_clip, disturb_clip

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray
  ) -> DeviceArray:
    deriv = jnp.zeros((self.dim_x,))
    deriv = deriv.at[0].set(state[2] * jnp.cos(state[3])) + disturbance[0]
    deriv = deriv.at[1].set(state[2] * jnp.sin(state[3])) + disturbance[1]
    deriv = deriv.at[2].set(control[0])
    deriv = deriv.at[3].set(state[2] * jnp.tan(state[4]) / self.wheelbase)
    deriv = deriv.at[4].set(control[1])
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray, disturbance: DeviceArray = jnp.zeros((2,))
  ) -> DeviceArray:
    """ Computes one-step time evolution of the system: x_+ = f(x, u).
    The discrete-time dynamics is as below:
        x_k+1 = x_k + v_k cos(psi_k) dt + d0  dt
        y_k+1 = y_k + v_k sin(psi_k) dt + d1 dt
        v_k+1 = v_k + u0_k dt
        psi_k+1 = psi_k + v_k tan(delta_k) / L dt
        delta_k+1 = delta_k + u1_k dt
    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].
    Returns:
        DeviceArray: next state.
    """
    return self._integrate_forward_dt(jnp.asarray(state), jnp.asarray(control), jnp.asarray(disturbance), self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: DeviceArray, ctrl_clip: DeviceArray, dist_clip, dt:float
  ) -> DeviceArray:
    k1 = self.disc_deriv(state, ctrl_clip, dist_clip)
    k2 = self.disc_deriv(state + k1*dt/2, ctrl_clip, dist_clip)
    k3 = self.disc_deriv(state + k2*dt/2, ctrl_clip, dist_clip)
    k4 = self.disc_deriv(state + k3*dt, ctrl_clip, dist_clip)
    
    state_nxt = state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6
    state_nxt = state_nxt.at[2].set(
         jnp.clip(state_nxt[2], self.v_min, self.v_max)
    )
    
    state_nxt = state_nxt.at[4].set(
         jnp.clip(state_nxt[4], self.delta_min, self.delta_max)
    )

    return state_nxt
  
  @partial(jax.jit, static_argnames='self')
  def get_jacobian_fx(
      self, obs: DeviceArray, control: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
      Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -obs[2]*jnp.sin(obs[3]), 0],
                      [0, 0, jnp.sin(obs[3]), obs[2]*jnp.cos(obs[3]), 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, jnp.tan(obs[4])/self.wheelbase, 0, obs[2]/(1e-6 + self.wheelbase*jnp.cos(obs[4])**2)],
                      [0, 0, 0, 0, 0]  ])
      
      Ad = jnp.eye(self.dim_x) + Ac*self.dt + 0.5*Ac@Ac*self.dt*self.dt
        
      return Ad
  
  @partial(jax.jit, static_argnames='self')
  def get_jacobian_fu(
      self, obs: DeviceArray, control: DeviceArray
  ) -> DeviceArray: 
      Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -obs[2]*jnp.sin(obs[3]), 0],
                      [0, 0, jnp.sin(obs[3]), obs[2]*jnp.cos(obs[3]), 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, jnp.tan(obs[4])/self.wheelbase, 0, obs[2]/(1e-6 + self.wheelbase*jnp.cos(obs[4])**2)],
                      [0, 0, 0, 0, 0]])
      
      Bc = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 0],
                      [0, 1]])
      
      Bd = self.dt * Bc

      return Bd
  
  @partial(jax.jit, static_argnames='self')
  def get_jacobian_fd(
      self, obs: DeviceArray, control: DeviceArray
  ) -> DeviceArray: 
      Dc = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0]])
      
      Dd = self.dt * Dc

      return Dd

  @partial(jax.jit, static_argnames='self')
  def get_jacobian(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray, nominal_disturbances:DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:
    jac = jax.jit(jax.vmap(self.get_jacobian_fx_fu_fd, in_axes=(1, 1, 1), out_axes=(2, 2, 2)))
    return jac(nominal_states, nominal_controls, nominal_disturbances)

  @partial(jax.jit, static_argnames='self')
  def get_jacobian_fx_fu_fd(self, obs: DeviceArray, control: DeviceArray, disturbance: DeviceArray) -> Tuple:
      Ac = jnp.array([[0, 0, jnp.cos(obs[3]), -obs[2]*jnp.sin(obs[3]), 0],
                      [0, 0, jnp.sin(obs[3]), obs[2]*jnp.cos(obs[3]), 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, jnp.tan(obs[4])/self.wheelbase, 0, obs[2]/(1e-6 + self.wheelbase*jnp.cos(obs[4])**2)],
                      [0, 0, 0, 0, 0]])
      
      Bc = np.array([[0, 0],
                      [0, 0],
                      [1, 0],
                      [0, 0],
                      [0, 1]])
      
      Dc = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0]])
            
      Ad = jnp.eye(self.dim_x) + Ac*self.dt + 0.5*Ac@Ac*self.dt*self.dt
      Bd = self.dt * Bc
      Dd = self.dt * Dc

      return Ad, Bd, Dd