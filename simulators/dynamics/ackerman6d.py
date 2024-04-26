"""
Please contact the author(s) of this library if you have any questions.
Authors:  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""

from typing import Tuple, Any
import numpy as np
from functools import partial
from jaxlib.xla_extension import DeviceArray
import jax
from jax import numpy as jnp
from jax import custom_jvp

from .base_dynamics import BaseDynamics


class Ackerman6D(BaseDynamics):

  def __init__(self, config: Any, action_space: np.ndarray) -> None:
    """
    Implements the bicycle dynamics (for Princeton race car). The state is the
    center of the rear axis.
    Args:
        config (Any): an object specifies configuration.
        action_space (np.ndarray): action space.
    """
    super().__init__(config, action_space)
    self.dim_x = 6  # [x, y, v, psi, delta].

    # load parameters
    self.wheelbase: float = config.WHEELBASE  # vehicle chassis length
    self.delta_min = config.DELTA_MIN
    self.delta_max = config.DELTA_MAX
    self.v_min = config.V_MIN
    self.v_max = config.V_MAX

  @partial(jax.jit, static_argnames='self')
  def integrate_forward_jax(
      self, state: DeviceArray, control: DeviceArray
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

    #! hacky
    state_nxt = self._integrate_forward_dt(state, ctrl_clip, self.dt)
    state_nxt = state_nxt.at[2].set(
        jnp.clip(state_nxt[2], self.v_min, self.v_max)
    )
    state_nxt = state_nxt.at[4].set(
        jnp.clip(state_nxt[4], self.delta_min, self.delta_max)
    )
    state_nxt = state_nxt.at[5].set(
        jnp.clip(state_nxt[5], self.delta_min, self.delta_max)
    )
    return state_nxt, ctrl_clip

  @partial(jax.jit, static_argnames='self')
  def disc_deriv(
      self, state: DeviceArray, control: DeviceArray
  ) -> DeviceArray:
    deriv = jnp.zeros((self.dim_x,))
    cotdelta = 0.5*(1/(jnp.tan(state[4])+1e-8) + 1/(jnp.tan(state[5])+1e-8))
    deriv = deriv.at[0].set(state[2] * jnp.cos(state[3]))
    deriv = deriv.at[1].set(state[2] * jnp.sin(state[3]))
    deriv = deriv.at[2].set(control[0])
    deriv = deriv.at[3].set(state[2] / (cotdelta*self.wheelbase+1e-8))
    deriv = deriv.at[4].set(control[1])
    deriv = deriv.at[5].set(control[2])
    return deriv

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward(
      self, state: DeviceArray, control: DeviceArray
  ) -> DeviceArray:
    """ Computes one-step time evolution of the system: x_+ = f(x, u).
    The discrete-time dynamics is as below:
        x_k+1 = x_k + v_k cos(psi_k) dt
        y_k+1 = y_k + v_k sin(psi_k) dt
        v_k+1 = v_k + u0_k dt
        psi_k+1 = psi_k + v_k tan(delta_k) / L dt
        delta_k+1 = delta_k + u1_k dt
    Args:
        state (DeviceArray): [x, y, v, psi, delta].
        control (DeviceArray): [accel, omega].
    Returns:
        DeviceArray: next state.
    """
    # @jax.jit
    # def _fwd_step(i, args):  # Euler method.
    #   _state, _ctrl = args
    #   return self._integrate_forward_dt(_state, _ctrl, self.int_dt), _ctrl

    # state_nxt = jax.lax.fori_loop(
    #     0, self.num_segment, _fwd_step, (state, control)
    # )[0]
    # return state_nxt
    return self._integrate_forward_dt(state, control, self.dt)

  @partial(jax.jit, static_argnames='self')
  def _integrate_forward_dt(
      self, state: DeviceArray, control: DeviceArray, dt: float
  ) -> DeviceArray:
    k1 = self.disc_deriv(state, control)
    k2 = self.disc_deriv(state + k1*dt/2, control)
    k3 = self.disc_deriv(state + k2*dt/2, control)
    k4 = self.disc_deriv(state + k3*dt, control)
    return state + (k1 + 2*k2 + 2*k3 + k4) * dt / 6