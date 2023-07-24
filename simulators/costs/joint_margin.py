import numpy as np
import jax
import jax.numpy as jnp
from jaxlib.xla_extension import DeviceArray
from functools import partial

from .base_margin import BaseMargin


class JointLowerHalfMargin(BaseMargin):
  """
  c = x[dim] - val
  """
  def __init__(self, value: float, buffer:float, dim0: int, dim1:int):
    super().__init__()
    self.dim0 = dim0
    self.dim1 = dim1
    self.value = value
    self.buffer = buffer

  @partial(jax.jit, static_argnames='self')
  def get_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    c = jnp.sqrt( state[self.dim0]**2 + state[self.dim1]**2 ) - self.value
    return c - self.buffer
  
  @partial(jax.jit, static_argnames='self')
  def get_target_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    return self.get_stage_margin(state, ctrl)
