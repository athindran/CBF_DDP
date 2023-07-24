from typing import Dict
from functools import partial

from jaxlib.xla_extension import DeviceArray
import jax.numpy as jnp
import jax
import math

from ..costs.base_margin import BaseMargin, SoftBarrierEnvelope
from ..costs.obs_margin import CircleObsMargin
from ..costs.quadratic_penalty import QuadraticControlCost
from ..costs.half_space_margin import LowerHalfMargin, UpperHalfMargin
from ..costs.joint_margin import JointLowerHalfMargin

class Integrator2DCost(BaseMargin):

  def __init__(self, config, plan_dyn):
    super().__init__()
    
    # Lagrange cost parameters.
    self.v_ref = config.V_REF  # reference velocity.
    
    self.w_vel = config.W_VEL
    self.w_accel = config.WT_ACCEL
    self.w_omega = config.WT_OMEGA
    self.w_track = config.W_TRACK

    # Soft constraint parameters.
    self.q1_v = config.Q1_V
    self.q2_v = config.Q2_V

    self.v_min = config.V_MIN
    self.v_max = config.V_MAX

    self.barrier_clip_min = config.BARRIER_CLIP_MIN
    self.barrier_clip_max = config.BARRIER_CLIP_MAX
    self.buffer = getattr(config, "BUFFER", 0.)

    self.dim_x = plan_dyn.dim_x
    self.dim_u = plan_dyn.dim_u

    self.vel_x_max_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
            UpperHalfMargin(value=self.v_max, buffer=0, dim=2)
        )
    
    self.vel_x_min_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
            LowerHalfMargin(value=self.v_min, buffer=0, dim=2)
        )
    
    self.vel_y_max_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
            UpperHalfMargin(value=self.v_max, buffer=0, dim=3)
        )
    
    self.vel_y_min_barrier_cost = SoftBarrierEnvelope(
            self.barrier_clip_min, self.barrier_clip_max, self.q1_v, self.q2_v,
            LowerHalfMargin(value=self.v_min, buffer=0, dim=3)
        )


  @partial(jax.jit, static_argnames='self')
  def get_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    # control cost

    cost = self.w_accel * ctrl[0]**2 + self.w_accel * ctrl[1]**2

    # state cost
    cost += self.w_vel * (state[2] - self.v_ref)**2

    cost += self.w_track * state[1]**2

    # soft constraint cost
    cost += self.vel_x_max_barrier_cost.get_stage_margin(state, ctrl)
    cost += self.vel_x_min_barrier_cost.get_stage_margin(state, ctrl)
    cost += self.vel_y_max_barrier_cost.get_stage_margin(state, ctrl)
    cost += self.vel_y_min_barrier_cost.get_stage_margin(state, ctrl)

    return cost
  
  @partial(jax.jit, static_argnames='self')
  def get_target_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    return self.get_stage_margin(
        state, ctrl
    )

class Integrator2DConstraintMargin( BaseMargin ):
  def __init__(self, config, plan_dyn):
    super().__init__()
    # System parameters.
    self.ego_radius = config.EGO_RADIUS

    # Racing cost parameters.
    self.w_accel = config.W_ACCEL

    self.use_yaw = getattr(config, 'USE_YAW', False)
    self.use_vel = getattr(config, 'USE_VEL', False)

    self.use_delta = False
    self.use_road = False

    self.obs_spec = config.OBS_SPEC
    self.obsc_type = config.OBSC_TYPE
    self.plan_dyn = plan_dyn

    self.dim_x = plan_dyn.dim_x
    self.dim_u = plan_dyn.dim_u

    self.v_min = config.V_MIN
    self.v_max = config.V_MAX
    
    self.obs_constraint = []
    if self.obsc_type=='circle':
      for circle_spec in self.obs_spec:
        self.obs_constraint.append(
          CircleObsMargin(
              circle_spec=circle_spec, buffer=config.EGO_RADIUS
          )
        )

    self.vmin_cost = JointLowerHalfMargin(self.v_min, dim0=2, dim1=3, buffer=0.)
  
  @partial(jax.jit, static_argnames='self')
  def get_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    
    """
    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    cost = jnp.inf

    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseMargin
      cost = jnp.minimum(cost, _obs_constraint.get_stage_margin(state, ctrl))

    #cost = jnp.minimum(cost, self.vmin_cost.get_stage_margin(state, ctrl))
    #cost = jnp.minimum(cost, self.vmin_y_cost.get_stage_margin(state, ctrl))

    return cost
  
  @partial(jax.jit, static_argnames='self')
  def get_target_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    
    """
    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    pass

  @partial(jax.jit, static_argnames='self')
  def get_cost_dict(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> Dict:
    """
    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """

    obs_cons = jnp.inf
    for _obs_constraint in self.obs_constraint:
      _obs_constraint: BaseMargin
      obs_cons = jnp.minimum(obs_cons, _obs_constraint.get_stage_margin(state, ctrl))
 
    return dict(
            obs_cons=obs_cons
        )
  
class IntegratorReachability2DMargin(BaseMargin):

  def __init__(self, config, plan_dyn):
    super().__init__()
    # Removing the square
    self.constraint = Integrator2DConstraintMargin(config, plan_dyn)
    R = jnp.array([[config.W_ACCEL, 0.0], [0.0, config.W_ACCEL]])
    self.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(plan_dyn.dim_u))
    self.constraint.ctrl_cost = QuadraticControlCost(R=R, r=jnp.zeros(plan_dyn.dim_u))
    self.N = config.N

  @partial(jax.jit, static_argnames='self')
  def get_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    """

    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    state_cost = self.constraint.get_stage_margin(
        state, ctrl
    )
    ctrl_cost = self.ctrl_cost.get_stage_margin(state, ctrl)
    return state_cost + ctrl_cost
  
  @partial(jax.jit, static_argnames='self')
  def get_target_stage_margin(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> DeviceArray:
    
    """
    Args:
        state (DeviceArray, vector shape)
        ctrl (DeviceArray, vector shape)

    Returns:
        DeviceArray: scalar.
    """
    pass
  
  #UNUSED FUNCTION   
  @partial(jax.jit, static_argnames='self')
  def get_traj_cost(
      self, state: DeviceArray, ctrl: DeviceArray
  ) -> float:
    state_costs = self.constraint.get_stage_margin(
        state, ctrl
    )

    ctrl_costs = self.ctrl_cost.get_stage_margin(state, ctrl)
    # TODO: critical points version

    return (jnp.min(state_costs[1:]) + jnp.sum(ctrl_costs)).astype(float)
