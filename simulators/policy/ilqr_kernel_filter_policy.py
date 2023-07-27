from typing import Tuple, Optional, Dict
import time

import jax
from jax import numpy as jnp

import copy
import numpy as np

#from .ilqr_reachability_policy import iLQRReachability
from .ilqr_reachavoid_policy import iLQRReachAvoid
from .ilqr_policy import iLQR
from .solver_utils import barrier_filter_linear, barrier_filter_quadratic, bicycle_linear_task_policy

from ..dynamics.base_dynamics import BaseDynamics
from ..costs.base_margin import BaseMargin


class iLQRKernelSafetyFilter(iLQR):

  def __init__( self, id: str, config, dyn: BaseDynamics, cost: BaseMargin, 
               task_cost:BaseMargin, **kwargs) -> None:
    super().__init__(id, config, dyn, cost)
    self.config = config

    self.filter_type = config.FILTER_TYPE
    self.constraint_type = config.CBF_TYPE
    self.gamma = config.BARRIER_GAMMA

    self.filter_steps = 0
    self.barrier_filter_steps = 0

    self.dyn = copy.deepcopy( dyn )
    self.cost = copy.deepcopy( cost )
    self.task_cost = task_cost

    self.rollout_dyn_0 = copy.deepcopy( dyn )
    self.rollout_dyn_1 = copy.deepcopy( dyn )
    self.rollout_dyn_2 = copy.deepcopy( dyn )

    self.filtering_model = kwargs['filtering_model']

    self.dim_x = dyn.dim_x
    self.dim_u = dyn.dim_u
    self.N = config.N

    self.is_task_ilqr = self.config.is_task_ilqr

    if self.config.is_task_ilqr:
      self.task_policy = iLQR(self.id, self.config, self.rollout_dyn_2, self.task_cost)
    else:
      self.task_policy = bicycle_linear_task_policy
    # Two ILQ solvers
    if self.config.COST_TYPE=="Reachavoid":
      self.solver_0 = iLQRReachAvoid(self.id, self.config, self.rollout_dyn_0, self.cost)
      self.solver_1 = iLQRReachAvoid(self.id, self.config, self.rollout_dyn_1, self.cost)
      self.solver_2 = iLQRReachAvoid(self.id, self.config, self.rollout_dyn_1, self.cost)

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      prev_sol: Optional[Dict] = None, warmup=False, **kwargs
  ) -> np.ndarray:

    # Cruise policy
    start_time = time.time()
    initial_state = np.array(kwargs['state'])

    if self.config.is_task_ilqr:
      task_ctrl, _ = self.task_policy.get_action(obs, None, **kwargs)
    else:
      task_ctrl = self.task_policy( initial_state )

    solver_info = {}
    V0, _ = self.filtering_model.predict_model( initial_state[np.newaxis, :] )
    print(V0)
    solver_info['Vopt'] = V0[0]
    solver_info['status'] = 1
    solver_info['marginopt'] = V0[0]
    solver_info['is_inside_target_next'] = False
    gamma = self.gamma
    cutoff = gamma*solver_info['Vopt']

    state_imaginary, task_ctrl = self.dyn.integrate_forward(
            state=initial_state, control=task_ctrl
    )
    V1, Vgrad1 = self.filtering_model.predict_model( state_imaginary[np.newaxis, :] )
    solver_info['Vopt_next'] = V1[0]
    solver_info['marginopt_next'] = V1[0]
    solver_info['is_inside_target_next'] = False

    initial_control = task_ctrl
    initial_state_jnp = jnp.array( initial_state[:, np.newaxis] )
    initial_control_jnp = jnp.array( initial_control[:, np.newaxis] )
    
    # Setting tolerance to zero does not cause big improvements at the cost of more unnecessary looping
    cbf_tol = -1e-5

    # Checking CBF constraint violation
    constraint_violation = solver_info['Vopt_next'] - cutoff
    scaled_c = constraint_violation
      
    #Scaling parameter
    kappa = 1.2
    num_iters = 0

    # Exit loop once CBF constraint satisfied or maximum iterations violated
    while((constraint_violation<cbf_tol or warmup) and num_iters<5):
      num_iters= num_iters + 1

      # Extract information from solver for enforcing constraint
      grad_x = jnp.array(Vgrad1.ravel())
      _, B0 = self.dyn.get_jacobian( initial_state_jnp , initial_control_jnp )

      control_correction = barrier_filter_linear(grad_x, B0, scaled_c)

      filtered_control = initial_control + np.array( control_correction.ravel() )

      # Restart from current point and run again
      initial_control = np.array(filtered_control)

      state_imaginary, initial_control = self.dyn.integrate_forward(
            state=initial_state, control=initial_control
        )
      
      V1, Vgrad1 = self.filtering_model.predict_model( state_imaginary[np.newaxis, :] )

      solver_info['Vopt_next'] = V1[0]
      solver_info['marginopt_next'] = V1[0]
      solver_info['is_inside_target_next'] = False

      initial_control_jnp = jnp.array( initial_control[:, np.newaxis] )

      # CBF constraint violation
      constraint_violation = solver_info['Vopt_next'] - cutoff
      scaled_c = kappa*constraint_violation

    if True:
      if num_iters>0:
        self.barrier_filter_steps += 1
      solver_info['mark_barrier_filter'] = True
      solver_info['mark_complete_filter'] = False
      solver_info['barrier_filter_steps'] = self.barrier_filter_steps
      solver_info['filter_steps'] = self.filter_steps
      solver_info['process_time'] = time.time() - start_time
      solver_info['resolve'] = False
      solver_info['bootstrap_next_solution'] = None
      solver_info['reinit_controls'] = None
      solver_info['reinit_states'] = None
      #solver_info_0['reinit_J'] = solver_info_1['Vopt'] 
      solver_info['num_iters'] = num_iters
      solver_info['deviation'] = np.linalg.norm(initial_control - task_ctrl, ord=1)
      solver_info['qcqp_initialize'] = initial_control - task_ctrl
      
      return initial_control.ravel(), solver_info