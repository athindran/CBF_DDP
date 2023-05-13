"""
Please contact the author(s) of this library if you have any questions.
Authors:  Athindran Ramesh Kumar (arkumar@princeton.edu),  Kai-Chieh Hsu ( kaichieh@princeton.edu )
"""
from typing import Tuple, Optional, Dict
import time

import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray

import copy
import numpy as np

from .base_policy import BasePolicy
from .ilqr_reachability_policy import iLQRReachability
from .ilqr_reachavoid_policy import iLQRReachAvoid
from .ilqr_policy import iLQR

from ..dynamics.base_dynamics import BaseDynamics
from ..costs.base_margin import BaseMargin

from functools import partial

from .solver_utils import barrier_filter_linear, barrier_filter_quadratic

class iLQRSafetyFilter(iLQR):

  def __init__( self, id: str, config, dyn: BaseDynamics, cost: BaseMargin, task_cost:BaseMargin, **kwargs ) -> None:
    super().__init__(id, config, dyn, cost)
    self.config = config

    self.policy_type = "iLQR"
    self.filter_type = config.FILTER_TYPE
    self.constraint_type = config.CBF_TYPE
    self.gamma = config.BARRIER_GAMMA

    self.shield_threshold = 0.0
    self.shield_steps = 0
    self.barrier_shield_steps = 0

    self.dyn = copy.deepcopy( dyn )
    self.cost = copy.deepcopy( cost )
    self.task_cost = task_cost

    self.rollout_dyn_0 = copy.deepcopy( dyn )
    self.rollout_dyn_1 = copy.deepcopy( dyn )
    self.rollout_dyn_2 = copy.deepcopy( dyn )

    self.eps = getattr(config, "EPS", 1e-6)
    self.dim_x = dyn.dim_x
    self.dim_u = dyn.dim_u
    self.N = config.N

    self.task_ilqr = iLQR(self.id, self.config, self.rollout_dyn_2, self.task_cost)
    # Two ILQ solvers
    if self.config.COST_TYPE=="Reachability":
      self.solver_0 = iLQRReachability(self.id, self.config, self.rollout_dyn_0, self.cost)
      self.solver_1 = iLQRReachability(self.id, self.config, self.rollout_dyn_1, self.cost)
      self.solver_2 = iLQRReachability(self.id, self.config, self.rollout_dyn_1, self.cost)
    elif self.config.COST_TYPE=="Reachavoid":
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
    task_policy, _ = self.task_ilqr.get_action(obs, None, **kwargs)
    #task_policy = bicycle_linear_task_policy( initial_state )

    # Find safe policy from step 0 
    if prev_sol is not None:
      controls_initialize = prev_sol['reinit_controls']
    else:
      controls_initialize = None
    control_0, solver_info_0 = self.solver_0.get_action(obs, controls_initialize, **kwargs)
    
    solver_info_0['mark_barrier_shield'] = False
    solver_info_0['mark_complete_shield'] = False
    # Find safe policy from step 1
    state_imaginary, task_policy = self.dyn.integrate_forward(
            state=initial_state, control=task_policy
    )
    kwargs['state'] = state_imaginary
    boot_controls = solver_info_0['controls']

    control_1, solver_info_1 = self.solver_1.get_action(state_imaginary, boot_controls, **kwargs)

    solver_info_0['Vopt_next'] = solver_info_1['Vopt']
    solver_info_0['marginopt_next'] = solver_info_1['marginopt']

    if(self.filter_type=="LR"):
      solver_info_0['barrier_shield_steps'] = self.barrier_shield_steps
      if(solver_info_1['Vopt']<self.shield_threshold):
        self.shield_steps += 1
        solver_info_0['process_time'] = time.time() - start_time
        solver_info_0['shield_steps'] = self.shield_steps
        solver_info_0['resolve'] = True
        solver_info_0['reinit_controls'] = solver_info_0['controls']
        solver_info_0['mark_complete_shield'] = True
        solver_info_0['num_iters'] = 0
        solver_info_0['deviation'] = np.linalg.norm(control_0 - task_policy)
        return control_0, solver_info_0
      else:
        solver_info_0['shield_steps'] = self.shield_steps
        solver_info_0['process_time'] = time.time() - start_time
        solver_info_0['resolve'] = True
        solver_info_0['reinit_controls'] = solver_info_1['controls']
        #solver_info_0['reinit_J'] = solver_info_1['Vopt'] 
        solver_info_0['reinit_states'] = solver_info_1['states']
        solver_info_0['num_iters'] = 0
        solver_info_0['deviation'] = 0
        return task_policy, solver_info_0  
    elif(self.filter_type=="CBF"):
      gamma = self.gamma
      cutoff = gamma*solver_info_0['Vopt']
      
      initial_control = task_policy

      solver_initial = np.zeros((2,))
      if prev_sol is not None:
        solver_initial = prev_sol['qcqp_initialize']

      # Define initial state and initial performance policy
      initial_state_jnp = jnp.array( initial_state[:, np.newaxis] )
      initial_control_jnp = jnp.array( initial_control[:, np.newaxis] )
      num_iters = 0

      # Setting tolerance to zero does not cause big improvements at the cost of more unnecessary looping
      cbf_tol = -1e-5
      # Conditioning parameter
      eps_reg = 1e-8

      # Checking CBF constraint violation
      constraint_violation = solver_info_1['Vopt'] - cutoff
      scaled_c = constraint_violation
      
      #Scaling parameter
      kappa = 1.2

      # Exit loop once CBF constraint satisfied or maximum iterations violated
      while((constraint_violation<cbf_tol or warmup) and num_iters<5):
        num_iters= num_iters + 1

        # Extract information from solver for enforcing constraint
        grad_x = solver_info_1['grad_x']
        _, B0 = self.dyn.get_jacobian( initial_state_jnp , initial_control_jnp )

        if self.constraint_type=='quadratic':  
          grad_xx = np.array(solver_info_1['grad_xx'])

          # Get jacobian at initial point
          B0u = B0[:, :, 0]

          # Compute P, p
          P = B0u.T @ grad_xx @ B0u - eps_reg*jnp.eye(self.dim_u)

          # For some reason, hessian from jax is only approxiamtely symmetric
          P = 0.5*(P + P.T)
          p = grad_x.T @ B0u
          # Controls improvement direction
          control_correction = barrier_filter_quadratic(P, p, scaled_c, initialize=solver_initial)
        elif self.constraint_type=='linear':
          control_correction = barrier_filter_linear(grad_x, B0, scaled_c)

        filtered_control = initial_control + np.array( control_correction )

        # Restart from current point and run again
        initial_control = np.array(filtered_control)

        state_imaginary, initial_control = self.dyn.integrate_forward(
            state=initial_state, control=initial_control
        )
        kwargs['state'] = np.array(state_imaginary)  
        _, solver_info_1 = self.solver_2.get_action(state_imaginary, controls=solver_info_1['controls'], **kwargs)
        solver_info_0['Vopt_next'] = solver_info_1['Vopt']
        solver_info_0['marginopt_next'] = solver_info_1['marginopt']

        initial_control_jnp = jnp.array( initial_control[:, np.newaxis] )

        # CBF constraint violation
        constraint_violation = solver_info_1['Vopt'] - cutoff
        scaled_c = kappa*constraint_violation

      if solver_info_1['Vopt']>0:
        if num_iters>0:
          self.barrier_shield_steps += 1
          solver_info_0['mark_barrier_shield'] = True
        solver_info_0['barrier_shield_steps'] = self.barrier_shield_steps
        solver_info_0['shield_steps'] = self.shield_steps
        solver_info_0['process_time'] = time.time() - start_time
        solver_info_0['resolve'] = True
        solver_info_0['reinit_controls'] = solver_info_1['controls']
        solver_info_0['reinit_states'] = solver_info_1['states']
        #solver_info_0['reinit_J'] = solver_info_1['Vopt'] 
        solver_info_0['num_iters'] = num_iters
        solver_info_0['deviation'] = np.linalg.norm(initial_control - task_policy)
        solver_info_0['qcqp_initialize'] = initial_control - task_policy
        return initial_control.ravel(), solver_info_0
    
    self.shield_steps += 1
    # Safe policy
    solver_info_0['barrier_shield_steps'] = self.barrier_shield_steps
    solver_info_0['shield_steps'] = self.shield_steps 
    solver_info_0['process_time'] = time.time() - start_time
    solver_info_0['resolve'] = True
    solver_info_0['num_iters'] = num_iters
    solver_info_0['reinit_controls'] = solver_info_0['controls']
    solver_info_0['mark_complete_shield'] = True
    solver_info_0['deviation'] = np.linalg.norm(control_0 - task_policy)
    solver_info_0['qcqp_initialize'] = control_0 - task_policy

    return control_0, solver_info_0

