from typing import Tuple, Optional, Dict
import time
import numpy as np
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray
from jax.experimental import checkify
from functools import partial

from .ilqr_policy import iLQR


class iLQRReachability(iLQR):

  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None,**kwargs
  ) -> np.ndarray:
    status = 0
    self.tol = 1e-5

    if controls is None:
      controls = np.zeros((self.dim_u, self.N))
      # Some non-zero initialization
      controls[0, :] = self.dyn.ctrl_space[0, 0]
      controls = jnp.array(controls)
    else:
      assert controls.shape[1] == self.N
      controls = jnp.array(controls)

    # Rolls out the nominal trajectory and gets the initial cost.
    states, controls = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls
    )
    
    failure_margins = self.cost.constraint.get_mapped_margin(
        states, controls
    )

    # Target cost derivatives are manually computed for more well-behaved backpropagation

    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(states, controls)
    critical, reachable__margin = self.get_critical_points(failure_margins)
        
    J = (reachable__margin + jnp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()

    for i in range(self.max_iter):
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      c_x, c_u, c_xx, c_uu, c_ux = self.cost.get_derivatives(
          states, controls
      )

      fx, fu = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1])
      V_x, V_xx, k_open_loop, K_closed_loop, _, _ = self.backward_pass(
          c_x=c_x, c_u=c_u, c_xx=c_xx, c_uu=c_uu, c_ux=c_ux, fx=fx, fu=fu,
          critical=critical
      )
      
      # Choose the best alpha scaling using appropriate line search methods
      alpha_chosen = self.baseline_line_search( states, controls, K_closed_loop, k_open_loop, J)
      #alpha_chosen = self.armijo_line_search( states, controls, K_closed_loop, k_open_loop, critical, J, c_u)

      states, controls, J_new, critical, failure_margins, reachable_margin = self.forward_pass(states, controls, K_closed_loop, 
                                                                                               k_open_loop, alpha_chosen) 
      if (np.abs((J-J_new) / J) < self.tol):  # Small improvement.
        status = 1
        if J_new>0:
          converged = True

      J = J_new
      
      if alpha_chosen<self.min_alpha:
          status = 2
          break
      # Terminates early if the objective improvement is negligible.
      if converged:
        break

    t_process = time.time() - time0
    states = np.asarray(states)
    controls = np.asarray(controls)
    K_closed_loop = np.asarray(K_closed_loop)
    k_open_loop = np.asarray(k_open_loop)
    solver_info = dict(
        states=states, controls=controls, reinit_controls=controls, t_process=t_process, status=status, Vopt=J, marginopt=reachable_margin,
        grad_x=V_x, grad_xx=V_xx, B0=fu[:, :, 0], critical=critical, is_inside_target=False,  K_closed_loop=K_closed_loop, k_open_loop=k_open_loop
    )

    return controls[:, 0], solver_info

  
  @partial(jax.jit, static_argnames='self')
  def baseline_line_search(self, states, controls, K_closed_loop, k_open_loop, J, beta=0.7):
    alpha = 1.0
    J_new = -jnp.inf

    @jax.jit
    def run_forward_pass(args):
      states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = args
      alpha = beta*alpha
      _, _, J_new, _, _, _ = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
      #_, _, J_new, _, _, _, _ = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha)
      return states, controls, K_closed_loop, k_open_loop, alpha, J, J_new

    @jax.jit
    def check_terminated(args):
      _, _, _, _, alpha, J, J_new = args
      return jnp.logical_and( alpha>self.min_alpha, J_new<J )
    
    states, controls, K_closed_loop, k_open_loop, alpha, J, J_new = jax.lax.while_loop(check_terminated, run_forward_pass, 
                                                                                       (states, controls, K_closed_loop, k_open_loop, alpha, 
                                                                                        J, J_new))

    return alpha


  @partial(jax.jit, static_argnames='self')
  def get_critical_points(
      self, failure_margins: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    # Avoid cost is critical
    @jax.jit
    def failure_func(args):
      idx, critical, failure_margin, _ = args
      critical = critical.at[idx].set(1)
      return critical, failure_margin

    # Propagating the cost is critical
    @jax.jit
    def propagate_func(args):
      idx, critical, _, reachable_margin = args
      critical = critical.at[idx].set(0)
      return critical, reachable_margin

    @jax.jit
    def critical_pt(i, _carry):
      idx = self.N - 1 - i
      critical, reachable_margin = _carry

      failure_margin = failure_margins[idx]
      critical, reachable_margin = jax.lax.cond(
          ( (failure_margin < reachable_margin) ), failure_func, propagate_func,
          (idx, critical, failure_margin, reachable_margin)
      )

      return critical, reachable_margin

    critical = jnp.zeros(shape=(self.N,), dtype=int)

    reachable_margin = failure_margins[self.N - 1]
    critical = critical.at[self.N - 1].set(1)

    critical, reachable_margin = jax.lax.fori_loop(
        1, self.N, critical_pt, (critical, reachable_margin)
    )  # backward until timestep 1

    return critical, reachable_margin

  @partial(jax.jit, static_argnames='self')
  def forward_pass(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray,
      K_closed_loop: DeviceArray, k_open_loop: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
             DeviceArray, DeviceArray]:
    X, U = self.rollout(
        nominal_states, nominal_controls, K_closed_loop, k_open_loop, alpha
    )

    # J = self.cost.get_traj_cost(X, U, closest_pt, slope, theta)
    #! hacky
    failure_margins = self.cost.constraint.get_mapped_margin(X, U)

    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(X, U)

    critical, reachable_margin = self.get_critical_points(failure_margins)

    J = (reachable_margin + jnp.sum(ctrl_costs)).astype(float)
    
    return X, U, J, critical, failure_margins, reachable_margin

  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, 
      c_x: DeviceArray, c_u: DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, fx: DeviceArray, fu: DeviceArray,
      critical: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray]:
    """
    Jitted backward pass looped computation.

    Args:
        c_x (DeviceArray): (dim_x, N)
        c_u (DeviceArray): (dim_u, N)
        c_xx (DeviceArray): (dim_x, dim_x, N)
        c_uu (DeviceArray): (dim_u, dim_u, N)
        c_ux (DeviceArray): (dim_u, dim_x, N)
        fx (DeviceArray): (dim_x, dim_x, N-1)
        fu (DeviceArray): (dim_x, dim_u, N-1)

    Returns:
        Vx (DeviceArray): gain matrices (dim_x, 1)
        Vxx (DeviceArray): gain vectors (dim_x, dim_x, 1)
        Ks (DeviceArray): gain matrices (dim_u, dim_x, N - 1)
        ks (DeviceArray): gain vectors (dim_u, N - 1)
        Vx_critical (DeviceArray): gain matrices (dim_x, 1)
        Vxx_critical (DeviceArray): gain vectors (dim_x, dim_x, 1)
    """

    @jax.jit
    def failure_backward_func(args):
      idx, V_x, V_xx, ks, Ks, _, _ = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)
 
      return c_x[:, idx], c_xx[:, :, idx], ks, Ks, c_x[:, idx], c_xx[:, :, idx]

    @jax.jit
    def propagate_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

      Q_x = fx[:, :, idx].T @ V_x
      Q_xx = fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = c_ux[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)

      V_x = Q_x + Q_ux.T @ ks[:, idx]
      V_xx = Q_xx + Q_ux.T @ Ks[:, :, idx]

      return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical

    @jax.jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical = _carry
      idx = self.N - 2 - i

      V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = jax.lax.switch(
          critical[idx], [propagate_backward_func, failure_backward_func], (idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical)
      )

      return V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical

    # Initializes.
    Ks = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
    ks = jnp.zeros((self.dim_u, self.N - 1))
    
    V_x_critical = jnp.zeros((self.dim_x, ))
    V_xx_critical = jnp.zeros((self.dim_x, self.dim_x, ))

    reg_mat = self.eps * jnp.eye(self.dim_x)

    V_x = c_x[:, self.N - 1]
    V_xx = c_xx[:, :, self.N - 1]

    V_x, V_xx, ks, Ks, _, V_x_critical, V_xx_critical = jax.lax.fori_loop(
        0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical)
    )

    return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical