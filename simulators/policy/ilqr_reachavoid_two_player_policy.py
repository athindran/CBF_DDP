from typing import Tuple, Optional, Dict
import time
import numpy as np
import jax
from jax import numpy as jnp
from jaxlib.xla_extension import DeviceArray
#from jax.experimental import checkify
from functools import partial

from .ilqr_policy import iLQR


class iLQRReachAvoidGame(iLQR):

  @partial(jax.jit, static_argnames='self')
  def rollout_nominal(
      self, initial_state: DeviceArray, controls: DeviceArray, disturbances: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    @jax.jit
    def _rollout_nominal_step(i, args):
      X, U, D = args
      x_nxt, u_clip, d_clip = self.dyn.integrate_forward_jax(X[:, i], U[:, i], D[:, i])
      X = X.at[:, i + 1].set(x_nxt)
      U = U.at[:, i].set(u_clip)
      D = D.at[:, i].set(d_clip)
      return X, U, D

    X = jnp.zeros((self.dim_x, self.N))
    X = X.at[:, 0].set(initial_state)
    X, U, D = jax.lax.fori_loop(
        0, self.N - 1, _rollout_nominal_step, (X, controls, disturbances)
    )
    return X, U, D

  @partial(jax.jit, static_argnames='self')
  def rollout(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray, nominal_disturbances: DeviceArray,
      Ks1: DeviceArray, ks1: DeviceArray, Ks2: DeviceArray, ks2: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray]:

    @jax.jit
    def _rollout_step(i, args):
      X, U, D = args
      u_fb = jnp.einsum(
          "ik,k->i", Ks1[:, :, i], (X[:, i] - nominal_states[:, i])
      )
      u = nominal_controls[:, i] + alpha * ks1[:, i] + u_fb
      d_fb = jnp.einsum(
          "ik,k->i", Ks2[:, :, i], (X[:, i] - nominal_states[:, i])
      )
      d = nominal_disturbances[:, i] + alpha * ks2[:, i] + d_fb
      #d = jnp.array([0, 0])

      x_nxt, u_clip, d_clip = self.dyn.integrate_forward_jax(X[:, i], u, d)
      X = X.at[:, i + 1].set(x_nxt)
      U = U.at[:, i].set(u_clip)
      D = D.at[:, i].set(d_clip)
      return X, U, D

    X = jnp.zeros((self.dim_x, self.N))
    U = jnp.zeros((self.dim_u, self.N))  #  Assumes the last ctrl are zeros.
    D = jnp.zeros((self.dim_d, self.N))  #  Assumes the last ctrl are zeros.
    X = X.at[:, 0].set(nominal_states[:, 0])

    X, U, D = jax.lax.fori_loop(0, self.N, _rollout_step, (X, U, D))
    return X, U, D
  
  def get_action(
      self, obs: np.ndarray, controls: Optional[np.ndarray] = None,
      agents_action: Optional[Dict] = None,**kwargs
  ) -> np.ndarray:
    status = 0
    self.tol = 1e-4

    if controls is None:
      controls = np.zeros((self.dim_u, self.N))
      #controls = np.random.rand(self.dim_u, self.N)
      # Some non-zero initialization
      controls[0, :] = self.dyn.ctrl_space[0, 0]
      controls = jnp.array(controls)
    else:
      assert controls.shape[1] == self.N
      controls = jnp.array(controls)
    
    disturbances = np.zeros((self.dim_d, self.N))
    disturbances[0, :] = 0.01
    disturbances[1, :] = 0.01
    disturbances = jnp.array(disturbances)

    # Rolls out the nominal trajectory and gets the initial cost.
    states, controls, disturbances = self.rollout_nominal(
        jnp.array(kwargs.get('state')), controls, disturbances
    )
    
    failure_margins = self.cost.constraint.get_mapped_margin(
        states, controls, disturbances
    )

    # Target cost derivatives are manually computed for more well-behaved backpropagation
    target_margins = self.cost.get_mapped_target_margin(states, controls, disturbances)
    #target_margins, c_x_t, c_xx_t, c_u_t, c_uu_t = self.cost.get_mapped_target_margin_with_derivative(states, controls)

    is_inside_target = (target_margins[0]>0)
    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(states, controls, disturbances)
    critical, reachavoid__margin = self.get_critical_points(failure_margins, target_margins)
        
    J = (reachavoid__margin + jnp.sum(ctrl_costs)).astype(float)

    converged = False
    time0 = time.time()
    
    alpha_chosen = 1
    for i in range(self.max_iter):
      self.plotter.plot_cvg_animation(states, critical=np.array(critical), fig_name=str(i))
      # We need cost derivatives from 0 to N-1, but we only need dynamics
      c_x, c_u, c_d, c_xx, c_uu, c_dd, c_ux = self.cost.get_derivatives(
          states, controls, disturbances
      )

      c_x_t, c_u_t, c_d_t, c_xx_t, c_uu_t, c_dd_t, c_ux_t = self.cost.get_derivatives_target(
          states, controls, disturbances
      )

      fx, fu ,fd = self.dyn.get_jacobian(states[:, :-1], controls[:, :-1], disturbances[:, :-1])

      V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = self.backward_pass(
        c_x=c_x, c_u=c_u, c_d=c_d, c_xx=c_xx,
        c_uu=c_uu, c_ux=c_ux, c_x_t=c_x_t, c_u_t=c_u_t, c_xx_t=c_xx_t,
        c_uu_t=c_uu_t, c_ux_t=None, c_dd=c_dd, fx=fx, fu=fu, fd=fd, critical=critical
      )
      
      # Choose the best alpha scaling using appropriate line search methods
      #alpha_chosen, co , co_new = self.baseline_line_search( states, controls, disturbances, Ks1, ks1, Ks2, ks2, J, alpha_init=1.0)
      #alpha_chosen, co , co_new = self.armijo_line_search( states, controls, disturbances, Ks1, ks1, Ks2, ks2, J,
      #                                                    critical=critical, c_u=c_u, alpha_init=1.0)
      alpha_chosen, co , co_new = self.trust_region_search_conservative( states, controls, disturbances, Ks1, ks1, Ks2, ks2, J,
                                                          critical=critical, c_x=c_x, c_xx=c_xx, 
                                                          alpha_init=1.0)
  
      print(co, co_new)
      #alpha_chosen = self.armijo_line_search( states, controls, K_closed_loop, k_open_loop, critical, J, c_u)

      states, controls, disturbances, J_new, critical, failure_margins, target_margins, reachavoid_margin = self.forward_pass(states, controls, disturbances, 
                                                                                                                Ks1, ks1, Ks2, ks2, alpha_chosen)        
      print(J, J_new, alpha_chosen, self.min_alpha)
      #states, controls, J_new, critical, failure_margins, target_margins, reachavoid_margin, c_x_t, c_xx_t, c_u_t, c_uu_t = self.forward_pass(states, controls, K_closed_loop, k_open_loop, alpha_chosen) 
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


    self.plotter.make_animation(i + 1)
    t_process = time.time() - time0
    states = np.asarray(states)
    controls = np.asarray(controls)
    disturbances = np.asarray(disturbances)
    Ks1 = np.asarray(Ks1)
    ks1 = np.asarray(ks1)
    ks2 = np.asarray(ks2)
    solver_info = dict(
        states=states, controls=controls, disturbances=disturbances, reinit_controls=controls, t_process=t_process, status=status, Vopt=J, marginopt=reachavoid_margin,
        grad_x=V_x, grad_xx=V_xx, B0=fu[:, :, 0], critical=critical, is_inside_target=is_inside_target, Ks1=Ks1, ks1=ks1, Ks2=Ks2, ks2=ks2
    )

    return controls[:, 0], solver_info

  
  @partial(jax.jit, static_argnames='self')
  def baseline_line_search(self, states, controls, disturbances, Ks1, ks1, Ks2, ks2, J, alpha_init=1, beta=0.7):
    alpha = alpha_init
    J_new = -jnp.inf

    @jax.jit
    def run_forward_pass(args):
      states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, J_new= args
      alpha = beta*alpha
      _, _, _, J_new, _, _, _, _ = self.forward_pass(nominal_states=states, nominal_controls=controls, nominal_disturbances=disturbances, 
                                                     Ks1=Ks1, ks1=ks1, Ks2=Ks2, ks2=ks2, alpha=alpha)
      return states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, J_new

    @jax.jit
    def check_terminated(args):
      _, _, _, _, _, _, _, alpha, J, J_new = args
      return jnp.logical_and( alpha>self.min_alpha, J_new - J<0 )
    
    states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, J_new= jax.lax.while_loop(check_terminated, run_forward_pass, (states, controls, disturbances,
                                                                                                                            Ks1, ks1, Ks2, ks2, alpha, J, J_new))
    return alpha, J, J_new

  @partial(jax.jit, static_argnames='self')
  def armijo_line_search( self, states, controls, disturbances, Ks1, ks1, Ks2, ks2, J, critical, c_u, alpha_init=1.0, beta=0.7):
    alpha = 1.0
    J_new = -jnp.inf
    deltat = 0
    t_star = jnp.where(critical!=0, size=self.N-1)[0][0]
    
    @jax.jit
    def run_forward_pass(args):
      states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, deltat= args
      alpha = beta*alpha
      X, _, _, J_new, _, _, _, _ = self.forward_pass(nominal_states=states, nominal_controls=controls, nominal_disturbances=disturbances, 
                                                     Ks1=Ks1, ks1=ks1, Ks2=Ks2, ks2=ks2, alpha=alpha)      
      # Calculate gradient for armijo decrease condition
      delta_u =  Ks1[:, :, t_star] @ (X[:, t_star]  - states[:, t_star]) + ks1[:, t_star]

      grad_cost_u = c_u[:, t_star]

      deltat = grad_cost_u @ delta_u

      return  states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, deltat

    @jax.jit
    def check_terminated(args):
      _, _, _, _, _, _, _, alpha, J, _, J_new, deltat = args
      armijo_check = (J_new <=J + deltat*alpha )
      return jnp.logical_and( alpha>self.min_alpha, armijo_check )
    
    states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, deltat = jax.lax.while_loop(check_terminated, run_forward_pass, (states, controls, disturbances,
                                                                                                                            Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, deltat))
    return alpha, J, J_new
  
  @partial(jax.jit, static_argnames='self')
  def trust_region_search_conservative( self, states, controls, disturbances, Ks1, ks1, Ks2, ks2, J, critical, 
                         c_x, c_xx, alpha_init=1.0, beta=0.7):
    alpha = alpha_init
    J_new = -jnp.inf
    t_star = jnp.where(critical!=0, size=self.N-1)[0][0]

    self.margin = 2
    traj_diff = 0.2
    cost_error = 1.0
    old_cost_error = 2.0
    rho = 0.5

    @jax.jit
    def decrease_margin(args):
      self.margin = 0.75*self.margin
      return args
    
    @jax.jit
    def increase_margin(args):
      self.margin = 1.4*self.margin
      return args

    @jax.jit
    def fix_margin(args):
      return args
    
    @jax.jit
    def increase_or_fix_margin(args):
      cost_error, old_cost_error, traj_diff, rho = args
      _, _ = jax.lax.cond(jnp.logical_and( jnp.abs(traj_diff - self.margin)<0.01, rho>0.75), increase_margin, 
                                                fix_margin, (cost_error, old_cost_error))     
      return cost_error, old_cost_error, traj_diff, rho

    @jax.jit
    def run_forward_pass(args):
      states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, traj_diff, cost_error, old_cost_error, rho = args
      alpha = beta*alpha
      X, _, _, J_new, _, _, _, _ = self.forward_pass(nominal_states=states, nominal_controls=controls, nominal_disturbances=disturbances, 
                                                     Ks1=Ks1, ks1=ks1, Ks2=Ks2, ks2=ks2, alpha=alpha)      

      traj_diff = jnp.max(jnp.array([jnp.linalg.norm( x_new - x_old )
                          for x_new, x_old in zip(X[:2, :], states[:2, :])]))
      
      x_diff = X[:, t_star] - states[:, t_star]

      delta_cost_quadratic_approx = 0.5* (x_diff @ c_xx[:, :, t_star] + 2*c_x[:, t_star]) @ x_diff
      delta_cost_actual = J_new - J
      old_cost_error = jnp.abs(cost_error)
      cost_error = jnp.abs( delta_cost_quadratic_approx - delta_cost_actual )
      rho = delta_cost_actual/delta_cost_quadratic_approx
      return  states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, traj_diff, cost_error, old_cost_error, rho

    @jax.jit
    def check_continue(args):
      _, _, _, _, _, _, _, alpha, J, _, J_new, traj_diff, cost_error, old_cost_error, rho = args
      trust_region_violation = ( traj_diff>self.margin )
      improvement_violation = (J_new < J)

      cost_error, old_cost_error, traj_diff, rho = jax.lax.cond((rho<=0.25), decrease_margin, 
                                                increase_or_fix_margin, (cost_error, old_cost_error, traj_diff, rho))      
      return jnp.logical_and( alpha>self.min_alpha, jnp.logical_or(trust_region_violation , improvement_violation))
    
    states, controls, disturbances, Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, traj_diff, cost_error, _, _ = (
      jax.lax.while_loop(check_continue, run_forward_pass, (states, controls, disturbances,
                                                          Ks1, ks1, Ks2, ks2, alpha, J, t_star, J_new, 
                                                          traj_diff, cost_error, old_cost_error, rho)))
    print("Margin", self.margin)
    return alpha, J, J_new

  @partial(jax.jit, static_argnames='self')
  def get_critical_points(
      self, failure_margins: DeviceArray, target_margins:DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray]:

    # Avoid cost is critical
    @jax.jit
    def failure_func(args):
      idx, critical, failure_margin, _, _ = args
      critical = critical.at[idx].set(1)
      return critical, failure_margin

    # Avoid cost is not critical
    @jax.jit
    def target_propagate_func(args):
      _, _, _, target_margin, reachavoid_margin = args
      return jax.lax.cond(target_margin > reachavoid_margin, target_func, propagate_func, args)  

    # Reach cost is critical
    @jax.jit
    def target_func(args):
      idx, critical, _, target_margin, _ = args
      critical = critical.at[idx].set(2)
      return critical, target_margin

    # Propagating the cost is critical
    @jax.jit
    def propagate_func(args):
      idx, critical, _, _, reachavoid_margin = args
      critical = critical.at[idx].set(0)
      return critical, reachavoid_margin

    @jax.jit
    def critical_pt(i, _carry):
      idx = self.N - 1 - i
      critical, reachavoid_margin = _carry

      failure_margin = failure_margins[idx]
      target_margin = target_margins[idx]
      critical, reachavoid_margin = jax.lax.cond(
          ( (failure_margin < reachavoid_margin) | (failure_margin < target_margin) ), failure_func, target_propagate_func,
          (idx, critical, failure_margin, target_margin, reachavoid_margin)
      )

      return critical, reachavoid_margin

    critical = jnp.zeros(shape=(self.N,), dtype=int)

    reachavoid_margin = 0.
    critical, reachavoid_margin = jax.lax.cond(target_margins[self.N - 1] < failure_margins[self.N - 1], target_func, failure_func, 
                  (self.N - 1, critical, failure_margins[self.N - 1], target_margins[self.N - 1], reachavoid_margin)
    )

    critical, reachavoid_margin = jax.lax.fori_loop(
        1, self.N, critical_pt, (critical, reachavoid_margin)
    )  # backward until timestep 1

    return critical, reachavoid_margin

  @partial(jax.jit, static_argnames='self')
  def forward_pass(
      self, nominal_states: DeviceArray, nominal_controls: DeviceArray, nominal_disturbances: DeviceArray,
      Ks1: DeviceArray, ks1: DeviceArray, Ks2: DeviceArray, ks2: DeviceArray, alpha: float
  ) -> Tuple[DeviceArray, DeviceArray, float, DeviceArray, DeviceArray,
             DeviceArray, DeviceArray]:
    X, U, D = self.rollout(
        nominal_states, nominal_controls, nominal_disturbances, Ks1, ks1, Ks2, ks2, alpha
    )

    # J = self.cost.get_traj_cost(X, U, closest_pt, slope, theta)
    #! hacky
    failure_margins = self.cost.constraint.get_mapped_margin(X, U, D)
    target_margins = self.cost.get_mapped_target_margin(X, U, D)
    #target_margins, c_x_t, c_xx_t, c_u_t, c_uu_t = self.cost.get_mapped_target_margin_with_derivative(X, U)

    ctrl_costs = self.cost.ctrl_cost.get_mapped_margin(X, U, D)
    dist_costs = self.cost.dist_cost.get_mapped_margin(X, U, D)

    critical, reachavoid_margin = self.get_critical_points(failure_margins, target_margins)

    J = (reachavoid_margin + jnp.sum(ctrl_costs) - jnp.sum(dist_costs)).astype(float)
    
    #reachavoid_margin_comp = self.brute_force_critical_cost(np.array(failure_margins), np.array(target_margins))
    #assert jnp.abs(reachavoid_margin - reachavoid_margin_comp)==0
    #checked_margin = checkify.checkify(self.brute_force_critical_cost)
    #err, future_cost = checked_margin(failure_margins, target_margins, reachavoid_margin)
    #err.throw()

    return X, U, D, J, critical, failure_margins, target_margins, reachavoid_margin
    #, c_x_t, c_xx_t, c_u_t, c_uu_t

  """
  def brute_force_critical_cost(self, state_costs, target_costs):
    critical_cost_arr = np.zeros((self.N, ))
    for iters in range(self.N):
      min_avoid_cost = np.min(state_costs[0:iters+1])
      critical_cost_arr[iters] = np.minimum(min_avoid_cost, target_costs[iters])
    return np.max(critical_cost_arr)

  #@partial(jax.jit, static_argnames='self')
  def brute_force_critical_cost(self, failure_margins, target_margins):
    critical_cost_arr = jnp.zeros((self.N, ))

    def forward_looper(idx, args):
      critical_cost_arr, min_failure_margin = args
      min_failure_margin = jnp.minimum(min_failure_margin, failure_margins[idx])
      critical_cost_arr = critical_cost_arr.at[idx].set(jnp.minimum(min_failure_margin, target_margins[idx]))
      return critical_cost_arr, min_failure_margin
    
    critical_cost_arr, _ = jax.lax.fori_loop(0, self.N, forward_looper, (critical_cost_arr, failure_margins[0]))
    future_cost = jnp.max(critical_cost_arr)
    #checkify.check( future_cost - compare_cost==0, "Reach-avoid cost error")

    return future_cost
  """
  """
  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, 
      c_x: DeviceArray, c_u: DeviceArray, c_d:DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, c_x_t: DeviceArray, c_u_t: DeviceArray, c_xx_t: DeviceArray,
      c_uu_t: DeviceArray, c_ux_t: DeviceArray, c_dd: DeviceArray, fx: DeviceArray, fu: DeviceArray, fd:DeviceArray,
      critical: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray]:

    @jax.jit
    def failure_backward_func(args):
      idx, V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = args
      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_uu_11 = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx] 
      Q_uu_22 = c_dd[:, :, idx] + fd[:, :, idx].T @ (-V_xx + reg_mat) @ fd[:, :, idx] 
      Q_uu_12 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fd[:, :, idx]
      Q_uu_21 = -Q_uu_12.T
      Q_uu_1 = jnp.concatenate((Q_uu_11, Q_uu_12), axis=1)
      Q_uu_2 = jnp.concatenate((Q_uu_21, Q_uu_22), axis=1)
      Q_uu = jnp.concatenate( (Q_uu_1, Q_uu_2), axis=0)
      Q_ux_1 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux_2 = fd[:, :, idx].T @ (-V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux = jnp.concatenate((Q_ux_1, Q_ux_2), axis=0)

      Q_uu_inv = jnp.linalg.inv(Q_uu) 
      Ks = Q_uu_inv @ Q_ux
      Ks1i, Ks2i = jnp.split(Ks, 2, axis=0)

      Ks1 = Ks1.at[:, :, idx].set(-Ks1i)
      Ks2 = Ks2.at[:, :, idx].set(-Ks2i) 
      #f_cl = fx[:, :, idx] - fu[:, :, idx]@Ks1 - fd[:, :, idx]@Ks2

      V_x = c_x[:, idx]
      V_xx = c_xx[:, :, idx]

      Q_u_1 = fu[:, :, idx].T @ V_x
      Q_u_2 = - fd[:, :, idx].T @ V_x
      Q_u = jnp.concatenate((Q_u_1, Q_u_2), axis=0)
      ks = Q_uu_inv @ Q_u
      ks1i, ks2i = jnp.split(ks, 2, axis=0)
      
      ks1 = ks1.at[:, idx].set(-ks1i)
      ks2 = ks2.at[:, idx].set(-ks2i)
      beta = - fu[:, :, idx]@ks1i - fd[:, :, idx]@ks2i
      
      n1 = n1.at[idx].set(0.5*(ks1i.T@c_uu[:, :, idx] - 2*c_u[:, idx])@ks1i - (2*V_x - V_xx@(-beta)).T@(-beta) + n1[idx])
      n2 = n2.at[idx].set(0.5*(ks2i.T@c_dd[:, :, idx] - 2*c_d[:, idx])@ks2i + (2*V_x - V_xx@(-beta)).T@(-beta) + n2[idx])
 
      return V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical

    @jax.jit
    def target_backward_func(args):
      idx, V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_uu_11 = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]
      Q_uu_22 = c_dd[:, :, idx] + fd[:, :, idx].T @ (-V_xx + reg_mat) @ fd[:, :, idx]
      Q_uu_12 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fd[:, :, idx]
      Q_uu_21 = -Q_uu_12.T
      Q_uu_1 = jnp.concatenate((Q_uu_11, Q_uu_12), axis=1)
      Q_uu_2 = jnp.concatenate((Q_uu_21, Q_uu_22), axis=1)
      Q_uu = jnp.concatenate( (Q_uu_1, Q_uu_2), axis=0)
      Q_ux_1 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux_2 = fd[:, :, idx].T @ (-V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux = jnp.concatenate((Q_ux_1, Q_ux_2), axis=0)

      Q_uu_inv = jnp.linalg.inv(Q_uu) 
      Ks = Q_uu_inv @ Q_ux
      Ks1i, Ks2i = jnp.split(Ks, 2, axis=0)

      Ks1 = Ks1.at[:, :, idx].set(-Ks1i)
      Ks2 = Ks2.at[:, :, idx].set(-Ks2i) 

      #f_cl = fx[:, :, idx] - fu[:, :, idx]@Ks1 - fd[:, :, idx]@Ks2

      V_x = c_x_t[:, idx]
      V_xx = c_xx_t[:, :, idx]

      Q_u_1 = fu[:, :, idx].T @ V_x
      Q_u_2 = - fd[:, :, idx].T @ V_x
      Q_u = jnp.concatenate((Q_u_1, Q_u_2), axis=0)
      ks = Q_uu_inv @ Q_u
      ks1i, ks2i = jnp.split(ks, 2, axis=0)
      
      ks1 = ks1.at[:, idx].set(-ks1i)
      ks2 = ks2.at[:, idx].set(-ks2i)
      beta = - fu[:, :, idx]@ks1i - fd[:, :, idx]@ks2i
      
      n1 = n1.at[idx].set(0.5*(ks1i.T@c_uu[:, :, idx] - 2*c_u[:, idx])@ks1i - (2*V_x - V_xx@(-beta)).T@(-beta) + n1[idx])
      n2 = n2.at[idx].set(0.5*(ks2i.T@c_dd[:, :, idx] - 2*c_d[:, idx])@ks2i + (2*V_x - V_xx@(-beta)).T@(-beta) + n2[idx])

      return V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical

    @jax.jit
    def propagate_backward_func(args):
      idx, V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = args

      Q_uu_11 = c_uu[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]
      Q_uu_22 = c_dd[:, :, idx] + fd[:, :, idx].T @ (-V_xx + reg_mat) @ fd[:, :, idx]
      Q_uu_12 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fd[:, :, idx]
      Q_uu_21 = -Q_uu_12.T
      Q_uu_1 = jnp.concatenate((Q_uu_11, Q_uu_12), axis=1)
      Q_uu_2 = jnp.concatenate((Q_uu_21, Q_uu_22), axis=1)
      Q_uu = jnp.concatenate( (Q_uu_1, Q_uu_2), axis=0)
      Q_ux_1 = fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux_2 = fd[:, :, idx].T @ (-V_xx + reg_mat) @ fx[:, :, idx]
      Q_ux = jnp.concatenate((Q_ux_1, Q_ux_2), axis=0)

      Q_uu_inv = jnp.linalg.inv(Q_uu) 
      Ks = Q_uu_inv @ Q_ux
      Ks1i, Ks2i = jnp.split(Ks, 2, axis=0)

      Ks1 = Ks1.at[:, :, idx].set(-Ks1i)
      Ks2 = Ks2.at[:, :, idx].set(-Ks2i) 

      f_cl = fx[:, :, idx] - fu[:, :, idx]@Ks1i - fd[:, :, idx]@Ks2i

      V_xx = f_cl.T @V_xx @f_cl + Ks1i.T @ c_uu[:, :, idx] @ Ks1i + Ks2i.T @ c_dd[:, :, idx] @ Ks2i 

      Q_u_1 = fu[:, :, idx].T @ V_x
      Q_u_2 = - fd[:, :, idx].T @ V_x
      Q_u = jnp.concatenate((Q_u_1, Q_u_2), axis=0)
      ks = Q_uu_inv @ Q_u
      ks1i, ks2i = jnp.split(ks, 2, axis=0)
      
      ks1 = ks1.at[:, idx].set(-ks1i)
      ks2 = ks2.at[:, idx].set(-ks2i)
      beta = - fu[:, :, idx]@ks1i - fd[:, :, idx]@ks2i
      
      V_x = f_cl.T @ (V_x + V_xx@beta) + Ks1i.T@c_uu[:, :, idx]@ks1i - Ks1i.T@c_u[:, idx] + Ks2i.T@c_dd[:, :, idx]@ks2i - Ks2i.T@c_d[:, idx]
      n1 = n1.at[idx].set(0.5*(ks1i.T@c_uu[:, :, idx] - 2*c_u[:, idx])@ks1i - (2*V_x - V_xx@(-beta)).T@(-beta) + n1[idx])
      n2 = n2.at[idx].set(0.5*(ks2i.T@c_dd[:, :, idx] - 2*c_d[:, idx])@ks2i + (2*V_x - V_xx@(-beta)).T@(-beta) + n2[idx])
      
      return V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical

    @jax.jit
    def backward_pass_looper(i, _carry):
      V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = _carry
      idx = self.N - 2 - i

      V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = jax.lax.switch(
          critical[idx], [propagate_backward_func, failure_backward_func, target_backward_func], 
          (idx, V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical)
      )

      return V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical

    @jax.jit
    def failure_final_func(args):
      c_x, c_xx, _, _ = args
      return c_x[:, self.N - 1], c_xx[:, :, self.N - 1]

    @jax.jit
    def target_final_func(args):
      _, _, c_x_t, c_xx_t = args
      return c_x_t[:, self.N - 1], c_xx_t[:, :, self.N - 1]

    # Initializes.
    Ks1 = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
    ks1 = jnp.zeros((self.dim_u, self.N - 1))
    n1 = jnp.zeros((self.N - 1, ))
    
    Ks2 = jnp.zeros((self.dim_d, self.dim_x, self.N - 1))
    ks2 = jnp.zeros((self.dim_d, self.N - 1))
    n2 = jnp.zeros((self.N - 1, ))

    reg_mat = 1e-4 * jnp.eye(self.dim_x)

    # If critical is 2 choose target - hacky!!
    V_x, V_xx = jax.lax.cond(critical[self.N -1]==1, failure_final_func, target_final_func, (c_x, c_xx, c_x_t, c_xx_t))

    V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical = jax.lax.fori_loop(
        0, self.N - 1, backward_pass_looper, (V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical)
    )

    return V_x, V_xx, n1, ks1, Ks1, n2, ks2, Ks2, critical
  """
  
  @partial(jax.jit, static_argnames='self')
  def backward_pass(
      self, 
      c_x: DeviceArray, c_u: DeviceArray, c_d:DeviceArray, c_xx: DeviceArray,
      c_uu: DeviceArray, c_ux: DeviceArray, c_x_t: DeviceArray, c_u_t: DeviceArray, c_xx_t: DeviceArray,
      c_uu_t: DeviceArray, c_ux_t: DeviceArray, c_dd: DeviceArray, fx: DeviceArray, fu: DeviceArray, fd:DeviceArray,
      critical: DeviceArray
  ) -> Tuple[DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray, DeviceArray]:

    @jax.jit
    def failure_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

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
    def target_backward_func(args):
      idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical = args

      #! Q_x, Q_xx are not used if this time step is critical.
      # Q_x = c_x[:, idx] + fx[:, :, idx].T @ V_x
      # Q_xx = c_xx[:, :, idx] + fx[:, :, idx].T @ V_xx @ fx[:, :, idx]
      Q_ux = fu[:, :, idx].T @ (V_xx + reg_mat) @ fx[:, :, idx]
      Q_u = c_u_t[:, idx] + fu[:, :, idx].T @ V_x
      Q_uu = c_uu_t[:, :, idx] + fu[:, :, idx].T @ (V_xx + reg_mat) @ fu[:, :, idx]

      Q_uu_inv = jnp.linalg.inv(Q_uu)
      Ks = Ks.at[:, :, idx].set(-Q_uu_inv @ Q_ux)
      ks = ks.at[:, idx].set(-Q_uu_inv @ Q_u)
 
      return c_x_t[:, idx], c_xx_t[:, :, idx], ks, Ks, c_x_t[:, idx], c_xx_t[:, :, idx]

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
          critical[idx], [propagate_backward_func, failure_backward_func, target_backward_func], 
          (idx, V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical)
      )

      return V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical

    @jax.jit
    def failure_final_func(args):
      c_x, c_xx, _, _ = args
      return c_x[:, self.N - 1], c_xx[:, :, self.N - 1]

    @jax.jit
    def target_final_func(args):
      _, _, c_x_t, c_xx_t = args
      return c_x_t[:, self.N - 1], c_xx_t[:, :, self.N - 1]

    # Initializes.
    Ks = jnp.zeros((self.dim_u, self.dim_x, self.N - 1))
    ks = jnp.zeros((self.dim_u, self.N - 1))
    
    n1 = jnp.zeros((self.N - 1, ))
    
    Ks2 = jnp.zeros((self.dim_d, self.dim_x, self.N - 1))
    ks2 = jnp.zeros((self.dim_d, self.N - 1))
    n2 = jnp.zeros((self.N - 1, ))

    V_x_critical = jnp.zeros((self.dim_x, ))
    V_xx_critical = jnp.zeros((self.dim_x, self.dim_x, ))

    reg_mat = self.eps * jnp.eye(self.dim_x)

    # If critical is 2 choose target - hacky!!
    V_x, V_xx = jax.lax.cond(critical[self.N -1]==1, failure_final_func, target_final_func, (c_x, c_xx, c_x_t, c_xx_t))

    V_x, V_xx, ks, Ks, _, V_x_critical, V_xx_critical = jax.lax.fori_loop(
        0, self.N - 1, backward_pass_looper, (V_x, V_xx, ks, Ks, critical, V_x_critical, V_xx_critical)
    )

    return V_x, V_xx, n1, ks, Ks, n2, ks2, Ks2, critical
    #return V_x, V_xx, ks, Ks, V_x_critical, V_xx_critical