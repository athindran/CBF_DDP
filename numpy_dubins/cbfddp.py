import copy
import numpy as np
import time

from policies import ReachabilityLQPolicy
from utils import unroll_task_policy, barrier_filter_quadratic

from plotting.plotting import helper_plot

def run_barrier_ilq(env, marginFunc, horizon, max_steps, gamma=0.9, barrier_scaling=1.0, policy_type="Reachability", task_policy=None, barrier_tol=-1e-5, Rc= 1e-3, barrier_type="linear", plot_folder="./", animate=False):
    """
    Barrier filtering with iLQ
    """
    plan_env_1 = copy.deepcopy( env )
    plan_env_2 = copy.deepcopy( env )
    run_env = copy.deepcopy( env )

    current_horizon = horizon
    run_env_obs = run_env.reset()

    states_ilq = np.zeros((max_steps, plan_env_1.state_dim))
    controls_ilq = np.zeros((max_steps, plan_env_2.action_dim))
    deviations_ilq = np.zeros((max_steps, plan_env_2.action_dim))
    safe_controls_ilq = np.zeros((max_steps, plan_env_1.action_dim))
    values_ilq = np.zeros((max_steps, ))
    types_ilq = np.zeros((max_steps, ))
    process_times = np.zeros((max_steps, ))
    barrier_entries_ilq = np.zeros((max_steps, ))
    barrier_margins = np.zeros((max_steps, ))
    
    complete_filter_indices = []
    barrier_filter_indices = []
    yaw_filter_indices = []
    barrier_filter_steps = 0
    complete_filter_steps = 0

    reinit_controls = None
    if policy_type=="Reachability":
        lq_policy_1 = ReachabilityLQPolicy(plan_env_1.state_dim, plan_env_1.action_dim, marginFunc, plan_env_1, horizon=current_horizon, Rc=Rc)
        lq_policy_2 = ReachabilityLQPolicy(plan_env_2.state_dim, plan_env_2.action_dim, marginFunc, plan_env_2, horizon=current_horizon, Rc=Rc)

    for time_step in range(max_steps):
        print("Observation", run_env_obs)
        
        barrier_start_time = time.time()
        
        control_safe_1, solver_dict_plan_1, constraints_data_plan_1 = lq_policy_1.get_action( np.array( run_env_obs ) , initial_controls=reinit_controls)
        
        control_perf = task_policy( plan_env_1, run_env_obs)

        safe_controls_ilq[time_step] = control_safe_1

        imag_obs, control_perf = plan_env_2.step(run_env_obs, control_perf )
        
        control_perf_init = control_perf

        boot_controls = solver_dict_plan_1['controls']
        boot_controls[0:-1] = boot_controls[1:]
        _, solver_dict_plan_2, constraints_data_plan_2 = lq_policy_2.get_action( np.array( imag_obs ) , initial_controls=boot_controls)
        boot_controls = solver_dict_plan_2['controls']

        control_cbf = control_perf

        constraint_violation = solver_dict_plan_2['complete_margin'] - gamma*(solver_dict_plan_1['complete_margin'])
        scaled_c = constraint_violation 
        
        if constraint_violation<0:
            solver_dict_plan_2['task_active'] = False
        else:
            solver_dict_plan_2['task_active'] = True

        if animate:
            solver_dict_plan_2['trace_id'] = 'Barrier'
            solver_dict_plan_2['title'] = 'Barrier filtering'
            solver_dict_plan_2['trace_label'] = "CBF-DDP"+"($\gamma$=" + str( gamma ) + ")"
            solver_dict_plan_2['previous_trace'] = states_ilq[0:time_step]
            solver_dict_plan_2['run_steps'] = solver_dict_plan_2['states'].shape[0]
            solver_dict_plan_2['task_trace'] = unroll_task_policy(run_env_obs, copy.deepcopy(plan_env_1), task_policy, horizon=current_horizon)
            solver_dict_plan_2['controls_deviation'] = solver_dict_plan_2['controls']
            helper_plot([solver_dict_plan_2], marginFunc.obstacle_center, marginFunc.obstacle_radius, save_prefix=str(time_step), save_folder= plot_folder +  "barrier_ilq_progress/", flipped=env.full_controls, xlimits=env.xlimits, ylimits=env.ylimits, ego_radius=env.ego_radius, road_boundary=env.road_boundary, animate=True)
        
        _, Bd, _, _ = plan_env_2.get_jacobian(run_env_obs, control_cbf)

        barrier_process_time = time.time() - barrier_start_time
        barrier_failures = 0
        barrier_entries = 0

        eps_regularization = 1e-8

        P = -eps_regularization*np.eye(lq_policy_1.action_dim) + Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
        p = Bd.T @ constraints_data_plan_2['V_x']
        p_norm = np.linalg.norm( p )

        while constraint_violation<barrier_tol and barrier_failures<5 and barrier_entries<7 and barrier_process_time<50:
            barrier_entries = barrier_entries + 1
            if barrier_type=="linear":
                control_correction = -p*scaled_c/((p_norm)**2 + 1e-12)
            else:
                control_correction = barrier_filter_quadratic(P, p, scaled_c)
            control_cbf_new = control_cbf + control_correction
            
            # Testing barrier quality
            imag_obs_2, control_cbf_new = plan_env_2.step( run_env_obs, control_cbf_new )
            _, solver_dict_plan_3, constraints_data_plan_3 = lq_policy_2.get_action( np.array( imag_obs_2 ) , initial_controls=boot_controls)

            solver_dict_plan_2 = solver_dict_plan_3
            constraints_data_plan_2 = constraints_data_plan_3
            control_cbf = control_cbf_new

            _, Bd, _, _  = plan_env_2.get_jacobian(run_env_obs, control_cbf)
            P = -eps_regularization*np.eye(lq_policy_1.action_dim) + Bd.T @ constraints_data_plan_2['V_xx'] @ Bd
            p = Bd.T @ constraints_data_plan_2['V_x']
            p_norm = np.linalg.norm( p )
            constraint_violation = solver_dict_plan_2['complete_margin'] - gamma*(solver_dict_plan_1['complete_margin'])
            boot_controls = solver_dict_plan_2['controls']
            scaled_c = barrier_scaling*constraint_violation  
            barrier_process_time = time.time() - barrier_start_time

        print(
            "[{}]: solver 1 returns margin {:.3f} and uses {:.3f}, solver 2 returns margin {:.3f} and uses {:.3f}.".format(time_step, solver_dict_plan_1['margin'], solver_dict_plan_1['t_process'], solver_dict_plan_2['margin'], solver_dict_plan_2['t_process'])
            )
        
        if(solver_dict_plan_2['complete_margin']<=0):
            complete_filter_indices.append( time_step )
            control = control_safe_1
            reinit_controls = solver_dict_plan_1['controls']
            complete_filter_steps = complete_filter_steps + 1
            if solver_dict_plan_1['critical_constraint_type']>1:
                    yaw_filter_indices.append( time_step )
            types_ilq[time_step] = 1
        else:
            control = control_cbf
            reinit_controls = solver_dict_plan_2['controls']
            if(barrier_entries>0):
                barrier_filter_indices.append( time_step )
                barrier_filter_steps = barrier_filter_steps + 1
                types_ilq[time_step] = 2

        run_env_obs, control_clip = run_env.step(run_env_obs, control)

        states_ilq[time_step] = np.array(run_env_obs)
        safe_controls_ilq[time_step] = control_safe_1
        controls_ilq[time_step] = np.array(control_clip)
        values_ilq[time_step] = solver_dict_plan_1['margin']
        process_times[time_step] = barrier_process_time
        barrier_entries_ilq[time_step] = barrier_entries
        barrier_margins[time_step] = constraint_violation
        deviations_ilq[time_step] = control_clip - control_perf

        print(
            "[{}]: Total barrier solver and uses {:.3f}.".format(time_step, barrier_process_time)
            )
        
        if run_env_obs.size>3 and (run_env_obs[1]>env.track_length or run_env_obs[2]<env.vmin):
            break

    solver_dict = {"states": states_ilq, "controls": controls_ilq, "safe_controls": safe_controls_ilq, "values": values_ilq, "types": types_ilq, "id": "Barrier", "task_trace": None, "task_active": False, 
                    "controls_deviation": deviations_ilq, "status": 1, "label": "CBF-DDP", "barrier_filter": barrier_filter_steps, "complete_filter": complete_filter_steps,
                    "process_times":process_times, "previous_trace": None, "title": "", "barrier_margins": barrier_margins, "barrier_entries": barrier_entries_ilq,
                    "run_steps": time_step, "complete_filter_indices": complete_filter_indices, "barrier_filter_indices": barrier_filter_indices}

    return solver_dict
