from simulators import(load_config, CarSingle5DEnv)

from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib
import numpy as np

import jax
import os

def find_jerk(controls):
    x_jerk = np.abs(controls[1:, 0] - controls[0:-1, 0])
    y_jerk = np.abs(controls[1:, 1] - controls[0:-1, 1])
    mean_x_jerk = np.mean( x_jerk )
    mean_y_jerk = np.mean( y_jerk )
    std_x_jerk = np.std( x_jerk )
    std_y_jerk = np.std( y_jerk )
    return [mean_x_jerk, mean_y_jerk, std_x_jerk, std_y_jerk]

def plot_run_summary(dyn_id, env, state_history, action_history, config_solver, fig_folder="./", **kwargs):
    c_obs = 'k'
    c_ego = 'c'
    
    fig, axes = plt.subplots(
        2, 1, figsize=(config_solver.FIG_SIZE_X, 2*config_solver.FIG_SIZE_Y)
    )

    for ax in axes:
      # track, obstacles, footprint
      env.render_obs(ax=ax, c=c_obs)
      env.render_footprint(ax=ax, state=state_history[-3], c=c_ego)
      ax.axis(env.visual_extent)
      ax.set_aspect('equal')

    colors = {}
    colors['LR'] = 'r'
    colors['CBF'] = 'b'
    states = np.array(state_history).T
    ctrls = np.array(action_history).T

    ax = axes[0]
    sc = ax.scatter(
        states[0, :-1], states[1, :-1], s=12, c=states[2, :-1], cmap=cm.jet,
        vmin=0, vmax=1.5, edgecolor='none', marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"velocity [$m/s$]", size=20)

    ax = axes[1]
    sc = ax.scatter(
        states[0, :-1], states[1, :-1], s=12, c=ctrls[1, :], cmap=cm.jet,
        vmin=-0.8, vmax=0.8, edgecolor='none', marker='o'
    )
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label(r"second ctrl", size=20)
    fig.tight_layout()
    fig.savefig(os.path.join(fig_folder, "final.png"), dpi=200)
    plt.close('all')
    
    if dyn_id=="Bicycle5D":
      fig, axes = plt.subplots(
        1, 3, figsize=(15.0, 3.0)
      )
      ax = axes[0]
      ax.plot(kwargs["value_history"])
      ax.set_xlabel("Timestep")
      ax.set_ylabel("Receding Value function")
      ax.grid()
    
      ax = axes[1]
      ax.plot(ctrls[0, :])
      ax.set_xlabel("Timestep")
      ax.set_ylabel("Acceleration control 1")
      ax.grid()

      ax = axes[2]
      ax.plot(ctrls[1, :])
      ax.set_xlabel("Timestep")
      ax.set_ylabel("Steering control 1")
      ax.grid()

      fig.savefig(os.path.join(fig_folder, "auxiliary_controls.png"), dpi=200)

    fig, axes = plt.subplots(
        1, 2, figsize=(8.0, 2.5)
    )
    ax = axes[0]
    ax.plot(states[2, :])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Velocity")
    ax.grid()
    
    ax = axes[1]
    ax.plot(states[4, :])
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Delta")
    ax.grid()
    fig.savefig(os.path.join(fig_folder, "auxiliary_velocity.png"), dpi=200)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(kwargs["process_time_history"])
    plt.ylabel('Solver process time (s)')
    plt.xlabel('Time step')
    fig.savefig(os.path.join(fig_folder, "auxiliary_cycletimes.png"), dpi=200)

    fig = plt.figure(figsize=(7, 4))
    plt.plot(kwargs["solver_iters_history"])
    plt.ylabel('Solver iterations')
    plt.xlabel('Time step')
    fig.savefig(os.path.join(fig_folder, "auxiliary_cbfiters.png"), dpi=200)

def make_animation_plots(env, state_history, solver_info, config_solver, fig_prog_folder="./"):
    fig, ax = plt.subplots(
        1, 1, figsize=(config_solver.FIG_SIZE_X, config_solver.FIG_SIZE_Y)
    )
    states = np.array(state_history).T

    ax.axis(env.visual_extent)
    ax.set_aspect('equal')

    c_obs = 'k'
    c_ego = 'c'

    if config_solver.FILTER_TYPE == "none":
        c_trace = 'k'
    elif config_solver.FILTER_TYPE == "LR":
        c_trace = 'r'
        ax.set_title('LR-DDP')
    elif config_solver.FILTER_TYPE == "CBF":
        c_trace = 'b'
        ax.set_title('CBF-DDP')

    # track, obstacles, footprint
    env.render_obs(ax=ax, c=c_obs)

    if solver_info['mark_complete_filter']:
        env.render_footprint(ax=ax, state=state_history[-1], c='r', lw=0.5)
    elif solver_info['mark_barrier_filter']:
        env.render_footprint(ax=ax, state=state_history[-1], c='b', lw=0.5)
    else:    
        env.render_footprint(ax=ax, state=state_history[-1], c=c_ego, lw=0.5)

    # plan.
    ax.plot(
        solver_info['states'][0, :], solver_info['states'][1, :], linewidth=0.5,
        c='g', label='Safety plan'
    )
    # historyory.
    sc = ax.scatter(
        states[0, :-1], states[1, :-1], s=24, c=c_trace, marker='o'
    )
    ax.legend(fontsize=6, loc='upper left', bbox_to_anchor=(-0.05, 1.14), fancybox=True)    
    fig.savefig(
        os.path.join(fig_prog_folder,
                     str(states.shape[1] - 1) + ".png"), dpi=200
    )
    plt.close('all')

def make_yaw_report(prefix="./exps_may/ilqr/bic5D/yaw_testing/", plot_folder="./plots_paper/", tag="reachavoid", road_boundary=1.2):
    if not os.path.exists(plot_folder):
        os.makedirs(plot_folder)

    matplotlib.rc('xtick', labelsize=6) 
    matplotlib.rc('ytick', labelsize=6) 
    legend_fontsize = 5
    road_bounds = [road_boundary]
    yaw_consts = [None, 0.5*np.pi, 0.4*np.pi]
    label_yc = [None, 0.5, 0.4]

    suffixlist = []
    labellist = []
    colorlist = []
    stylelist = []
    rblist = []
    showlist = []
    showcontrollist = []
    colors = {}
    colors['CBF'] = 'b'
    colors['LR'] = 'r'
    styles = ['solid', 'dashed', 'dotted']

    for sh in ['CBF', 'LR']:
        for rb in road_bounds:
            for yindx, yc in enumerate(yaw_consts):
                if yc is not None:
                    suffixlist.append(os.path.join("road_boundary=" + str(rb) + ", yaw="+str(round(yc, 2)), sh))
                    labellist.append(sh+"-DDP $\delta \\theta \leq$"+str(label_yc[yindx])+"$\pi$")
                else:
                    suffixlist.append(os.path.join("road_boundary=" + str(rb) + ", yaw="+str(yc), sh))
                    labellist.append(sh+"-DDP No $\delta \\theta$")
                colorlist.append(colors[sh])
                stylelist.append(styles[yindx])
                rblist.append(rb)

                if sh=='LR' and yc is None:
                    showlist.append(False)
                else:
                    showlist.append(True)                
            
                if sh=='LR' and yc==0.5*np.pi:
                    showcontrollist.append(True)
                elif sh=='CBF' and yc is None:
                    showcontrollist.append(True)
                else:
                    showcontrollist.append(False)

    plot_states_list = []
    plot_actions_list = []
    plot_values_list = []
    plot_times_list = []
    plot_deviations_list = []
    plot_states_complete_filter_list = []
    plot_states_barrier_filter_list = []

    filter_type = []
    filter_params = []
    for suffix in suffixlist:
        plot_data = np.load(prefix+"/"+suffix+"/figure/save_data.npy", allow_pickle=True)
        plot_data = plot_data.ravel()[0]
        plot_states_complete_filter_list.append( np.array(plot_data['complete_indices'] ) )
        plot_states_barrier_filter_list.append( np.array(plot_data['barrier_indices'] ) )
        plot_states_list.append( np.array(plot_data['states'] ) )
        plot_actions_list.append( np.array(plot_data['actions']) )
        plot_values_list.append( np.array(plot_data['values']) )
        plot_times_list.append( np.array(plot_data['process_times']) )
        plot_deviations_list.append( np.array(plot_data['deviation_history']) )

        config_file = prefix+"/"+suffix+"/config.yaml"
        config = load_config( config_file )
        config_env = config['environment']
        config_agent = config["agent"] 
        config_solver= config["solver"]
        config_cost = config["cost"]
        config_cost.N = config_solver.N

        action_space = np.array(config_agent.ACTION_RANGE, dtype=np.float32)
        max_iters = config_solver.MAX_ITER_RECEDING
        
        if config_solver.FILTER_TYPE=='CBF':
            filter_type.append(1)
            filter_params.append(config_solver.BARRIER_GAMMA)
        elif config_solver.FILTER_TYPE=='LR':
            filter_type.append(2)
            filter_params.append(config_solver.SHIELD_THRESHOLD) 

        c_obs = 'k'
        env = CarSingle5DEnv(config_env, config_agent, config_cost)

        fig, axes = plt.subplots(
            1, 1, figsize=(5.1, 1.15)
        )

        for ax in [axes]:
            # track, obstacles, footprint
            env.render_obs(ax=ax, c=c_obs)
            ax.axis(env.visual_extent)
            ax.set_aspect('equal')

        lgd_c = True
        lgd_b = True
        for idx, state_data in enumerate(plot_states_list):
            if showlist[idx]:
                sc = ax.plot(
                    state_data[:, 0], state_data[:, 1], c=colorlist[int(idx)], alpha = 1.0, 
                    label=labellist[int(idx)], linewidth=0.8, linestyle=stylelist[int(idx)]
                )

                complete_filter_indices = plot_states_complete_filter_list[idx]
                barrier_filter_indices = plot_states_barrier_filter_list[idx]
                if len(complete_filter_indices)>0:
                    if lgd_c:
                        ax.plot(state_data[complete_filter_indices, 0], state_data[complete_filter_indices, 1], 'o', color=colorlist[int(idx)], alpha=0.7, markersize=3.0, label='Complete filter')
                        lgd_c = False
                    else:
                        ax.plot(state_data[complete_filter_indices, 0], state_data[complete_filter_indices, 1], 'o', color=colorlist[int(idx)], alpha=0.7, markersize=3.0)
                if len(barrier_filter_indices)>0:
                    if lgd_b:
                        ax.plot(state_data[barrier_filter_indices, 0], state_data[barrier_filter_indices, 1], 'x', color=colorlist[int(idx)], alpha=0.7, markersize=3.0, label='CBF filter')
                        lgd_b = False
                    else:
                        ax.plot(state_data[barrier_filter_indices, 0], state_data[barrier_filter_indices, 1], 'x', color=colorlist[int(idx)], alpha=0.7, markersize=3.0)
    
            ax.legend(fontsize=legend_fontsize, loc='upper left', ncol=4, bbox_to_anchor=(0.0, 1.35), fancybox=True, shadow=True)
            ax.set_xticks([0, 0.5*env.visual_extent[1], env.visual_extent[1]], fontsize=legend_fontsize)
            ax.set_yticks([env.visual_extent[2], 0, env.visual_extent[3]], fontsize=legend_fontsize)
            ax.plot(np.linspace(0, env.visual_extent[1], 100), np.array([rblist[idx]]*100), 'k--')
            ax.plot( np.linspace(0, env.visual_extent[1], 100), np.array([-1*rblist[idx]]*100), 'k--')
            ax.set_xlabel('X position', fontsize=legend_fontsize)
            ax.set_ylabel('Y position', fontsize=legend_fontsize)


        fig.savefig(
            plot_folder + tag + "_jax_trajectories.pdf", dpi=200,bbox_inches='tight'
        )
        fig.savefig(
            plot_folder + tag + "_jax_trajectories.png", dpi=200,bbox_inches='tight'
        )

        matplotlib.rc('xtick', labelsize=4) 
        matplotlib.rc('ytick', labelsize=4)

        fig, axes = plt.subplots(2, 1, figsize=(2.4, 1.25))
        
        maxsteps = 0
        for idx, controls_data in enumerate(plot_actions_list):
            if showcontrollist[idx]:
                nsteps = controls_data.shape[0]
                maxsteps = np.maximum(maxsteps, nsteps)
                fillarray = np.zeros(nsteps)
                fillarray[np.array(plot_states_barrier_filter_list[idx], dtype=np.int)] = 1
                axes[0].plot(controls_data[:, 0], label=labellist[int(idx)], c=colorlist[int(idx)], alpha = 1.0, linewidth=1, linestyle=stylelist[idx])
                axes[1].plot(controls_data[:, 1], label=labellist[int(idx)], c=colorlist[int(idx)], alpha = 1.0, linewidth=1, linestyle=stylelist[idx])
                axes[0].fill_between(range(nsteps), action_space[0, 0], action_space[0, 1], where=fillarray, color='b', alpha=0.3)
                axes[1].fill_between(range(nsteps), action_space[1, 0], action_space[1, 1], where=fillarray, color='b', alpha=0.3)

            legend_fontsize=4
            #axes[0].set_xlabel('Time index', fontsize=legend_fontsize)
            axes[0].set_ylabel('Acceleration', fontsize=legend_fontsize)
            #axes[0].grid(True)
            axes[0].set_xticks([], fontsize=3, labelsize=3)
            axes[0].set_yticks([action_space[0, 0], 0, action_space[0, 1]], fontsize=3, labelsize=3)
            axes[0].legend(fontsize=legend_fontsize, loc='upper left', ncol=3, bbox_to_anchor=(-0.05, 1.4), fancybox=True, shadow=True)

            axes[1].set_xlabel('Time index', fontsize=legend_fontsize)
            axes[1].set_ylabel('Steering control', fontsize=legend_fontsize)
            #axes[1].grid(True)
            axes[1].set_xticks([0, 0.5*maxsteps, maxsteps], fontsize=3, labelsize=3)
            axes[1].set_yticks([action_space[1, 0], 0, action_space[1, 1]], fontsize=3, labelsize=3)
            #axes[1].legend(fontsize=legend_fontsize)

        fig.savefig(
            plot_folder + tag + "_jax_controls.pdf", dpi=200, bbox_inches='tight'
        )
        fig.savefig(
            plot_folder + tag + "_jax_controls.png", dpi=200, bbox_inches='tight'
        )


    print("Reporting stats")
    for idx, controls_data in enumerate(plot_actions_list):
        print("Type: ", labellist[idx])
        jerklist = find_jerk(controls_data)
        timelist = plot_times_list[idx]
        print("Acceleration jerk: ", jerklist[0], " +- ", jerklist[2])
        print("Steer jerk: ", jerklist[1], " +- ", jerklist[3])
        print("Total deviation: ", np.sum(plot_deviations_list[idx]))
        print("Process time: ", np.mean(timelist), "+-", np.std(timelist))