from matplotlib import pyplot as plt
from matplotlib import cm


def plot_cvg_animation(env, states, fig_folder="./",):
    c_obs = 'k'
    c_ego = 'c'
    
    fig = plt.figure(
        figsize=(5, 4)
    )

    ax = plt.gca()

    # track, obstacles, footprint
    env.render_obs(ax=ax, c=c_obs)
    env.render_target(ax=ax, c='g')
    ax.set_aspect('equal')
    
    sc = ax.plot(
        states[0, :-1], states[1, :-1], c='k', alpha=1.0, linewidth=1.5, linestyle='solid'
    )
    
    fig.savefig(
            fig_folder + "_jax_ilq_convergece.png", dpi=200, 
            bbox_inches='tight'
    )