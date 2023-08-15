from matplotlib import pyplot as plt
import imageio
import os
import numpy as np

class StatePlotter():
    def __init__(self, env, fig_folder):
        self.env = env
        self.fig_folder = fig_folder

    def plot_cvg_animation(self, states, critical=None, fig_name='test'):
        c_obs = 'k'
    
        fig = plt.figure(
            figsize=(4, 3)
        )

        ax = plt.gca()

        # track, obstacles, footprint
        self.env.render_obs(ax=ax, c=c_obs)
        self.env.render_target(ax=ax, c='g')
        ax.set_xlim([0, 1.6])
        ax.set_ylim([-0.6, 0.6])
        ax.set_aspect('equal')
        ax.set_xticks(ticks=[0, 1.5], labels=[0, 1.5], fontsize=6)
        ax.set_yticks(ticks=[-0.6, 0.6], labels=[-0.6, 0.6], fontsize=6)
        ax.set_xlabel('X position', fontsize=6)
        ax.set_ylabel('Y position', fontsize=6)
    
        sc = ax.plot(
            states[0, :], states[1, :], c='k', alpha=1.0, linewidth=1.0, linestyle='solid'
        )

        if critical is not None:
           failure_pinch = np.argwhere(critical==1).ravel()
           target_pinch = np.argwhere(critical==2).ravel()
           ax.plot(states[0, failure_pinch], states[1, failure_pinch], 'r+', markersize=3.5)
           ax.plot(states[0, target_pinch], states[1, target_pinch], 'k+', markersize=3.5)

        ax.yaxis.set_label_coords(-0.04, 0.5)
        ax.xaxis.set_label_coords(0.5, -0.04)
    
        fig.savefig(
            self.fig_folder + "jax_ilq_convergence_"+ str(fig_name) +".png", dpi=200, 
            bbox_inches='tight'
        )

        plt.close()

    def make_animation(self, niters):
        gif_path = os.path.join(self.fig_folder, 'rollout.gif')
        with imageio.get_writer(gif_path, mode='I') as writer:
          for i in range(niters):
            filename = self.fig_folder + "jax_ilq_convergence_"+ str(i) +".png"
            image = imageio.imread(filename)
            writer.append_data(image)