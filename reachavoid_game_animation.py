from typing import Dict
import os
import sys
sys.path.append(".")

import copy
import numpy as np
import imageio
import argparse
from shutil import copyfile
import jax
import os
os.environ["CUDA_VISIBLE_DEVICES"] = " "

from simulators import(load_config, CarSingle5DEnv, BicycleReachAvoid5DMargin, PrintLogger, Bicycle5DCost, StatePlotter)
from summary.utils import(make_animation_plots, make_yaw_report, plot_run_summary)
from summary.convergence_plot import plot_cvg_animation

jax.config.update('jax_platform_name', 'cpu')

def main(config_file, plot_tag, road_boundary, is_task_ilqr):
  config = load_config(config_file)
  config_env = config['environment']
  config_agent = config['agent']
  config_solver = config['solver']
  config_solver.is_task_ilqr = is_task_ilqr
  config_cost = config['cost']
  dyn_id = config_agent.DYN

  # Provide common fields to cost
  config_cost.N = config_solver.N
  config_cost.V_MIN = config_agent.V_MIN
  config_cost.DELTA_MIN = config_agent.DELTA_MIN
  config_cost.V_MAX = config_agent.V_MAX
  config_cost.DELTA_MAX = config_agent.DELTA_MAX

  config_cost.TRACK_WIDTH_RIGHT = road_boundary
  config_cost.TRACK_WIDTH_LEFT = road_boundary
  config_env.TRACK_WIDTH_RIGHT = road_boundary
  config_env.TRACK_WIDTH_LEFT = road_boundary

  env = CarSingle5DEnv(config_env, config_agent, config_cost)
  x_cur = np.array(getattr(config_solver, "INIT_STATE", [0., 0., 0.5, 0., 0.]))
  env.reset(x_cur)

  # region: Constructs placeholder and initializes iLQR
  config_ilqr_cost = copy.deepcopy(config_cost)
  
  policy_type = None
  cost = None
  config_solver.COST_TYPE = config_cost.COST_TYPE
  if config_cost.COST_TYPE == "Reachavoid":
    if config_solver.FILTER_TYPE == "none":
      policy_type = "iLQRReachAvoidGame"
      #cost = BicycleReachAvoid5DMargin(config_ilqr_cost, copy.deepcopy(env.agent.dyn))
      cost = BicycleReachAvoid5DMargin(config_ilqr_cost, copy.deepcopy(env.agent.dyn))
      task_cost = Bicycle5DCost(config_ilqr_cost, copy.deepcopy(env.agent.dyn))
      env.cost = cost  #! hacky

 
  plotter = StatePlotter(env, './cvg_animate/') 
  env.agent.init_policy(
      policy_type=policy_type, config=config_solver, cost=cost, task_cost=task_cost, plotter=plotter
  )

  # region: Runs iLQR
  # Warms up jit
  _, solver_info = env.agent.policy.get_action(obs=x_cur, state=x_cur, warmup=True)
  
  states = solver_info['states']
  actions = solver_info['controls']
  disturbances = solver_info['disturbances']

  plot_cvg_animation(env, states)
  print('Converged cost', solver_info['Vopt'])
  #print('Gains', solver_info['Ks1'], solver_info['ks1'])
  #print('Controls', solver_info['controls'])
  #print('Disturbances', solver_info['disturbances'])
  env.report()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "-cf", "--config_file", help="Config file path", type=str,
      default=os.path.join("./simulators/test_config_yamls", "test_config.yaml")
  )

  parser.add_argument(
      "-pt", "--plot_tag", help="Save final plots", type=str,
      default=os.path.join("reachavoid")
  )

  parser.add_argument(
      "-rb", "--road_boundary", help="Choose road width", type=float,
      default=2.0
  )

  parser.add_argument('--naive_task', dest='naive_task', action='store_true')
  parser.add_argument('--no-naive_task', dest='naive_task', action='store_false')
  parser.set_defaults(naive_task=False)  
  
  args = parser.parse_args()
  main(args.config_file, args.plot_tag, args.road_boundary, (not args.naive_task))