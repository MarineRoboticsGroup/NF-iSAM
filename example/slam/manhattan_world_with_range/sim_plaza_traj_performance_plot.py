import random

import matplotlib.pyplot as plt
import numpy as np
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os
from slam.RunBatch import group_nodes_factors_incrementally
from scipy import stats
import matplotlib
from factors.Factors import PriorFactor, SE2RelativeGaussianLikelihoodFactor
from geometry.TwoDimension import SE2Pose
import seaborn as sns
import pandas as pd
from example.slam.manhattan_world_with_range.mmd_rmse_time_da_plot_grid import reorder_samples
from slam.FactorGraphSimulator import read_factor_graph_from_file
matplotlib.rcParams.update({'font.size': 16})

if __name__ == '__main__':
    plot_traj = True
    plot_performance = True

    selected_step_nums = [6, 19, 39, 135]
    case_dir = "manhattan_plaza/res/seed0/pada0.4_r2_odom0.01_mada3"

    if plot_traj:
        # case_list = [case_dir+'/'+dir for dir in os.listdir(case_dir) if os.path.isdir(case_dir+'/'+dir)]
        case_list = [f"{case_dir}"]

        plot_args = {'xlim': (-150, 350), 'ylim': (-150, 350), 'fig_size': (8, 8), 'truth_label_offset': (8, 0)}

        num_samples = 500

        run_id = 5

        color_list = ['m','darkorange','black','y','c','b','g','r']
        colors = {}
        for case_folder in case_list:
            fg_file = case_folder+"/factor_graph.fg"
            run_folder = f"run{run_id}"

            run_dir = f"{case_folder}/{run_folder}"

            nodes, truth, factors = read_factor_graph_from_file(fg_file)

            for var in nodes:
                if var.type == VariableType.Landmark:
                    colors[var] = color_list.pop(0)
                else:
                    colors[var] = 'r'

            plot_dir = f"{case_folder}/figures"
            if(os.path.exists(plot_dir)):
                pass
            else:
                os.mkdir(plot_dir)

            nodes_factors_by_step = group_nodes_factors_incrementally(
                nodes=nodes, factors=factors, incremental_step=1)

            rbt_vars = []
            var2pose = {}
            odom_x = []
            odom_y = []
            for step in range(len(nodes_factors_by_step)):
                step_nodes, step_factors = nodes_factors_by_step[step]
                for f in step_factors:
                    if isinstance(f, PriorFactor):
                        rbt_vars.append(f.vars[0])
                        var2pose[f.vars[0]] = SE2Pose(*f.observation)
                        odom_y.append(var2pose[rbt_vars[-1]].y)
                        odom_x.append(var2pose[rbt_vars[-1]].x)
                    elif isinstance(f, SE2RelativeGaussianLikelihoodFactor):
                        if f.var1 == rbt_vars[-1]:
                            var2pose[f.var2] = var2pose[f.var1] * SE2Pose(*f.observation)
                            rbt_vars.append(f.var2)
                            odom_y.append(var2pose[rbt_vars[-1]].y)
                            odom_x.append(var2pose[rbt_vars[-1]].x)

            fig = plt.figure(figsize=(5 * len(selected_step_nums), 5 ))
            gs = fig.add_gridspec(1, len(selected_step_nums), hspace=0.0, wspace=0.15)
            axs = gs.subplots()

            for i, step in enumerate(selected_step_nums):
                cur_sample = None
                step_nodes, step_factors = nodes_factors_by_step[step]

                recent_rbt_vars = []
                for var in step_nodes:
                    if var.type == VariableType.Pose:
                        recent_rbt_vars.append(var)

                order_file = f"{run_dir}/step{step}_ordering"
                sample_file = f"{run_dir}/step{step}"
                sol_label = "NF-iSAM"

                if os.path.exists(sample_file):
                    cur_sample = np.loadtxt(fname=sample_file)
                    if cur_sample.shape[0] > num_samples:
                        cur_sample = cur_sample[np.random.choice(np.arange(len(cur_sample)), num_samples, False)]
                    order = Variable.file2vars(order_file=order_file)
                    ax = axs[i]
                    ax.plot(odom_x, odom_y, '-', c = '0.8')
                    plot_2d_samples(ax=ax, samples_array=cur_sample, variable_ordering=order,
                                    show_plot=False, equal_axis=False,
                                    colors={variable: pose for variable, pose in
                                           colors.items() if variable in order},
                                    # truth_factors={factor for factor in cur_factors},
                                    truth=truth,
                                    truth_factors=factors, title=f'Step {step}',
                                    plot_all_meas = False,
                                    plot_meas_give_pose = recent_rbt_vars,
                                    rbt_traj_no_samples = True,
                                    truth_R2 = True,
                                    truth_SE2 = False,
                                    ylabel=None,
                                    truth_odometry_color = 'k',
                                    truth_landmark_markersize = 10,
                                    truth_landmark_marker = 'x',
                                    **plot_args)
                    if i == 0:
                        ax.set_ylabel("y (m)")
            fig.savefig(f"{plot_dir}/traj_run{run_id}.png",dpi=300,bbox_inches="tight")
    if plot_performance:
        case_dir = "manhattan_plaza/res/seed0/pada0.4_r2_odom0.01_mada3"
        fg_path = f"{case_dir}/factor_graph.fg"
        nodes, truth, factors = read_factor_graph_from_file(
            f"{fg_path}")
        step_nums = np.arange(136)
        sample_num = 500
        f_size = 8
        plt.rc('xtick', labelsize=f_size)
        plt.rc('ytick', labelsize=f_size)
        plot_dir = f"{case_dir}/figures"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        df_path = f"{plot_dir}/time_rmse_data.txt"
        if not os.path.exists(df_path):
            time_rmse_data = []  # run, step, time, rmse
            for i in range(1, 7):
                run_dir = f"{case_dir}/run{i}"
                timing = np.loadtxt(f"{run_dir}/step_timing")
                for j, step in enumerate(step_nums):
                    run_sample_file = f"{run_dir}/step{step}"
                    run_order_file = f"{run_dir}/step{step}_ordering"
                    run_order = Variable.file2vars(order_file=run_order_file)
                    run_samples = np.loadtxt(run_sample_file)
                    if (run_samples.shape[0] >= sample_num):
                        downsampling_indices = np.array(random.sample(list(range(run_samples.shape[0])), sample_num))
                        run_samples = run_samples[downsampling_indices, :]
                    else:
                        print(f"run {i} has fewer samples than others at step {step}.")

                    # only keep translation dim
                    run_samples = reorder_samples(ref_order=run_order, sample_order=run_order, samples=run_samples)
                    assert (run_samples.shape[1] % 2 == 0)

                    true_xy = []
                    for var in run_order:
                        true_xy.append(truth[var][0])
                        true_xy.append(truth[var][1])
                    true_xy = np.array(true_xy)

                    mean_xy = np.mean(run_samples, axis=0)
                    diff_xy = mean_xy - true_xy
                    rmse = np.sqrt(np.mean(diff_xy ** 2))
                    time_rmse_data.append([i, step, timing[step], rmse])
            data = pd.DataFrame(time_rmse_data,
                                columns=['run', 'step', 'time', 'rmse'])
            data.to_csv(f"{df_path}", index=False)
        dist_time_df = pd.read_csv(f"{df_path}")
        dist_time_df.head()
        target_names = ['time', 'rmse']
        log_scale_names = ['rmse']
        label_names = ['Time (sec)', 'RMSE (m)']
        fig = plt.figure(figsize=(4, 2 * len(target_names)))
        gs = fig.add_gridspec(len(target_names), 1, hspace=0.05, wspace=0.1, top=.9, bottom=.1)
        axs = gs.subplots(sharex=True)
        for i, name in enumerate(target_names):
            ax = axs[i]
            if name in log_scale_names:
                ax.set(yscale="log")
            sns.lineplot(ax=ax, data=dist_time_df, x="step", y=name)
            ax.set_ylabel(label_names[i], fontsize=f_size + 2)
            ax.get_yaxis().set_label_coords(-0.1, 0.5)
            ax.set_xlabel("Time step", fontsize=f_size + 2)
            if name == "rmse":
                ax.set_yticks([1, 10, 100])
                for step in selected_step_nums:
                    ax.scatter(step, 1, s=30, marker='+', color='red', linewidths=.5)
                    ax.text(step, 1.2, s=f"{step}", fontsize=f_size - 4)
        plt.show()
        fig.savefig(f"{plot_dir}/performance_grid.png", dpi=300, bbox_inches='tight')