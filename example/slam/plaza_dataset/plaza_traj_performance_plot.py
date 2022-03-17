import random

import matplotlib.pyplot as plt
import numpy as np

from example.slam.manhattan_world_with_range.process_gtsam import getVars, getMeans
from utils.Visualization import plot_2d_samples
from utils.Functions import kabsch_umeyama
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
    plot_nf_traj = False
    plot_gtsam_traj = False
    plot_performance = True
    compute_performance = True
    KU_align = True

    selected_step_nums = []
    parent_dir =  "datasets/RangeOnlyDataset"
    plaza_list = ["Plaza1", "Plaza2"]
    ada_list = [0.0,0.2,0.4,0.6]
    label_list = ["0% ADA","20% ADA","40% ADA","60% ADA"]

    if plot_nf_traj:
        # case_list = [case_dir+'/'+dir for dir in os.listdir(case_dir) if os.path.isdir(case_dir+'/'+dir)]
        num_samples = 500

        run_id = 1
        run_folder = f"run{run_id}"

        plot_dir = f"{parent_dir}/figures"

        fig = plt.figure(figsize=(5.5 * len(ada_list), 5 * len(plaza_list)))
        gs = fig.add_gridspec(len(plaza_list), len(ada_list), hspace=0.05*len(plaza_list), wspace=0.05*len(ada_list), left=.2,right=.95,top=.95, bottom=.1)
        axs = gs.subplots()

        for i, plaza_n in enumerate(plaza_list):
            for j, ada_frac in enumerate(ada_list):
                if ada_frac == 0:
                    case_folder = f"{parent_dir}/{plaza_n}EFG"
                else:
                    case_folder = f"{parent_dir}/{plaza_n}ADA{ada_frac}EFG"
                fg_file = case_folder+"/factor_graph.fg"
                run_dir = f"{case_folder}/{run_folder}"
                nodes, truth, factors = read_factor_graph_from_file(fg_file)

                if(os.path.exists(plot_dir)):
                    pass
                else:
                    os.mkdir(plot_dir)

                nodes_factors_by_step = group_nodes_factors_incrementally(
                    nodes=nodes, factors=factors, incremental_step=1)

                rbt_vars = []
                var2odo_pose = {}
                odom_x = []
                odom_y = []
                for step in range(len(nodes_factors_by_step)):
                    step_nodes, step_factors = nodes_factors_by_step[step]
                    for f in step_factors:
                        if isinstance(f, PriorFactor):
                            rbt_vars.append(f.vars[0])
                            var2odo_pose[f.vars[0]] = SE2Pose(*f.observation)
                            odom_y.append(var2odo_pose[rbt_vars[-1]].y)
                            odom_x.append(var2odo_pose[rbt_vars[-1]].x)
                        elif isinstance(f, SE2RelativeGaussianLikelihoodFactor):
                            if f.var1 == rbt_vars[-1]:
                                var2odo_pose[f.var2] = var2odo_pose[f.var1] * SE2Pose(*f.observation)
                                rbt_vars.append(f.var2)
                                odom_y.append(var2odo_pose[rbt_vars[-1]].y)
                                odom_x.append(var2odo_pose[rbt_vars[-1]].x)

                step_nums = np.loadtxt(f"{run_dir}/step_list", dtype=int)
                step = step_nums[-1]
                order_file = f"{run_dir}/step{step}_ordering"
                sample_file = f"{run_dir}/step{step}"
                if os.path.exists(sample_file):
                    cur_sample = np.loadtxt(fname=sample_file)
                    if cur_sample.shape[0] > num_samples:
                        cur_sample = cur_sample[np.random.choice(np.arange(len(cur_sample)), num_samples, False)]
                    order = Variable.file2vars(order_file=order_file)

                    cur_sample = reorder_samples(ref_order=order, sample_order=order, samples=cur_sample)
                    assert (cur_sample.shape[1] % 2 == 0)

                    lmk_idx = np.array([idx for idx in range(len(order)) if order[idx].type == VariableType.Landmark])
                    rbt_idx = np.array([idx for idx in range(len(order)) if idx not in lmk_idx])

                    ax = axs[i,j]

                    true_xy = []
                    for var in order:
                        true_xy.append(truth[var][0])
                        true_xy.append(truth[var][1])
                    true_xy = np.array(true_xy)

                    mean_xy = np.mean(cur_sample, axis=0)
                    true_xy = true_xy.reshape((-1, 2))
                    mean_xy = mean_xy.reshape((-1, 2))
                    if KU_align:
                        R, c, t = kabsch_umeyama(true_xy, mean_xy)
                        mean_xy = np.array([t + c * R @ b for b in mean_xy])

                    if j == 0:
                        ax.plot(odom_x, odom_y, '--', c = '0.8',label="Odometry")

                    ax.plot(true_xy[rbt_idx,0],true_xy[rbt_idx,1], color='g', label="Ground Truth")
                    ax.scatter(true_xy[lmk_idx,0],true_xy[lmk_idx,1],color='g',marker='x')
                    for k in lmk_idx:
                        var = order[k]
                        dx, dy = 3, -3
                        if i == 0:
                            if var.name == "L2":
                                dx, dy = -9, 0
                        elif i == 1:
                            if var.name == "L3":
                                dx, dy = -9, 0
                        ax.text(truth[var][0]+dx, truth[var][1]+dy, var.name)
                    ax.plot(mean_xy[rbt_idx,0],mean_xy[rbt_idx,1],color='r', label="NF-iSAM")
                    ax.scatter(mean_xy[lmk_idx,0],mean_xy[lmk_idx,1],color='r',marker='x')
                    if i == len(plaza_list) - 1:
                        ax.set_xlabel("x (m)")
                        ax.text(0.5, -0.2, label_list[j],
                                va='center', ha='center',
                                transform=ax.transAxes,
                                color='k', fontsize=20)
                    if j == 0:
                        ax.set_ylabel("y (m)")
                        ax.text(-0.25, 0.5, plaza_n,
                                va='center', ha='center',
                                transform=ax.transAxes,
                                color='k', rotation=90,fontsize=20)
        axs[0, 0].legend()
        handles, labels = axs[0,0].get_legend_handles_labels()
        axs[0,0].get_legend().set_visible(False)
        fig.legend(handles, labels, loc="upper center", ncol=3, prop={'size': 18},bbox_to_anchor=(.55, 1.02))
        fig.savefig(f"{plot_dir}/NFiSAM_plaza.png",dpi=300,bbox_inches="tight")

    if plot_gtsam_traj:
        # case_list = [case_dir+'/'+dir for dir in os.listdir(case_dir) if os.path.isdir(case_dir+'/'+dir)]
        run_folder = "gtsam"

        plot_dir = f"{parent_dir}/figures"

        fig = plt.figure(figsize=(5.5 * len(plaza_list), 5))
        gs = fig.add_gridspec(1, len(plaza_list), hspace=0.05, wspace=0.05*len(ada_list), left=.05,right=.95,top=.95, bottom=.1)
        axs = gs.subplots()

        for i, plaza_n in enumerate(plaza_list):
            case_folder = f"{parent_dir}/{plaza_n}EFG"
            fg_file = case_folder+"/factor_graph.fg"
            run_dir = f"{case_folder}/{run_folder}"
            nodes, truth, factors = read_factor_graph_from_file(fg_file)

            if(os.path.exists(plot_dir)):
                pass
            else:
                os.mkdir(plot_dir)
            step_num = 0
            order_file = f"{run_dir}/step{step_num}_ordering"
            marg_file = f"{run_dir}/step_{step_num}_marginal"
            if (os.path.exists(order_file) and
                   os.path.exists(marg_file)):
                print("process ", marg_file)
                print(f"step {step_num}")
                order = getVars(order_file=order_file)
                var2mean = getMeans(order, marg_file)

                lmk_idx = np.array([idx for idx in range(len(order)) if order[idx].type == VariableType.Landmark])
                rbt_idx = np.array([idx for idx in range(len(order)) if idx not in lmk_idx])

                ax = axs[i]

                true_xy = []
                for var in order:
                    true_xy.append([truth[var][0],truth[var][1]])
                true_xy = np.array(true_xy)

                mean_xy = []
                for var in order:
                    mean_xy.append([var2mean[var].x,var2mean[var].y])
                mean_xy = np.array(mean_xy)

                if KU_align:
                    R, c, t = kabsch_umeyama(true_xy, mean_xy)
                    mean_xy = np.array([t + c * R @ b for b in mean_xy])

                ax.plot(true_xy[rbt_idx,0],true_xy[rbt_idx,1], color='g', label="Ground Truth")
                ax.scatter(true_xy[lmk_idx,0],true_xy[lmk_idx,1],color='g',marker='x')
                for k in lmk_idx:
                    var = order[k]
                    dx, dy = 3, -3
                    if i == 0:
                        if var.name == "L2":
                            dx, dy = -9, 0
                    elif i == 1:
                        if var.name == "L3":
                            dx, dy = -9, 0
                    ax.text(truth[var][0]+dx, truth[var][1]+dy, var.name)
                ax.plot(mean_xy[rbt_idx,0],mean_xy[rbt_idx,1],color='r', label="GTSAM")
                ax.scatter(mean_xy[lmk_idx,0],mean_xy[lmk_idx,1],color='r',marker='x')
                ax.set_xlabel("x (m)")
                ax.text(0.5, -0.2, plaza_n,
                        va='center', ha='center',
                        transform=ax.transAxes,
                        color='k', fontsize=20)
                if i == 0:
                    ax.set_ylabel("y (m)")
        axs[0].legend()
        handles, labels = axs[0].get_legend_handles_labels()
        axs[0].get_legend().set_visible(False)
        fig.legend(handles, labels, loc="upper center", ncol=3, prop={'size': 18},bbox_to_anchor=(.5, 1.08))
        fig.savefig(f"{plot_dir}/GTSAM_plaza.png",dpi=300,bbox_inches="tight")

    if plot_performance:
        run_folder = "run1"
        sample_num = 500
        f_size = 8
        inc_step = 5
        plt.rc('xtick', labelsize=f_size)
        plt.rc('ytick', labelsize=f_size)
        plot_dir = f"{parent_dir}/figures"
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)
        df_path = f"{plot_dir}/time_rmse_data.txt"
        if not os.path.exists(df_path):
            time_rmse_data = []  # run, step, time, rmse, case
            for i, plaza_n in enumerate(plaza_list):
                for k, ada_frac in enumerate(ada_list):
                    if ada_frac == 0:
                        case_folder = f"{parent_dir}/{plaza_n}EFG"
                    else:
                        case_folder = f"{parent_dir}/{plaza_n}ADA{ada_frac}EFG"
                    print(f"process case {case_folder}")
                    fg_path = case_folder+"/factor_graph.fg"
                    run_dir = f"{case_folder}/{run_folder}"
                    nodes, truth, factors = read_factor_graph_from_file(
                        f"{fg_path}")
                    rbt_poses = [var for var in nodes if var.type == VariableType.Pose]
                    rbt_num = len(rbt_poses)
                    # step_nums = np.arange(136)
                    # step_nums = np.arange(156)
                    # for i in range(1, 7):
                    step_nums = np.loadtxt(f"{run_dir}/step_list",dtype=int)
                    timing = np.loadtxt(f"{run_dir}/step_timing")
                    kabsch_umeyama_mat = None
                    for j, step in enumerate(step_nums[::-1]):
                        print(f"process step {step}")
                        run_sample_file = f"{run_dir}/step{step}"
                        run_order_file = f"{run_dir}/step{step}_ordering"
                        run_order = Variable.file2vars(order_file=run_order_file)
                        run_samples = np.loadtxt(run_sample_file)
                        if (run_samples.shape[0] >= sample_num):
                            downsampling_indices = np.array(random.sample(list(range(run_samples.shape[0])), sample_num))
                            run_samples = run_samples[downsampling_indices, :]
                        else:
                            print(f"{run_folder} has fewer samples than others at step {step}.")

                        # only keep translation dim
                        run_samples = reorder_samples(ref_order=run_order, sample_order=run_order, samples=run_samples)
                        assert (run_samples.shape[1] % 2 == 0)

                        true_xy = []
                        for var in run_order:
                            true_xy.append(truth[var][0])
                            true_xy.append(truth[var][1])
                        true_xy = np.array(true_xy)

                        mean_xy = np.mean(run_samples, axis=0)
                        if KU_align:
                            tmp_true = true_xy.reshape((-1,2))
                            tmp_mean = mean_xy.reshape((-1,2))
                            if kabsch_umeyama_mat is None:
                                R, c, t = kabsch_umeyama(tmp_true, tmp_mean)
                                kabsch_umeyama_mat = [R,c,t]
                            else:
                                R, c, t = kabsch_umeyama_mat
                            tmp_mean = np.array([t + c * R @ b for b in tmp_mean])
                            mean_xy = tmp_mean.flatten()
                        diff_xy = mean_xy - true_xy
                        rmse = np.sqrt(np.mean(diff_xy ** 2))
                        time_rmse_data.append([min((step+1)*inc_step,rbt_num), timing[step], rmse, plaza_n, ada_frac])
            data = pd.DataFrame(time_rmse_data,
                                columns=['step', 'time', 'rmse','case','ADA Frac.'])
            data.to_csv(f"{df_path}", index=False)
        dist_time_df = pd.read_csv(f"{df_path}")
        dist_time_df.head()
        target_names = ['time', 'rmse']
        log_scale_names = ['rmse']
        label_names = ['Time (sec)', 'RMSE (m)']
        fig = plt.figure(figsize=(2 * len(plaza_list), 2 * len(target_names)))
        gs = fig.add_gridspec(len(plaza_list), len(target_names), hspace=0.2, wspace=0.25, top=.9, bottom=.1)
        axs = gs.subplots()
        for i, name in enumerate(target_names):
            for j, plaza_n in enumerate(plaza_list):
                tmp_df = dist_time_df.query(f"case == '{plaza_n}'")
                ax = axs[i, j]
                if i==0 and j == 0:
                    legend_on = 'brief'
                else:
                    legend_on = False
                sns.lineplot(ax=ax, data=tmp_df, x="step", y=name, hue="ADA Frac.", legend=legend_on)
                if name in log_scale_names:
                    ax.set(yscale="log")
                if j == 0:
                    ax.set_ylabel(label_names[i], fontsize=f_size)
                    ax.get_yaxis().set_label_coords(-0.25, 0.5)
                else:
                    ax.set_ylabel('')
                if i == len(target_names) - 1:
                    ax.set_xlabel("Key Poses", fontsize=f_size)
                    ax.text(0.5, -0.35, plaza_n,
                            va='center', ha='center',
                            transform=ax.transAxes,
                            color='k', fontsize=f_size + 2)
                else:
                    ax.set_xlabel('')
        axs[0,0].legend()
        handles, labels = axs[0,0].get_legend_handles_labels()
        axs[0,0].get_legend().set_visible(False)
        fig.legend(handles[1:], label_list, loc="upper center", ncol=len(ada_list), prop={'size': f_size-2}, bbox_to_anchor=(.5, .98))
                # if name == "rmse":
                #     ax.set_yticks([0.1, 1, 10, 100])
                #     for step in selected_step_nums:
                #         ax.scatter(step, 1, s=30, marker='+', color='red', linewidths=.5)
                #         ax.text(step, 1.2, s=f"{step}", fontsize=f_size - 4)
        fig.savefig(f"{plot_dir}/performance_grid.png", dpi=300, bbox_inches='tight')