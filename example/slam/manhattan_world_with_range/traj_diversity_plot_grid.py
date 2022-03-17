import json
import random
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from factors.Factors import AmbiguousDataAssociationFactor, PriorFactor, KWayFactor, BinaryFactor, \
    SE2RelativeGaussianLikelihoodFactor

from slam.RunBatch import group_nodes_factors_incrementally

from slam.FactorGraphSimulator import read_factor_graph_from_file
from slam.Variables import Variable, VariableType, R2Variable, SE2Variable
import os
import pandas as pd
import sklearn.metrics.pairwise

from utils.Functions import array_order_to_dict
from utils.Statistics import mmd, MMDb, MMDu2
import seaborn as sns
from matplotlib.ticker import MaxNLocator, ScalarFormatter

from utils.Visualization import plot_2d_samples


def reorder_samples(ref_order: List[Variable],
                    sample_order: List[Variable],
                    samples: np.ndarray):
    # res = np.zeros_like(samples)
    # cur_dim = 0
    # for var in ref_order:
    #     var_idx = sample_order.index(var)
    #     if var_idx == 0:
    #         sample_dim = 0
    #     else:
    #         sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
    #     res[:,cur_dim:cur_dim+var.dim] = samples[:,sample_dim:sample_dim+var.dim]
    #     cur_dim += var.dim
    # return res
    res = []
    cur_dim = 0
    for var in ref_order:
        var_idx = sample_order.index(var)
        if var_idx == 0:
            sample_dim = 0
        else:
            sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
        res.append(samples[:, sample_dim:sample_dim + 2])
        cur_dim += var.dim
    return np.hstack(res)

if __name__ == '__main__':

    # m_s = 12
    f_size = 14
    m_s = 10

    plot_scatter = False
    plot_perf = True
    compute_traj_div_perf = False

    all_scatter = True

    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    # plt.rc('xtick', labelsize=f_size)
    # plt.rc('ytick', labelsize=f_size)
    setup_folder = "random_4x4/res"

    plot_dir = f"random_4x4/res/figures"
    if (os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)

    # case_dirs = [f"{setup_folder}/{f}" for f in os.listdir(setup_folder) if os.path.isdir(f"{setup_folder}/{f}")]
    case_nums = list(range(0,6))
    case_dirs = [f"{setup_folder}/seed{i}" for i in case_nums]

    gtsam_folder = "gtsam"
    reference_folder = "dyn1"
    nf_folder = "run1"
    mm_folder = "caesar1"
    fg_file = "factor_graph.fg"

    folder2lgd = {}
    folder2lgd[nf_folder] = 'NF-iSAM'
    folder2lgd[mm_folder] = 'Caesar.jl'
    folder2lgd[reference_folder] = 'dynesty'
    folder2lgd[gtsam_folder] = 'GTSAM'

    folder2linestyle = {}
    folder2linestyle[nf_folder] = 'solid'
    folder2linestyle[reference_folder] = 'dashdot'
    folder2linestyle[mm_folder] = 'dashed'
    folder2linestyle[gtsam_folder] = 'dotted'

    folder2color = {}
    folder2color[nf_folder] = 'r'
    folder2color[reference_folder] = 'g'
    folder2color[mm_folder] = 'b'
    folder2color[gtsam_folder] = 'k'

    solver2linestyle = {}
    solver2linestyle[folder2lgd[nf_folder]] = ''
    solver2linestyle[folder2lgd[reference_folder]] = (3, 1.25, 1.5, 1.25)
    solver2linestyle[folder2lgd[mm_folder]] = (4, 1.5)
    solver2linestyle[folder2lgd[gtsam_folder]] = (1, 1)

    solver2color = {}
    solver2color[folder2lgd[nf_folder]] = 'r'
    solver2color[folder2lgd[reference_folder]] = 'g'
    solver2color[folder2lgd[mm_folder]] = 'b'
    solver2color[folder2lgd[gtsam_folder]] = 'k'

    solver2marker = {}
    solver2marker[folder2lgd[nf_folder]] = 'o'
    solver2marker[folder2lgd[reference_folder]] = 'X'
    solver2marker[folder2lgd[mm_folder]] = '^'
    solver2marker[folder2lgd[gtsam_folder]] = 'P'

    case_all_step_data = [] #Case, Step, Solver, Time, RMSE, Average MMD


    mmd_folders = [nf_folder, mm_folder, gtsam_folder]
    all_folders = [nf_folder, reference_folder, mm_folder, gtsam_folder]
    if all_scatter:
        scatter_folders = all_folders
    else:
        scatter_folders = [nf_folder, reference_folder]
    final_step = 15
    sample_num = 500

    mmd_func = MMDb
    MMD_type = 'MMDb'
    kernel_scale = 10

    color_list = ['m','darkorange','black','y','c','b','g','r']

    xlim=[-50, 220]
    ylims=np.array([[30, 130],[0, 170],[-30, 190],[-70,190],[-10,200],[-100,210],[-30,130],[50,190],[-30,120],[30,180]])

    y_lens = [ylim[1] - ylim[0] for ylim in ylims[case_nums]]
    y_len = sum(y_lens)
    x_len = xlim[1] - xlim[0]
    y_x_ratio = y_len / x_len

    fig_scale = 3
    scatter_fig = plt.figure(figsize=(fig_scale*len(scatter_folders),fig_scale*y_x_ratio))
    gs = scatter_fig.add_gridspec(len(case_dirs), len(scatter_folders),width_ratios=[1]*len(scatter_folders),
                                  height_ratios = y_lens, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey='row')

    for k, case_dir in enumerate(case_dirs):
        nodes, truth, factors = read_factor_graph_from_file(
            f"{case_dir}/{fg_file}")
        nodes_factors_by_step = group_nodes_factors_incrementally(
            nodes=nodes, factors=factors, incremental_step=1)

        colors = {}
        for i, node in enumerate(nodes):
            colors[node] = color_list[i % len(color_list)]

        if plot_scatter:
            for i, folder in enumerate(scatter_folders):
                dir = f"{case_dir}/{folder}"
                step = final_step
                if folder == reference_folder:
                    sample_file = f"{dir}/step{step}.sample"
                    order_file = f"{dir}/step{step}_ordering"
                else:
                    sample_file = f"{dir}/step{step}"
                    order_file = f"{dir}/step{step}_ordering"
                if os.path.exists(sample_file):
                    samples = np.loadtxt(sample_file)
                    order = Variable.file2vars(order_file=order_file)
                    if (samples.shape[0] >= sample_num):
                        downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                        trimmed = samples[downsampling_indices, :]
                    else:
                        print(f"{folder} has fewer samples than others at step {step}.")
                        trimmed = samples
                    part_color = {key: colors[key] for key in order}
                    ax = plot_2d_samples(ax=axs[k, i], samples_array=trimmed, variable_ordering=order,
                                         show_plot=False, equal_axis=False, colors=part_color, marker_size=.1,
                                         xlabel=None, ylabel=None,
                                         xlim=xlim, ylim=ylims[k])
                    for node in order:
                        if (node.name == 'L1'):
                            dx, dy = [0, -20]
                        elif (node.name == 'L2'):
                            dx, dy = [0, -20]
                        elif (node.name == 'X1'):
                            dx, dy = [0, 10]
                        elif (node.name == 'X0'):
                            dx, dy = [-25, 0]
                        elif (node.name == 'X2'):
                            dx, dy = [0, 10]
                        elif (node.name == 'X5'):
                            dx, dy = [10, 0]
                        elif (node.name == 'X4'):
                            dx, dy = [0, 10]
                        elif (node.name == 'X3'):
                            dx, dy = [0, 10]
                        else:
                            dx, dy = [15, 0]
                        if len(truth[node]) == 2:
                            x, y = truth[node][:2]
                            color = "blue"
                            marker = "x"
                            ax.scatter(x, y, s=20, marker=marker, color=color, linewidths=1.0)
                            ax.text(x + dx, y + dy, s=node.name, fontsize=f_size - 3)
                        # else:
                        #     x, y, th = truth[node][:3]
                        #     color = "red"
                        #     marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                        #     marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                        # ax.plot([x], [y], c=color, markersize=2,mar marker="+")
                        # ax.scatter(x, y, s=90,facecolors='none',edgecolors='r')

                    for factor in factors:
                        if not isinstance(factor, PriorFactor) and (set(factor.vars).issubset(set(order))):
                            if isinstance(factor, KWayFactor):
                                var1 = factor.root_var
                                var2s = factor.child_vars
                                for var2 in var2s:
                                    x1, y1 = truth[var1][:2]
                                    x2, y2 = truth[var2][:2]
                                    ax.plot([x1, x2], [y1, y2], linestyle='--', dashes=(5, 5), c='red', linewidth=.5)
                            elif isinstance(factor, SE2RelativeGaussianLikelihoodFactor):
                                var1, var2 = factor.vars
                                x1, y1 = truth[var1][:2]
                                x2, y2 = truth[var2][:2]
                                ax.plot([x1, x2], [y1, y2], c='lime', linewidth=.5)
                            elif isinstance(factor, BinaryFactor):
                                var1, var2 = factor.vars
                                x1, y1 = truth[var1][:2]
                                x2, y2 = truth[var2][:2]
                                ax.plot([x1, x2], [y1, y2], c='k', linewidth=.5)
                    # ax.axis("equal")
                else:
                    axs[k, i].text(25, 60, 'No solution', fontsize=30)
                if i == 0:
                    axs[k, i].set_ylabel(f"Case {k+1}", fontsize=f_size + 4)
                    axs[k, i].get_yaxis().set_label_coords(-.2, 0.5)
                if k == len(case_dirs)-1:
                    axs[k, i].set_xlabel(folder2lgd[folder], fontsize=f_size + 4)
            # scatter_fig.savefig(f"{plot_dir}/scatter_final_step.png",dpi=300,bbox_inches="tight")
            if all_scatter:
                scatter_fig.savefig(f"{plot_dir}/all_scatter_final_step.png", dpi=300, bbox_inches="tight")
            else:
                scatter_fig.savefig(f"{plot_dir}/scatter_final_step.png", dpi=300, bbox_inches="tight")
        # translation dim

        # case_all_step_data = []  # Case, Step, Solver, Time, RMSE, Average MMD

        #reading timing files
        folder2timing = {}
        if compute_traj_div_perf:
            for i, folder in enumerate(all_folders):
                dir = f"{case_dir}/{folder}"
                if os.path.exists(f"{dir}/timing"):
                    folder2timing[folder] = np.loadtxt(f"{dir}/timing").tolist()
                elif os.path.exists(f"{dir}/step_timing"):
                    folder2timing[folder] = np.loadtxt(f"{dir}/step_timing").tolist()
                else:
                    print(dir)
                    raise ValueError("No timing files.")

            for j, step in enumerate(np.arange(16)):
                ref_dir = f"{case_dir}/{reference_folder}"
                ref_sample_file = f"{ref_dir}/step{step}.sample"
                ref_order_file = f"{ref_dir}/step{step}_ordering"
                ref_order = Variable.file2vars(order_file=ref_order_file)
                ref_samples = np.loadtxt(ref_sample_file)
                if (ref_samples.shape[0] >= sample_num):
                    downsampling_indices = np.array(random.sample(list(range(ref_samples.shape[0])), sample_num))
                    ref_samples = ref_samples[downsampling_indices, :]
                else:
                    print(f"{reference_folder} has fewer samples than others at step {step}.")

                # only keep translation dim
                ref_samples = reorder_samples(ref_order=ref_order, sample_order=ref_order, samples=ref_samples)
                assert (ref_samples.shape[1] % 2 == 0)

                true_xy = []
                for var in ref_order:
                    true_xy.append(truth[var][0])
                    true_xy.append(truth[var][1])
                true_xy = np.array(true_xy)

                mean_xy = np.mean(ref_samples, axis=0)
                diff_xy = mean_xy - true_xy
                rmse = np.sqrt(np.mean(diff_xy**2))
                time_data = folder2timing[reference_folder]
                # Case, Step, Solver, Time, RMSE, Average MMD
                case_all_step_data.append([k, step, folder2lgd[reference_folder], time_data[j], rmse, 0])
                for i, folder in enumerate(mmd_folders):
                    dir = f"{case_dir}/{folder}"
                    sample_file = f"{dir}/step{step}"
                    order_file =  f"{dir}/step{step}_ordering"
                    if os.path.exists(sample_file):
                        avg_mmd = 0
                        all_runtime = 0
                        last_rmse = 0
                        samples = np.loadtxt(sample_file)
                        order = Variable.file2vars(order_file=order_file)
                        if (samples.shape[0] >= sample_num):
                            downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                            samples = samples[downsampling_indices, :]
                        else:
                            print(f"{folder} has fewer samples than others at step {step}.")

                        samples = reorder_samples(ref_order=ref_order, sample_order=order, samples=samples)
                        assert (samples.shape[1] % 2 == 0)
                        cur_idx = 0
                        for var in ref_order:
                            m = mmd_func(samples[:,cur_idx:cur_idx+2], ref_samples[:,cur_idx:cur_idx+2], kernel_scale*np.sqrt(2))
                            cur_idx += 2
                            print(f"MMD of {var.name} for {folder2lgd[folder]} at step {step}: {m}")
                            avg_mmd += m
                        avg_mmd /= len(ref_order)
                        mean_xy = np.mean(samples, axis=0)
                        diff_xy = mean_xy - true_xy
                        rmse = np.sqrt(np.mean(diff_xy**2))
                        last_rmse = rmse
                        case_all_step_data.append([k, step, folder2lgd[folder], folder2timing[folder].pop(0), rmse, avg_mmd])
    columns = ["Case", "Step", "Solver", "Time", "RMSE", "Average_MMD"]
    dd_df = None
    if compute_traj_div_perf:
        dd_df = pd.DataFrame(case_all_step_data, columns=columns)
        dd_df.to_csv(f"{plot_dir}/compute_traj_div_perf.txt", index=False)
    else:
        dd_df = pd.read_csv(f"{plot_dir}/compute_traj_div_perf.txt")
    if dd_df is not None and plot_perf:
        dd_df.head()
        var_names = ["Case", "Step", "Solver"]
        target_names = ["Time", "RMSE", "Average_MMD"]
        label_names = ['Time (sec)', 'RMSE (m)', 'Average MMD']
        aspect_ratio = 1/3
        fig_scale = 2
        fig = plt.figure(figsize=(2 * fig_scale, fig_scale * 2 * len(label_names) * aspect_ratio))
        gs = fig.add_gridspec(len(label_names), 1, hspace=0.1*len(label_names)* aspect_ratio, wspace=0.05, top=.9, bottom=.1, right=.95, left=.05)
        axs = gs.subplots(sharex='col')

        row = 0
        label_dx = -0.13
        # plot ADA
        for i, folder in enumerate(all_folders):
            tmp_data = dd_df.query(f"Solver == '{folder2lgd[folder]}'")
            for j, target in enumerate(target_names):
                if folder == reference_folder and target == 'Average_MMD':
                    continue
                if folder == gtsam_folder and target == 'Time':
                    continue
                ax = axs[j]
                ax.set(yscale="log")
                if target == "RMSE":
                    ax=sns.lineplot(ax=ax, data=tmp_data, x="Step", y=target, label=folder2lgd[folder],color=solver2color[folder2lgd[folder]],
                              # marker=solver2marker[folder2lgd[folder]],
                              linestyle=folder2linestyle[folder],
                              markersize=int(m_s/4))
                else:
                    ax=sns.lineplot(ax=ax, data=tmp_data, x="Step", y=target,color=solver2color[folder2lgd[folder]],
                              # marker=solver2marker[folder2lgd[folder]],
                              linestyle=folder2linestyle[folder],
                              markersize=int(m_s/4))
                ax.set_ylabel(label_names[j], fontsize=f_size - 2)
                ax.tick_params(labelsize=f_size - 4)
                ax.set_xlabel("Step", fontsize=f_size - 2)
                ax.get_yaxis().set_label_coords(label_dx, 0.5)
        handles, labels = axs[1].get_legend_handles_labels()
        axs[1].get_legend().set_visible(False)
        fig.legend(handles, labels, loc="upper center", ncol=len(all_folders), prop={'size': 10},
                   bbox_to_anchor=(.5, 1.0))
        plt.show()
        fig.savefig(f"{plot_dir}/traj_diversity_performance_grid.png", dpi=300, bbox_inches='tight')
