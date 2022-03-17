import random
from typing import List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from factors.Factors import AmbiguousDataAssociationFactor

from slam.RunBatch import group_nodes_factors_incrementally

from slam.FactorGraphSimulator import read_factor_graph_from_file
from slam.Variables import Variable, VariableType, R2Variable, SE2Variable
import os
import pandas as pd
import sklearn.metrics.pairwise

from utils.Functions import array_order_to_dict
from utils.Statistics import mmd, MMDb, MMDu2
import seaborn as sns
from matplotlib.ticker import MaxNLocator

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
    compute_mmd = True
    compute_dist_time = False
    compute_da_prob = False
    use_MMDb = True
    plot_mmd = True
    plot_dist_time = False
    plot_da_prob = False

    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    # plt.rc('xtick', labelsize=f_size)
    # plt.rc('ytick', labelsize=f_size)
    setup_folder = "journal_paper"
    case_folder = "case1"
    gtsam_folder = "gtsam"
    reference_folder = "dyn1"
    nf_folder = "run1"
    mm_folder = "caesar1"
    fg_file = "factor_graph.fg"

    folder2lgd = {}
    folder2lgd[nf_folder] = "NF-iSAM"
    folder2lgd[mm_folder] = "Caesar.jl"
    folder2lgd[reference_folder] = "dynesty"
    if case_folder[-2:] == "da":
        folder2lgd[gtsam_folder] = "Max-mixtures"
    else:
        folder2lgd[gtsam_folder] = "GTSAM"
        compute_da_prob = False
        plot_da_prob = False

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

    case_dir = f"{setup_folder}/{case_folder}"
    plot_dir = f"{setup_folder}/{case_folder}/figures"
    if(os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)

    nodes, truth, factors = read_factor_graph_from_file(
        f"{case_dir}/{fg_file}")

    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=1)
    step2dafactors = []
    true_da_pair = []
    for i, var_factor in enumerate(nodes_factors_by_step):
        cur_da_f = []
        if i > 0:
            cur_da_f += step2dafactors[-1]
        for f in var_factor[1]:
            if isinstance(f, AmbiguousDataAssociationFactor):
                cur_da_f.append(f)
                true_da_pair.append((f.root_var.name, f.child_vars[0].name))
        step2dafactors.append(cur_da_f)

    mmd_folders = [nf_folder,mm_folder, gtsam_folder]
    all_folders = [nf_folder,reference_folder, mm_folder, gtsam_folder]
    # mmd_folders = [mm_folder]

    step_nums = [0,1,2,3,4,5]

    sample_num = 1000
    # translation dim
    t_dim = 2

    raw_mmd_data = [] #append list of [solver,step,  var, MMD]
    raw_dist_time_data = [] # append list of [ solver,step, RMSE, time]
    raw_ada_prob_data = [] # append list of [ solver,step, true_x, true_l, prob]

    if plot_mmd and not compute_mmd:
        if not os.path.exists(f"{plot_dir}/raw_mmd_data.txt"):
            compute_mmd = True

    if plot_dist_time and not compute_dist_time:
        if not os.path.exists(f"{plot_dir}/raw_dist_time_data.txt"):
            compute_dist_time = True

    if use_MMDb:
        mmd_func = MMDb
        MMD_type = 'MMDb'
    else:
        mmd_func = MMDu2
        MMD_type = 'MMDu2'
    kernel_scale = 1

    #reading timing files
    folder2timing = {}
    for i, folder in enumerate(all_folders):
        dir = f"{setup_folder}/{case_folder}/{folder}"
        if os.path.exists(f"{dir}/timing"):
            folder2timing[folder] = np.loadtxt(f"{dir}/timing").tolist()
        elif os.path.exists(f"{dir}/step_timing"):
            folder2timing[folder] = np.loadtxt(f"{dir}/step_timing").tolist()
        else:
            raise ValueError("No timing files.")

    for step in step_nums:
        ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
        ref_sample_file = f"{ref_dir}/step{step}.sample"
        ref_order_file = f"{ref_dir}/step{step}_ordering"
        ref_order = Variable.file2vars(order_file=ref_order_file)
        ref_samples = np.loadtxt(ref_sample_file)
        if (ref_samples.shape[0] >= sample_num):
            downsampling_indices = np.array(random.sample(list(range(ref_samples.shape[0])), sample_num))
            ref_samples = ref_samples[downsampling_indices, :]
        else:
            print(f"{reference_folder} has fewer samples than others at step {step}.")

        # compute DA prob
        ADAs = step2dafactors[step]
        if compute_da_prob:
            sample_dict = array_order_to_dict(ref_samples, ref_order)
            for f in ADAs:
                weights = f.posterior_weights(sample_dict)
                # assume the first child var is the true association
                raw_ada_prob_data.append([folder2lgd[reference_folder], step, f.root_var.name,
                                          f.child_vars[0].name, weights[0]])

        # only keep translation dim
        ref_samples = reorder_samples(ref_order=ref_order, sample_order=ref_order, samples=ref_samples)
        assert (ref_samples.shape[1] % 2 == 0)

        true_xy = []
        for var in ref_order:
            true_xy.append(truth[var][0])
            true_xy.append(truth[var][1])
        true_xy = np.array(true_xy)

        if compute_dist_time:
            mean_xy = np.mean(ref_samples, axis=0)
            diff_xy = mean_xy - true_xy
            rmse = np.sqrt(np.sum(diff_xy**2)/len(ref_order))
            raw_dist_time_data.append([folder2lgd[reference_folder],step, rmse, folder2timing[reference_folder].pop(0)])

        for i, folder in enumerate(mmd_folders):
            dir = f"{setup_folder}/{case_folder}/{folder}"
            if folder == gtsam_folder:
                sample_file = f"{dir}/batch{step}"
                order_file = f"{dir}/batch_{step}_ordering"
            else:
                sample_file = f"{dir}/step{step}"
                order_file =  f"{dir}/step{step}_ordering"
            if os.path.exists(sample_file):
                samples = np.loadtxt(sample_file)
                order = Variable.file2vars(order_file=order_file)
                if (samples.shape[0] >= sample_num):
                    downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                    samples = samples[downsampling_indices, :]
                else:
                    print(f"{folder} has sewer samples than others at step {step}.")
                # ADA prob
                if compute_da_prob:
                    sample_dict = array_order_to_dict(samples, order)
                    for f in ADAs:
                        weights = f.posterior_weights(sample_dict)
                        raw_ada_prob_data.append([folder2lgd[folder], step, f.root_var.name,
                                                  f.child_vars[0].name, weights[0]])

                samples = reorder_samples(ref_order=ref_order, sample_order=order, samples=samples)
                assert (samples.shape[1] % 2 == 0)
                if compute_mmd:
                    m = mmd_func(samples, ref_samples, kernel_scale*np.sqrt(samples.shape[1]))
                    print(f"MMD of joint posterior for {folder2lgd[folder]} at step {step}: {m}")
                    raw_mmd_data.append([folder2lgd[folder], 'Joint',step,  m, kernel_scale*np.sqrt(samples.shape[1])])

                    cur_idx = 0
                    for var in ref_order:
                        m = mmd_func(samples[:,cur_idx:cur_idx+2], ref_samples[:,cur_idx:cur_idx+2], kernel_scale*np.sqrt(2))
                        cur_idx += 2
                        print(f"MMD of {var.name} for {folder2lgd[folder]} at step {step}: {m}")
                        raw_mmd_data.append([folder2lgd[folder], var.name, step, m, kernel_scale*np.sqrt(2)])

                if compute_dist_time:
                    mean_xy = np.mean(samples, axis=0)
                    diff_xy = mean_xy - true_xy
                    rmse = np.sqrt(np.sum(diff_xy ** 2) / len(ref_order))
                    raw_dist_time_data.append(
                        [folder2lgd[folder],step, rmse, folder2timing[folder].pop(0)])

    mmd_df = None
    if compute_mmd and len(raw_mmd_data) > 0:
        mmd_df = pd.DataFrame(raw_mmd_data, columns=['Solver','Var','Step',MMD_type, 'Sigma'])
        mmd_df.to_csv(f"{plot_dir}/raw_mmd_data.txt", index=False)
    else:
        #start to plot
        mmd_df = pd.read_csv(f"{plot_dir}/raw_mmd_data.txt")
        
    dist_time_df = None
    if compute_dist_time and len(raw_dist_time_data) > 0:
        dist_time_df = pd.DataFrame(raw_dist_time_data, columns=['Solver','Step','RMSE', 'Time'])
        dist_time_df.to_csv(f"{plot_dir}/raw_dist_time_data.txt", index=False)
    else:
        #start to plot
        dist_time_df = pd.read_csv(f"{plot_dir}/raw_dist_time_data.txt")

    da_prob_df = None
    if compute_da_prob and len(raw_ada_prob_data) > 0:
        da_prob_df = pd.DataFrame(raw_ada_prob_data, columns=['Solver','Step','Pose', 'Landmark', 'Probability'])
        da_prob_df.to_csv(f"{plot_dir}/raw_ada_prob_data.txt", index=False)
    else:
        #start to plot
        if os.path.exists(f"{plot_dir}/raw_ada_prob_data.txt"):
            da_prob_df = pd.read_csv(f"{plot_dir}/raw_ada_prob_data.txt")


    if plot_mmd:
        mmd_df.head()
        target_names = ['Joint', 'L1', 'L2']
        fig = plt.figure(figsize=(8,2*len(target_names)))
        gs = fig.add_gridspec(len(target_names), 1, hspace=0.05, wspace=0.1, top=.9, bottom=.1)
        axs = gs.subplots()

        for i, name in enumerate(target_names):
            var_mmd = mmd_df.query(f"Var == '{name}'")
            if i >= len(target_names) - 1:
                ax=sns.lineplot(ax=axs[i], data=var_mmd, x="Step", y=MMD_type, hue="Solver", style="Solver",markers=solver2marker, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            else:
                ax=sns.lineplot(ax=axs[i], data=var_mmd, x="Step", y=MMD_type, hue="Solver", style="Solver",markers=solver2marker, legend=False, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            ax.set_ylabel(f"{name} MMD", fontsize=f_size + 2)
            ax.get_yaxis().set_label_coords(-0.08, 0.5)
            ax.set_xlabel(f"Time step", fontsize=f_size + 2)
        handles, labels = axs[-1].get_legend_handles_labels()
        axs[-1].get_legend().set_visible(False)
        fig.legend(handles[1:], labels[1:], loc="upper center", ncol=len(mmd_folders), prop={'size': 12},bbox_to_anchor=(.5, .98) )
        plt.show()
        fig.savefig(f"{plot_dir}/mmd_grid.png", dpi=300, bbox_inches='tight')

    if plot_dist_time:
        dist_time_df.head()
        target_names = ['Time', 'RMSE']
        label_names = ['Runtime (sec)','RMSE (m)']
        fig = plt.figure(figsize=(8,2*len(target_names)))
        gs = fig.add_gridspec(len(target_names), 1, hspace=0.05, wspace=0.1, top=.9, bottom=.1)
        axs = gs.subplots()

        for i, name in enumerate(target_names):
            axs[i].set(yscale="log")
            if i >= len(target_names) - 1:
                ax=sns.lineplot(ax=axs[i], data=dist_time_df, x="Step", y=name, hue="Solver", style="Solver",markers=solver2marker, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            else:
                ax=sns.lineplot(ax=axs[i], data=dist_time_df, x="Step", y=name, hue="Solver", style="Solver",markers=solver2marker, legend=False, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            ax.set_ylabel(label_names[i], fontsize=f_size + 2)
            ax.get_yaxis().set_label_coords(-0.08, 0.5)
            ax.set_xlabel(f"Time step", fontsize=f_size + 2)
        handles, labels = axs[-1].get_legend_handles_labels()
        axs[-1].get_legend().set_visible(False)
        fig.legend(handles[1:], labels[1:], loc="upper center", ncol=len(all_folders), prop={'size': 10},bbox_to_anchor=(.5, 1.02) )
        plt.show()
        fig.savefig(f"{plot_dir}/performance_grid.png", dpi=300, bbox_inches='tight')

    if plot_da_prob:
        da_prob_df.head()
        num_plots = len(true_da_pair)
        fig = plt.figure(figsize=(8,2*num_plots))
        gs = fig.add_gridspec(num_plots, 1, hspace=0.05, wspace=0.1, top=.9, bottom=.1)
        axs = gs.subplots(sharex=True)

        for i, rbt_lmk in enumerate(true_da_pair):
            var_mmd = da_prob_df.query(f"Landmark == '{rbt_lmk[1]}'" and f"Pose == '{rbt_lmk[0]}'")
            if i >= len(true_da_pair) - 1:
                ax=sns.lineplot(ax=axs[i], data=var_mmd, x="Step", y="Probability", hue="Solver", style="Solver",markers=solver2marker, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            else:
                ax=sns.lineplot(ax=axs[i], data=var_mmd, x="Step", y="Probability", hue="Solver", style="Solver",markers=solver2marker, legend=False, dashes=solver2linestyle, palette=solver2color, markersize =m_s)
            ax.set_ylabel(f"P({rbt_lmk[0]}-{rbt_lmk[1]})", fontsize=f_size + 2)
            ax.get_yaxis().set_label_coords(-0.08, 0.5)
            ax.set_xlabel(f"Time step", fontsize=f_size + 2)
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        # handles, labels = axs[-1].get_legend_handles_labels()
        # axs[-1].get_legend().set_visible(False)
        # fig.legend(handles[1:], labels[1:], loc="upper center", ncol=len(all_folders), prop={'size': 12},bbox_to_anchor=(.5, .98) )
        plt.show()
        fig.savefig(f"{plot_dir}/da_prob_grid.png", dpi=300, bbox_inches='tight')