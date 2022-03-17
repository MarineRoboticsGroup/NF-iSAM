import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from slam.FactorGraphSimulator import read_factor_graph_from_file
from slam.Variables import Variable, VariableType
import os

if __name__ == '__main__':

    # m_s = 12
    f_size = 22
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    # plt.rc('xtick', labelsize=f_size)
    # plt.rc('ytick', labelsize=f_size)
    setup_folder = "lawnmower_4x4/res/seed1"
    case_folder = "pada0.4_r2_odom0.01_mada3"
    gtsam_folder = "gtsam"
    reference_folder = "dyn1"
    nf_folder = "run1"
    mm_folder = "caesar1"
    fg_file = "factor_graph.fg"

    folder2lgd = {}
    folder2lgd[nf_folder] = "NF-iSAM"
    folder2lgd[mm_folder] = "Caesar.jl"
    folder2lgd[reference_folder] = "dynesty"
    folder2lgd[gtsam_folder] = "Max-mixtures"

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

    case_dir = f"{setup_folder}/{case_folder}"
    ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
    gtsam_dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"
    nf_dir = f"{setup_folder}/{case_folder}/{nf_folder}"

    plot_dir = f"{setup_folder}/{case_folder}/figures"
    if(os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)

    nodes, truth, factors = read_factor_graph_from_file(
        f"{case_dir}/{fg_file}")

    folders = [nf_folder, reference_folder,mm_folder, gtsam_folder]
    # folders = [mm_folder]

    step_nums = [5,10,15]
    lmk_var_names = ['L0', 'L1', 'L2']
    name2truth = {}
    for var in nodes:
        if var.name in lmk_var_names:
            name2truth[var.name] = [truth[var][0],truth[var][1]]

    xlim = [-50, 230]
    ylim = [-50, 230]
    kde_pts = 500
    kde_bw= None#'silverman'

    sample_num = 1000

    fig = plt.figure(figsize=(2*len(lmk_var_names)*2*3, len(step_nums)*3))
    gs = fig.add_gridspec(len(step_nums), 2*len(lmk_var_names), hspace=0.15, wspace=0.05)
    axs = gs.subplots(sharex=True, sharey=False)

    legend_plot = []
    legend_label = []

    for i, folder in enumerate(folders):
        for j, step in enumerate(step_nums):
            dir = f"{setup_folder}/{case_folder}/{folder}"
            if folder == reference_folder:
                sample_file = f"{dir}/step{step}.sample"
                order_file = f"{dir}/step{step}_ordering"
            else:
                sample_file = f"{dir}/step{step}"
                order_file =  f"{dir}/step{step}_ordering"
            if os.path.exists(sample_file):
                samples = np.loadtxt(sample_file)
                order = Variable.file2vars(order_file=order_file)
                if (samples.shape[0] >= sample_num):
                    downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                    trimmed = samples[downsampling_indices, :]
                else:
                    print(f"{folder} has fewer samples than others at step {step}.")
                    trimmed = samples
                var2sample = {}
                cur_idx = 0
                for var in order:
                    if var.name in lmk_var_names:
                        var2sample[var.name] = trimmed[:, cur_idx:cur_idx+var.dim]
                    if len(var2sample) == len(lmk_var_names):
                        break
                    cur_idx += var.dim

                cur_idx = 0
                l = None
                for var_name in lmk_var_names:
                    if var_name in var2sample:
                        true_x, true_y = name2truth[var_name]
                        x, y = var2sample[var_name][:,0], var2sample[var_name][:,1]
                        xpts = np.linspace(xlim[0], xlim[1], kde_pts)
                        ypts = np.linspace(ylim[0], ylim[1], kde_pts)
                        x_kernel = gaussian_kde(x, bw_method=kde_bw)
                        y_kernel = gaussian_kde(y, bw_method=kde_bw)
                        axs[j, cur_idx].plot(xpts,x_kernel(xpts),linewidth=2, linestyle = folder2linestyle[folder], color=folder2color[folder], label=folder2lgd[folder])
                        axs[j, cur_idx].scatter(true_x, 1e-6,s=150, marker='+', color=folder2color[nf_folder], linewidths=1.0)
                        axs[j, cur_idx+1].plot(ypts,y_kernel(ypts),linewidth=2, linestyle = folder2linestyle[folder], color=folder2color[folder], label=folder2lgd[folder])
                        axs[j, cur_idx+1].scatter(true_y, 1e-6,s=150, marker='+', color=folder2color[nf_folder], linewidths=1.0)
                        cur_idx += 2
                    else:
                        axs[j, cur_idx].axis("off")
                        axs[j, cur_idx+1].axis("off")
                        # if step == max(step_nums) and var_name == lmk_var_names[-1]:
                        #     legend_plot.append(l)
                        #     legend_label.append(folder2lgd[folder])

                        # ax_histy.set_xlim([0, 1.1 * max(y_kernel(pts))])
                # ax_histy.invert_yaxis()
                # ax_histy.legend(prop={'size': 9})
    lines, labels = axs[len(step_nums)-1, 0].get_legend_handles_labels()
    fig.legend(lines, labels,loc="upper center", ncol=2, prop={'size': 25},bbox_to_anchor=(.78, .85) )

    cur_idx = 0
    for var_name in lmk_var_names:
        axs[len(step_nums)-1, cur_idx].set_xlabel(f"{var_name} x (m)", fontsize=f_size + 2)
        axs[len(step_nums)-1, cur_idx+1].set_xlabel(f"{var_name} y (m)", fontsize=f_size + 2)
        cur_idx += 2
    for j, step in enumerate(step_nums):
        axs[j, 0].set_ylabel(f"Step {step} KDE", fontsize=f_size + 2)
        axs[j, 0].get_yaxis().set_label_coords(-0.18, 0.5)

    for ax in axs.flat:
        ax.label_outer()
        # ax.set_aspect('equal', 'box')
    # fig.tight_layout()
    fig.savefig(f"{plot_dir}/kde_grid.png", dpi=300,bbox_inches='tight')
    # fig.show()
