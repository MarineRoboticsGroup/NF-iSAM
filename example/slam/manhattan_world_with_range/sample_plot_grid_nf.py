import random
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from factors.Factors import PriorFactor, KWayFactor, BinaryFactor
from slam.FactorGraphSimulator import read_factor_graph_from_file
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os

if __name__ == '__main__':

    # m_s = 12
    f_size = 16
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)
    xlim = [-50, 220]
    ylim = [-50, 220]
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

    # step_nums = [5,10,15]
    step_nums = [15]
    # step_nums = [5]

    sample_num = 500

    color_list = ['m','darkorange','black','y','c','b','g','r']
    colors = {}
    for i, node in enumerate(nodes):
        colors[node] = color_list[i%len(color_list)]

    fig = plt.figure(figsize=(5*len(folders),5*len(step_nums)))
    gs = fig.add_gridspec(len(step_nums), len(folders))#, hspace=0, wspace=0)
    axs = gs.subplots(sharex=True, sharey=True)

    for i, folder in enumerate(folders):
        for j, step in enumerate(step_nums):
            dir = f"{setup_folder}/{case_folder}/{folder}"
            if folder == reference_folder:
                sample_file = f"{dir}/step{step}.sample"
                if not os.path.exists(sample_file):
                    sample_file = f"{dir}/step0.sample"
                    order_file = f"{dir}/step0_ordering"
                else:
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

                # ax = plt.gca()
                # plt.axis("equal")
                part_color = {key:colors[key] for key in order}
                # plt.xlim([-10, 30])
                ax = plot_2d_samples(ax=axs[i], samples_array=trimmed, variable_ordering=order,
                                     show_plot=False, equal_axis=False, colors=part_color,marker_size=.1,xlabel=None,ylabel=None,
                                     xlim=xlim, ylim=ylim)
                for node in order:
                    if len(truth[node]) == 2:
                        x, y = truth[node][:2]
                        color = "blue"
                        marker = "x"
                        dx, dy = 1, 1
                        ax.text(x+dx, y+dy, s=node.name, fontsize=f_size - 4)
                    else:
                        x, y, th = truth[node][:3]
                        color = "red"
                        marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                        marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                    ax.scatter(x, y, s=50, marker=marker, color=color, linewidths=1.0)
                    # ax.plot([x], [y], c=color, markersize=2,mar marker="+")
                for factor in factors:
                    if not isinstance(factor, PriorFactor) and (set(factor.vars).issubset(set(order))):
                        if isinstance(factor, KWayFactor):
                            var1 = factor.root_var
                            var2s = factor.child_vars
                            for var2 in var2s:
                                x1, y1 = truth[var1][:2]
                                x2, y2 = truth[var2][:2]
                                ax.plot([x1, x2], [y1, y2],linestyle='--', dashes=(5, 5), c='red', linewidth=.5)
                        elif isinstance(factor, BinaryFactor):
                            var1, var2 = factor.vars
                            x1, y1 = truth[var1][:2]
                            x2, y2 = truth[var2][:2]
                            ax.plot([x1, x2], [y1, y2], c='k', linewidth=.5)

                # ax.axis("equal")
            else:
                axs[i].text(25, 60, 'No solution', fontsize=30)

            if i == 0:
                axs[i].set_ylabel(f"Step {step}",fontsize=f_size+4)
            if step == max(step_nums):
                axs[i].set_xlabel(folder2lgd[folder],fontsize=f_size+4)

            # plt.xlabel('x(m)',fontsize=f_size+2)
            # plt.ylabel('y(m)',fontsize=f_size+2)
            # plt.xlim((-10, 30))
            # plt.ylim((-8, 23))

            # plt.xlim((-15, 35))
            # plt.ylim((-12, 25))

            # if folder == reference_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_dynesty.png",bbox_inches='tight')
            # elif folder == gtsam_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_Max-mixture.png",bbox_inches='tight')
            # elif folder == nf_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_NF-iSAM.png",bbox_inches='tight')
            # elif folder == mm_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_Caesar.png",bbox_inches='tight')
            # plt.show()
            # plt.close()
    for ax in axs.flat:
        ax.label_outer()
        ax.set_aspect('equal', 'box')
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/scatter_grid2.png", dpi=300)
    # fig.show()
