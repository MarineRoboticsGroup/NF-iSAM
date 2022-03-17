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
    f_size = 20
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
    folder2lgd[reference_folder] = "NSFG"
    if case_folder[-2:] == "da":
        folder2lgd[gtsam_folder] = "Max-mixtures"
    else:
        folder2lgd[gtsam_folder] = "GTSAM"

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

    step_nums = [0,1,2,3,4,5]
    # step_nums = [5]

    sample_num = 1000

    color_list = ['m','darkorange','black','y','c','b','g','r']
    colors = {}
    for i, node in enumerate(nodes):
        colors[node] = color_list[i]

    fig = plt.figure(figsize=(5*len(step_nums),5*len(folders)))
    gs = fig.add_gridspec(len(folders), len(step_nums), hspace=0.05, wspace=0.08)
    axs = gs.subplots(sharex=True, sharey=True)

    for i, folder in enumerate(folders):
        for step in step_nums:
            dir = f"{setup_folder}/{case_folder}/{folder}"
            if folder == reference_folder:
                sample_file = f"{dir}/step{step}.sample"
                order_file = f"{dir}/step{step}_ordering"
            elif folder == gtsam_folder:
                sample_file = f"{dir}/step{step}"
                order_file = f"{dir}/step{step}_ordering"
            elif folder == mm_folder:
                sample_file = f"{dir}/step{step}"
                order_file = f"{dir}/step{step}_ordering"
            else:
                sample_file = f"{dir}/step{step}"
                order_file =  f"{dir}/step{step}_ordering"
            print(sample_file)
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
                ax = plot_2d_samples(ax=axs[i,step], samples_array=trimmed, variable_ordering=order,
                                     show_plot=False, equal_axis=False, colors=part_color,marker_size=.1,xlabel=None,ylabel=None,
                                     xlim=[-100, 120], ylim=[-100, 120])
                for node in order:
                    if len(truth[node]) == 2:
                        x, y = truth[node][:2]
                        color = "blue"
                        marker = "x"
                    else:
                        x, y, th = truth[node][:3]
                        color = "red"
                        marker = mpl.markers.MarkerStyle(marker='$\u2193$')  # Downwards arrow in Unicode: ↓
                        marker._transform = marker.get_transform().rotate_deg(90 + th * 180 / np.pi)
                    # ax.plot([x], [y], c=color, markersize=2,mar marker="+")
                    ax.scatter(x, y, s=50, marker=marker, color=color, linewidths=1.0)
                    # ax.scatter(x, y, s=90,facecolors='none',edgecolors='r')
                    if (node.name == 'L1'):
                        dx, dy = [0,-20]
                    elif (node.name == 'L2'):
                        dx, dy = [0,-20]
                    elif (node.name == 'X1'):
                        dx, dy = [0,10]
                    elif (node.name == 'X0'):
                        dx, dy = [-25,0]
                    elif (node.name == 'X2'):
                        dx, dy = [0, 10]
                    elif(node.name =='X5'):
                        dx, dy = [10, 0]
                    elif(node.name == 'X4'):
                        dx, dy = [0, 10]
                    elif(node.name == 'X3'):
                        dx, dy = [0, 10]
                    else:
                        dx, dy = [-15,0]
                    ax.text(x+dx, y+dy, s=node.name, fontsize=f_size - 4)
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
                axs[i,step].text(-35, 0, 'No solution', fontsize=30)

            if step == 0:
                axs[i,step].set_ylabel(folder2lgd[folder],fontsize=f_size+4)
            if i == len(folders)-1:
                axs[i,step].set_xlabel(f"Time Step {step}",fontsize=f_size+4)

            # plt.xlabel('x(m)',fontsize=f_size+2)
            # plt.ylabel('y(m)',fontsize=f_size+2)
            # plt.xlim((-10, 30))
            # plt.ylim((-8, 23))

            # plt.xlim((-15, 35))
            # plt.ylim((-12, 25))

            # if folder == reference_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_dynesty.png",bbox_inches='tight')
            # elif folder == gtsam_folder:
            #     plt.savefig(f"{plot_dir}/small_case_step{step}_GTSAM.png",bbox_inches='tight')
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
    fig.savefig(f"{plot_dir}/scatter_grid.png", dpi=300,bbox_inches="tight")
    # fig.show()
