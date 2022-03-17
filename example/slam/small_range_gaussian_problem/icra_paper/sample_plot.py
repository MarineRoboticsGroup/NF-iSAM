import random

import matplotlib.pyplot as plt
import numpy as np

from factors.Factors import PriorFactor
from slam.FactorGraphSimulator import read_factor_graph_from_file
from utils.Visualization import plot_2d_samples
from slam.Variables import Variable, VariableType
import os

if __name__ == '__main__':

    m_s = 12
    f_size = 22
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)
    setup_folder = os.path.dirname(os.path.abspath(__file__))
    case_folder = "case1"
    gtsam_folder = "gtsam"
    reference_folder = "reference"
    nf_folder = "run1"
    mm_folder = "mmisam"

    case_dir = f"{setup_folder}/{case_folder}"
    ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
    gtsam_dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"
    nf_dir = f"{setup_folder}/{case_folder}/{nf_folder}"

    plot_dir = f"{setup_folder}/{case_folder}/video_figures"
    if(os.path.exists(plot_dir)):
        pass
    else:
        os.mkdir(plot_dir)

    nodes, truth, factors = read_factor_graph_from_file(
        f"{case_dir}/factor_graph")

    folders = [reference_folder, nf_folder, gtsam_folder,mm_folder]
    # folders = [mm_folder]

    step_nums = [0,1,2,3,4,5]
    # step_nums = [5]

    sample_num = 500

    color_list = ['b','g','r','c','m','y','darkorange','black']
    colors = {}
    for i, node in enumerate(nodes):
        colors[node] = color_list[i]

    for folder in folders:
        for step in step_nums:
            dir = f"{setup_folder}/{case_folder}/{folder}"
            if folder == reference_folder:
                samples = np.loadtxt(fname=f"{dir}/step_{step}")
                order = Variable.file2vars(order_file=f"{dir}/step_{step}_ordering")
            elif folder == gtsam_folder:
                samples = np.loadtxt(fname=f"{dir}/batch{step}")
                order = Variable.file2vars(order_file=f"{dir}/batch_{step}_ordering")
            else:
                samples = np.loadtxt(fname=f"{dir}/batch{step+1}")
                order =  Variable.file2vars(order_file=f"{dir}/batch_{step+1}_ordering")
            if (samples.shape[0] >= sample_num):
                downsampling_indices = np.array(random.sample(list(range(samples.shape[0])), sample_num))
                trimmed = samples[downsampling_indices, :]
            else:
                raise ValueError()
                trimmed = samples
            plt.figure()
            # ax = plt.gca()
            # plt.axis("equal")
            part_color = {key:colors[key] for key in order}
            # plt.xlim([-10, 30])
            ax = plot_2d_samples(samples_array=trimmed, variable_ordering=order,
                                 show_plot=False, equal_axis=True, colors=part_color,marker_size=5)
            for node in order:
                x, y = truth[node][:2]
                color = "blue" if node.type == VariableType.Landmark else \
                    "black"
                ax.plot([x], [y], c=color, markersize=15,markeredgewidth=3, marker="+")
                # ax.scatter(x, y, s=90,facecolors='none',edgecolors='r')
                if (node.name == 'L1'):
                    dx, dy = [1,1]
                elif (node.name == 'L2'):
                    dx, dy = [-3.0, -3.0]
                elif (node.name == 'X1'):
                    dx, dy = [2,-2]
                elif(node.name =='X5'):
                    dx, dy = [-2, 2]
                elif(node.name == 'X4'):
                    dx, dy = [0, 2]
                elif(node.name == 'X3'):
                    dx, dy = [1,1]
                else:
                    dx, dy = [2,0.3]
                ax.text(x+dx, y+dy, s=node.name,fontsize=f_size-8, fontweight='bold')
            for factor in factors:
                if not isinstance(factor, PriorFactor) and (set(factor.vars).issubset(set(order))):
                    var1, var2 = factor.vars
                    x1, y1 = truth[var1][:2]
                    x2, y2 = truth[var2][:2]
                    ax.plot([x1, x2], [y1, y2], c='k', linewidth=1)
            plt.axis("equal")
            plt.xlabel('x(m)',fontsize=f_size+2)
            plt.ylabel('y(m)',fontsize=f_size+2)
            plt.xlim((-10, 30))
            plt.ylim((-8, 23))

            # plt.xlim((-15, 35))
            # plt.ylim((-12, 25))

            if folder == reference_folder:
                plt.savefig(f"{plot_dir}/small_case_step{step}_dynesty.png",bbox_inches='tight')
            elif folder == gtsam_folder:
                plt.savefig(f"{plot_dir}/small_case_step{step}_GTSAM.png",bbox_inches='tight')
            elif folder == nf_folder:
                plt.savefig(f"{plot_dir}/small_case_step{step}_NF-iSAM.png",bbox_inches='tight')
            elif folder == mm_folder:
                plt.savefig(f"{plot_dir}/small_case_step{step}_Caesar.png",bbox_inches='tight')
            plt.show()
            plt.close()