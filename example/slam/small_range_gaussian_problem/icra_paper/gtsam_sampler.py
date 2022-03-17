import os
import random

import numpy as np
from typing import List, Tuple, Dict, Union, Iterable
from geometry.TwoDimension import SE2Pose, Point2
from slam.Variables import SE2Variable, VariableType, R2Variable, Variable
import matplotlib.pyplot as plt

from utils.Statistics import mmd
from utils.Visualization import plot_2d_samples


def getVars(order_file:str, pose_space: str = "SE2"):
    var_list = []
    order = np.genfromtxt(order_file, dtype='str')
    for var in order:
        if var[0] == "X":
            if pose_space == "SE2":
                var_list.append(SE2Variable(name=var,variable_type=VariableType.Pose))
            elif pose_space == "R2":
                var_list.append(R2Variable(name=var,variable_type=VariableType.Pose))
        elif var[0] == "L":
            var_list.append(R2Variable(name=var, variable_type=VariableType.Landmark))
    return var_list

def getMeans(vars:List[Variable], marg_file: str)->dict:
    res = {}
    with open(marg_file) as fp:
        for cnt, line in enumerate(fp):
            # print("Line {}: {}".format(cnt, line))
            data = line.strip().split(' ')
            data_len = len(data)
            if data_len > 2:
                if isinstance(vars[cnt], SE2Variable):
                    res[vars[cnt]] = SE2Pose.by_array([float(item) for item in data[:3]])
                elif isinstance(vars[cnt], R2Variable):
                    res[vars[cnt]] = Point2.by_array([float(item) for item in data[:2]])
                else:
                    raise TypeError("Unkonwn type.")
    return res

def getSamples(vars:List[Variable], var2mean:dict, joint: np.ndarray, sample_num:int):
    dim, _ = joint.shape
    res = np.zeros((sample_num,dim))
    for i in range(sample_num):
        noise = np.random.multivariate_normal(mean=np.zeros(dim),
                                              cov=joint)
        cur_idx = 0
        for var in vars:
            mean = var2mean[var]
            if isinstance(mean, Point2):
                res[i,cur_idx:cur_idx+2] = mean.array + noise[cur_idx:cur_idx+2]
                cur_idx += 2
            elif isinstance(mean, SE2Pose):
                noised_pose = mean * SE2Pose.by_exp_map(noise[cur_idx:cur_idx+3])
                res[i,cur_idx:cur_idx+3] = noised_pose.array
                cur_idx += 3
            else:
                raise TypeError("mean can only be SE2Pose or Point2 type.")
    return res

def reorder_samples(ref_order: List[Variable],
                    sample_order: List[Variable],
                    samples: np.ndarray):
    res = np.zeros_like(samples)
    cur_dim = 0
    for var in ref_order:
        var_idx = sample_order.index(var)
        if var_idx == 0:
            sample_dim = 0
        else:
            sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
        res[:,cur_dim:cur_dim+var.dim] = samples[:,sample_dim:sample_dim+var.dim]
        cur_dim += var.dim
    return res

if __name__ == '__main__':
    setup_folder = os.path.dirname(os.path.abspath(__file__))
    case_folder = "case1"
    gtsam_folder = "gtsam"
    reference_folder = "reference"
    compute_mmd = True
    pose_space = "SE2"
    sample_num = 500
    save_plot = False

    ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
    dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"

    batch_num = 0
    order_file = f"{dir}/batch_{batch_num}_ordering"
    marg_file = f"{dir}/batch_{batch_num}_marginal"
    joint_file = f"{dir}/batch_{batch_num}_joint"

    while(os.path.exists(order_file) and
          os.path.exists(marg_file) and
          os.path.exists(joint_file)):
        print(f"batch {batch_num}")
        nodes = getVars(order_file=order_file)
        joint = np.genfromtxt(joint_file)
        var2mean = getMeans(nodes, marg_file)
        samples = getSamples(nodes, var2mean, joint, sample_num)
        np.savetxt(fname=f"{dir}/batch{batch_num}", X=samples)

        if save_plot:
            plt.figure()
            ax = plot_2d_samples(samples_array=samples,
                                 variable_ordering=nodes,
                                 show_plot=False, equal_axis=False,legend_on=True, has_orientation=False)
            plt.savefig(f"{dir}/batch{batch_num}.png")

        if compute_mmd:
            print("Computing MMD")
            ref_order = getVars(order_file=f"{ref_dir}/step_{batch_num}_ordering")
            comp_samples = reorder_samples(ref_order=ref_order,sample_order=nodes,samples=samples)
            ref_samples = np.loadtxt(f"{ref_dir}/step_{batch_num}")
            ref_sample_num = ref_samples.shape[0]
            if ref_sample_num > sample_num:
                downsampling_indices = np.array(random.sample(list(range(ref_sample_num)), sample_num))
                ref_samples = ref_samples[downsampling_indices, :]
            else:
                downsampling_indices = np.array(random.sample(list(range(sample_num)), ref_sample_num))
                comp_samples = comp_samples[downsampling_indices, :]
            m = mmd(comp_samples, ref_samples)
            file = open(f"{dir}/batch_{batch_num}_mmd", "w+")
            file.write(str(m))
            file.close()
        batch_num += 1
        order_file = f"{dir}/batch_{batch_num}_ordering"
        marg_file = f"{dir}/batch_{batch_num}_marginal"
        joint_file = f"{dir}/batch_{batch_num}_joint"