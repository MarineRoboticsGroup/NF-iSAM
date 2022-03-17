import json
import os
import random
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import linregress
from slam.RunBatch import group_nodes_factors_incrementally, incVarFactor2DRp

from geometry.TwoDimension import SE2Pose
from slam.FactorGraphSimulator import factor_graph_to_string
from slam.Variables import R2Variable, SE2Variable, VariableType
from factors.Factors import UnarySE2ApproximateGaussianPriorFactor, SE2RelativeGaussianLikelihoodFactor, \
    SE2R2RangeGaussianLikelihoodFactor, AmbiguousDataAssociationFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat, savemat

from utils.Functions import theta_to_pipi, NumpyEncoder, reject_outliers

# m_s = 12
# f_size = 24
# plt.rc('xtick', labelsize=f_size)
# plt.rc('ytick', labelsize=f_size)
from utils.Visualization import plot_2d_samples


def alignDRp(odo_path_data:np.ndarray):
    aligned_DRp = deepcopy(odo_path_data)
    for i in range(odo_path_data.shape[0]-1):
        cur_pose = SE2Pose.by_array(odo_path_data[i,1:4])
        next_pose = SE2Pose.by_array(odo_path_data[i+1,1:4])
        distance, bearing = cur_pose.range_and_bearing(next_pose._point)
        #just need to align orientation to next pose
        aligned_DRp[i, 3] = cur_pose.theta + bearing
    return aligned_DRp

def DRp2DR(DRp:np.ndarray):
    res = np.zeros((DRp.shape[0]-1, 3))
    for i in range(res.shape[0]):
        last_pose = SE2Pose.by_array(DRp[i, 1:4])
        cur_pose = SE2Pose.by_array(DRp[i+1, 1:4])
        tf = last_pose.inverse() * cur_pose
        assert abs(tf.y) < 1e-5 # no lateral motion
        res[i] = np.array([DRp[i+1, 0], tf.x, tf.theta])
    return res

def DR2DRp(DR: np.ndarray, pose0_data: np.ndarray):
    odom_dr = np.zeros((1 + len(DR), 4))
    odom_dr[0] = pose0_data[:]
    pose0 = SE2Pose(*pose0_data[1:])
    for i, one_odom in enumerate(DR):
        pose0 = pose0 * SE2Pose(one_odom[1], 0, one_odom[2])
        odom_dr[i + 1, 1:] = pose0.array[:]
    odom_dr[1:,0] = DR[:,0]
    return odom_dr

if __name__ == '__main__':
    # define the name of the directory to be created
    rd_seed = 10
    ada_prob = 0.0
    random.seed(rd_seed)
    plot_raw_data = True
    convert_mat2EFG = True
    compute_odom_err = True
    compute_range_err = True
    viz_efg_data = False
    outlier_rejection = False
    data_dir = "RangeOnlyDataset"
    raw_case = "Plaza1"
    c_case_name = f"{raw_case}_c"
    data_params = {}
    calib_file = f"{data_dir}/correction_files/{raw_case}_range_err_calib.txt"
    if os.path.exists(calib_file):
        with open(calib_file, "r") as fp:
            print("Converting JSON encoded data into Python dictionary")
            data_params = json.load(fp)

    data_path = f"{data_dir}/{c_case_name}.mat"
    if ada_prob >1e-3:
        efg_dir = f"{data_dir}/{raw_case}ADA{ada_prob}EFG"
    else:
        efg_dir = f"{data_dir}/{raw_case}EFG"
    if not os.path.exists(efg_dir):
        os.mkdir(efg_dir)
    
    heading_offset = 0.0
    if c_case_name[:6] == "Plaza2":
        heading_offset = np.pi

    pose_gt, lmk_gt, odo_data, odo_path_data, range_data = [None] * 5
    data = loadmat(data_path)
    for key in data:
        if key[-2:] == 'GT':
            pose_gt = data[key]
            if c_case_name[-2:] != '_c':
                pose_gt[:, -1] = theta_to_pipi(heading_offset + pose_gt[:, -1])
            np.savetxt(fname=f"{efg_dir}/GT.txt",X=pose_gt)
        elif key[-2:] == 'TL':
            lmk_gt = data[key]
            np.savetxt(fname=f"{efg_dir}/TL.txt",X=lmk_gt)
        elif key[-2:] == 'DR':
            odo_data = data[key]
            np.savetxt(fname=f"{efg_dir}/DR.txt",X=odo_data)
        elif key[-2:] == 'Rp':
            odo_path_data = data[key]
            np.savetxt(fname=f"{efg_dir}/DRp.txt",X=odo_path_data)
        elif key[-2:] == 'TD':
            range_data = data[key]
            np.savetxt(fname=f"{efg_dir}/TD.txt",X=range_data)
    if plot_raw_data:
        # visualize raw data
        # plot ground truth
        subplot_num = 2
        fig = plt.figure(figsize=(5,5*subplot_num))
        gs = fig.add_gridspec(subplot_num, 1)#, hspace=0, wspace=0)
        axs = gs.subplots()
        gt_ax = axs[0]
        dr_ax = axs[1]
        gt_ax.plot(pose_gt[:,1], pose_gt[:,2], color='k',label='Ground Truth')
        gt_ax.scatter(lmk_gt[:,1],lmk_gt[:,2],marker='x',s = 10.0,c='k')
        for lmk_data in lmk_gt:
            lmk_id, x, y = lmk_data
            lmk_id = int(lmk_id)
            gt_ax.text(x, y, str(lmk_id))
        gt_ax.plot(odo_path_data[:,1], odo_path_data[:,2], color='r',label='DRp data')
        gt_ax.legend()
        gt_ax.set_xlabel("x (m)")
        gt_ax.set_ylabel("y (m)")
        # if c_case_name[:6] == "Plaza2":
        #     heading_offset = np.pi
        pose0_data = odo_path_data[0,:]
        odom_dr = DR2DRp(odo_data, pose0_data)
        dr_ax.plot(odom_dr[:,1],odom_dr[:,2],label='DR data')
        dr_ax.plot(odo_path_data[:,1], odo_path_data[:,2], color='r',label='DRp data')
        dr_ax.legend()
        dr_ax.set_xlabel("x (m)")
        dr_ax.set_ylabel("y (m)")
        fig.savefig(f"{efg_dir}/raw.png", bbox_inches="tight", dpi = 300)
        fig.show()
        plt.close()

    # fg = open(case_path+'/factor_graph.fg','w+')
    # lines = factor_graph_to_string(vars,factors, var2truth)
    # fg.write(lines)
    # fg.close()

    # filtering range measurements with odometry by time stamp tolerance
    if convert_mat2EFG:
        # inc_step = 10
        data_params["rd_seed"] = rd_seed
        data_params["ada_prob"] = ada_prob

        sample_every_n_step = 1
        if c_case_name[:8] == "Gesling2":
            sample_every_n_step = 2
        elif c_case_name[:6] == "Plaza2":
            sample_every_n_step = 3
        else:
            sample_every_n_step = 4
        data_params['sample_every_n_step']=sample_every_n_step
        dx_threshold = 1e-2
        dth_threshold = 1e-3
        dy_threshold = 1e-2

        prior_noise_diag = np.array([.0001, .00001, .000001])**2
        # odo_noise_diag = np.array([0.05, 0.01, 0.2])**2
        sync_time_tol = .1
        if "sync_time_tol" in data_params:
            sync_time_tol = data_params["sync_time_tol"]
        data_params["dx_threshold"] = dx_threshold
        data_params["dth_threshold"] = dth_threshold

        lmkID2gt = {}
        lmkID2name = {}
        for i in range(lmk_gt.shape[0]):
            lmkID2gt[int(lmk_gt[i, 0])] = lmk_gt[i, 1:]
            lmkID2name[int(lmk_gt[i, 0])] = f"L{i}"

        range_times = range_data[:, 0:1]
        gt_times = pose_gt[:, 0:1]
        mytree = cKDTree(gt_times)
        dist, indexes = mytree.query(range_times)
        synced_range_indices = np.where(dist < sync_time_tol)[0]
        sync_pose_indices = indexes[synced_range_indices]
        if len(sync_pose_indices) < len(indexes):
            print("Filtered asynchronized range measurements.")

        # estimate range error
        range_err_std = None
        if compute_range_err or 'range_err_std' not in data_params:
            gt_lmks = np.array([lmkID2gt[int(i)] for i in range_data[synced_range_indices, 2]])
            gt_dist = np.linalg.norm(pose_gt[sync_pose_indices, 1:3] - gt_lmks, axis=1)
            err_dist = range_data[synced_range_indices, -1] - gt_dist

            if outlier_rejection:
                inlier_idx = reject_outliers(err_dist)
                print("Range inlier idx: ", inlier_idx)
                print("Inlier rate: ", f"{len(inlier_idx)}/{len(err_dist)}")
                mask = np.zeros(err_dist.shape[0],dtype=bool)
                mask[inlier_idx] = True
                range_err_std = np.std(err_dist[mask])
            else:
                range_err_std = np.std(err_dist)
            if 'range_err_std' in data_params:
                if abs(data_params['range_err_std'] - range_err_std) < 1e-4:
                    print("Raw range STD: ",data_params['range_err_std'],
                          ", New range STD: ",range_err_std)
            else:
                data_params['range_err_std'] = range_err_std
            print("Range STD: ", range_err_std)

            fig = plt.figure()
            axs = fig.subplots()
            axs.scatter(range_data[synced_range_indices, -1], err_dist)
            axs.set_xlabel("Measurement (m)")
            axs.set_ylabel("Error (m)")
            axs.yaxis.set_label_coords(-0.1, .5)
            fig.savefig(f"{efg_dir}/range_err.png", bbox_inches="tight", dpi=300)
            fig.show()
        else:
            range_err_std = data_params['range_err_std']

        # analyze odometry errors
        dr_tf_data = np.zeros((odo_path_data.shape[0]-1,3))
        gt_tf_data = np.zeros((odo_path_data.shape[0]-1,3))
        err_tf_data = np.zeros((odo_path_data.shape[0]-1,3))
        added_range_data = []
        step_range_num = []
        last_dr_pose = None
        last_gt_pose = None
        for i in range(0, odo_path_data.shape[0]):
            cur_gt_pose = SE2Pose(*pose_gt[i, 1:])
            cur_dr_pose = SE2Pose(*odo_path_data[i, 1:])
            if i > 0:
                cur_dr_pose = SE2Pose(*odo_path_data[i,1:])
                cur_gt_pose = SE2Pose(*pose_gt[i,1:])
                cur_dr_time = odo_path_data[i, 0]
                assert abs(cur_dr_time - pose_gt[i, 0]) < 1e-4
                dr_tf_pose = last_dr_pose.inverse() * cur_dr_pose
                dr_tf_data[i-1] = dr_tf_pose.array
                gt_tf_pose = last_gt_pose.inverse() * cur_gt_pose
                gt_tf_data[i - 1] = gt_tf_pose.array
                err_tf_pose = gt_tf_pose.inverse() * dr_tf_pose
                err_tf_data[i - 1] = err_tf_pose.log_map()
            range_idx = synced_range_indices[np.where(sync_pose_indices == i)[0]]
            step_range_num.append([i, len(range_idx)])
            for id in range_idx:
                cur_lmk_id, cur_range = range_data[id, 2:4]
                cur_lmk_id = int(cur_lmk_id)
                lmk_gt = lmkID2gt[cur_lmk_id]
                lmk_alias = int(lmkID2name[cur_lmk_id][1:])
                range_gt = np.linalg.norm(cur_gt_pose.array[:2] - lmk_gt)
                range_err = cur_range - range_gt
                added_range_data.append([i, lmk_alias, cur_range, range_err, id])
            last_dr_pose = cur_dr_pose
            last_gt_pose = cur_gt_pose

        odom_noise_cov = None
        if compute_odom_err or 'odom_noise_cov' not in data_params:
            fig = plt.figure()
            axs = fig.subplots(3,1)
            labels = ["dx (m)", "dy (m)", "dth (rad)"]
            for i in range(3):
                axs[i].plot(gt_tf_data[:,i], label='GT')
                axs[i].plot(dr_tf_data[:,i], label='DR')
                axs[i].set_ylabel(labels[i])
                axs[i].set_xlabel("Step")
                axs[i].yaxis.set_label_coords(-0.1, .5)
                axs[i].legend()
            fig.savefig(f"{efg_dir}/tf_cmp.png", bbox_inches="tight", dpi=300)
            fig.show()

            fig = plt.figure()
            axs = fig.subplots(3,1)
            labels = ["ddx (m)", "ddy (m)", "ddth (rad)"]
            for i in range(3):
                axs[i].plot(dr_tf_data[:,i]-gt_tf_data[:,i], label='DR')
                axs[i].set_ylabel(labels[i])
                axs[i].set_xlabel("Step")
                axs[i].yaxis.set_label_coords(-0.1, .5)
                axs[i].legend()
            fig.savefig(f"{efg_dir}/tf_diff.png", bbox_inches="tight", dpi=300)
            fig.show()

            fig = plt.figure()
            axs = fig.subplots(3,1)
            labels = ["logdP_x", "logdP_y", "logdP_th"]
            for i in range(3):
                axs[i].scatter(gt_tf_data[:,i], err_tf_data[:,i], label='Correlation')
                axs[i].set_ylabel(labels[i])
                axs[i].set_xlabel("Ground Truth")
                axs[i].yaxis.set_label_coords(-0.1, .5)
                axs[i].legend()
            fig.savefig(f"{efg_dir}/tf_err_col.png", bbox_inches="tight", dpi=300)
            fig.show()

            fig = plt.figure()
            axs = fig.subplots(3,1)
            labels = ["logdP_x", "logdP_y", "logdP_th"]
            for i in range(3):
                axs[i].plot(err_tf_data[:,i], label='err')
                axs[i].set_ylabel(labels[i])
                axs[i].set_xlabel("Step")
                axs[i].yaxis.set_label_coords(-0.1, .5)
                axs[i].legend()
            fig.savefig(f"{efg_dir}/tf_err.png", bbox_inches="tight", dpi=300)
            fig.show()

            if outlier_rejection:
                dx_in = reject_outliers(err_tf_data[:,0])
                dy_in = reject_outliers(err_tf_data[:,1])
                dth_in = reject_outliers(err_tf_data[:,2])
                inlier_idx = list(set(list(dx_in)+list(dy_in)+list(dth_in)))
                print("TF inlier idx: ", inlier_idx)
                print("Inlier rate: ", f"{len(inlier_idx)}/{len(err_tf_data)}")
                mask = np.zeros(err_tf_data.shape[0],dtype=bool)
                mask[inlier_idx] = True
                odom_noise_cov = np.cov(err_tf_data[mask], rowvar=False)
            else:
                odom_noise_cov = np.cov(err_tf_data, rowvar=False)
            np.savetxt(fname=f"{efg_dir}/err_cov.txt",X=odom_noise_cov)
            data_params['odom_noise_cov'] = odom_noise_cov
            
            tmp_range_data = np.array(added_range_data)
            tmp_step_range_num = np.array(step_range_num)
            fig = plt.figure()
            axs = fig.subplots(3,1)
            axs[0].plot(tmp_range_data[:,0], tmp_range_data[:,3])
            axs[0].set_ylabel("Range Error (m)")
            axs[1].plot(tmp_range_data[:,0], tmp_range_data[:,1])
            axs[1].set_ylabel("LMK ID")
            axs[2].plot(tmp_step_range_num[:,0], tmp_step_range_num[:,1])
            axs[2].set_ylabel("Range Meas. #")
            fig.savefig(f"{efg_dir}/range_err.png", bbox_inches="tight", dpi=300)
            fig.show()
            print("admitted range measurements: ", len(added_range_data))
        else:
            odom_noise_cov = data_params['odom_noise_cov']

        # problem initialization
        cur_pose_id = 0
        rbtvars = []
        lmkvars = []
        factors = []
        var2truth = {}
        acc_odo = SE2Pose()
        acc_odo_num = 0
        range_std = range_err_std
        pose2ranges = {}
        skipped_poses = None
        data_params['range_std'] = range_std
        with open(f"{efg_dir}/data_params", 'w') as fp:
            json.dump(data_params, fp, cls=NumpyEncoder)
        for i in range(pose_gt.shape[0]):
            if i == 0:
                cur_pose_node = SE2Variable(f"X{cur_pose_id}")
                cur_pose_id += 1
                prior_pose = SE2Pose(x=pose_gt[i, 1], y=pose_gt[i, 2], theta=pose_gt[i, 3])
                rbtvars.append(cur_pose_node)
                var2truth[cur_pose_node] = prior_pose.array
                factors.append(UnarySE2ApproximateGaussianPriorFactor(var=cur_pose_node,
                                                                      prior_pose=prior_pose,
                                                                      covariance=np.diag(prior_noise_diag)))
            else:
                # accumulate odom measurement and skip stationary poses
                dx, dy, dth = dr_tf_data[i-1,:]
                if (dx < dx_threshold and dy < dy_threshold and dth < dth_threshold):
                    pass
                else:
                    acc_odo = acc_odo * SE2Pose(dx, dy, dth)
                    acc_odo_num += 1

            # there are range measurements for pose i
            if i in sync_pose_indices:
                add_range = False
                if skipped_poses is None:
                    add_range = True
                    skipped_poses = 0
                elif skipped_poses >= sample_every_n_step - 1:
                    add_range = True
                    skipped_poses = 0
                else:
                    skipped_poses += 1

                if add_range:
                    range_idx = synced_range_indices[np.where(sync_pose_indices == i)[0]]
                    if acc_odo_num == 0:
                        # the robot has not moved since last pose being added
                        cur_pose_node = rbtvars[-1]
                    else:
                        #process range measurements on previous node
                        if rbtvars[-1] in pose2ranges:
                            lmk2ranges = pose2ranges[rbtvars[-1]]
                            add_ada = False
                            if len(lmk2ranges) == 1:
                                tmp_node = R2Variable(lmkID2name[list(lmk2ranges.keys())[0]], variable_type=VariableType.Landmark)
                                if tmp_node in lmkvars and random.random()<ada_prob:
                                    add_ada = True
                            # odd = random.random()
                            # lmk_set = set(lmk_vars)
                            # if odd < self._args.outlier_prob:
                            #     if var not in lmk_set:
                            #         lmk_vars.append(var)
                            #         var2truth[var] = np.array([lmk_pt.x, lmk_pt.y])
                            #         if ax is not None:
                            #             plot_point(ax, point=lmk_pt, marker_size=40, color='blue', label=lmk.name,
                            #                        label_offset=(3, 3))
                            #     outlier_r = noisy_r + self._args.outlier_scale * rbt._range_std
                            #     factors.append(BinaryFactorWithNullHypo(var1=rbt_var,
                            #                                             var2=var,
                            #                                             weights=self._args.outlier_weights,
                            #                                             binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                            #                                             observation=outlier_r,
                            #                                             sigma=r_sigma,
                            #                                             null_sigma_scale=self._args.outlier_scale))
                            # elif odd < self._args.outlier_prob + self._args.ada_prob and var in lmk_set and len(
                            #         lmk_vars) > 1:
                            #     lmk_num = len(lmk_vars)
                            #     create_da = False
                            #     if only_one_da and not has_da:
                            #         create_da = True
                            #     elif not only_one_da:
                            #         create_da = True
                            #     if create_da:
                            #         factors.append(AmbiguousDataAssociationFactor(observer_var=rbt_var,
                            #                                                       observed_vars=lmk_vars,
                            #                                                       weights=np.ones(lmk_num) / lmk_num,
                            #                                                       binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                            #                                                       observation=noisy_r,
                            #                                                       sigma=r_sigma))
                            for lmkID in lmkID2name:
                                if lmkID in lmk2ranges:
                                    ranges = lmk2ranges[lmkID]
                                    cur_lmk_node = R2Variable(lmkID2name[lmkID], variable_type=VariableType.Landmark)
                                    if cur_lmk_node not in lmkvars:
                                        lmkvars.append(cur_lmk_node)
                                        var2truth[cur_lmk_node] = lmkID2gt[lmkID]
                                    if add_ada:
                                        observed = [cur_lmk_node] + list(set(lmkvars) - {cur_lmk_node})
                                        range_factor = AmbiguousDataAssociationFactor(observer_var=rbtvars[-1],
                                                                       observed_vars=observed,
                                                                       weights=np.ones(len(observed)) / len(observed),
                                                                       binary_factor_class=SE2R2RangeGaussianLikelihoodFactor,
                                                                       observation=np.mean(ranges),
                                                                       sigma=range_std)
                                        print("add range factor between ", f"{rbtvars[-1].name}",
                                              " to ", " ".join([var.name for var in observed]))
                                    else:
                                        range_factor = SE2R2RangeGaussianLikelihoodFactor(
                                            var1=rbtvars[-1],
                                            var2=cur_lmk_node,
                                            observation=np.mean(ranges),
                                            sigma=range_std)
                                        print("add range factor between ", f"{rbtvars[-1].name}",
                                              " to ", f"{cur_lmk_node.name}.")
                                    factors.append(range_factor)
                            pose2ranges = {}

                        cur_pose_node = SE2Variable(f"X{cur_pose_id}")
                        var2truth[cur_pose_node] = pose_gt[i, 1:4]
                        odo_factor = SE2RelativeGaussianLikelihoodFactor(
                            var1=rbtvars[-1],
                            var2=cur_pose_node,
                            observation=acc_odo,
                            covariance=acc_odo_num*odom_noise_cov)
                        rbtvars.append(cur_pose_node)
                        factors.append(odo_factor)
                        cur_pose_id += 1
                        acc_odo_num = 0
                        acc_odo = SE2Pose()
                    for id in range_idx:
                        cur_lmk_id, cur_range = range_data[id, 2:4]
                        cur_lmk_id = int(cur_lmk_id)
                        if cur_pose_node not in pose2ranges:
                            pose2ranges[cur_pose_node] = {}
                        if cur_lmk_id not in pose2ranges[cur_pose_node]:
                            pose2ranges[cur_pose_node][cur_lmk_id] = []
                        pose2ranges[cur_pose_node][cur_lmk_id].append(cur_range)
        assert len(pose2ranges) < 2
        for pose_var in pose2ranges:
            lmk2ranges = pose2ranges[pose_var]
            for lmkID in lmkID2name:
                if lmkID in lmk2ranges:
                    ranges = lmk2ranges[lmkID]
                    cur_lmk_node = R2Variable(lmkID2name[lmkID], variable_type=VariableType.Landmark)
                    if cur_lmk_node not in lmkvars:
                        lmkvars.append(cur_lmk_node)
                        var2truth[cur_lmk_node] = lmkID2gt[lmkID]
                    range_factor = SE2R2RangeGaussianLikelihoodFactor(
                        var1=pose_var,
                        var2=cur_lmk_node,
                        observation=np.mean(ranges),
                        sigma=range_std)
                    factors.append(range_factor)
                    print("add range factor between ", f"{pose_var.name}",
                          " to ", f"{cur_lmk_node.name}.")
        vars = rbtvars + lmkvars
        fg = open(f"{efg_dir}/factor_graph.fg", 'w+')
        lines = factor_graph_to_string(vars, factors, var2truth)
        fg.write(lines)
        fg.close()

        #viz exported EFG
        if viz_efg_data:
            fig_dir = f"{efg_dir}/plot_gt"
            if not os.path.exists(fig_dir):
                os.mkdir(fig_dir)
            nodes_factors_by_step = group_nodes_factors_incrementally(
                nodes=vars, factors=factors, incremental_step=1)
            odom_x, odom_y = incVarFactor2DRp(nodes_factors_by_step)
            added_vars = []
            added_arr = []
            for step in range(len(nodes_factors_by_step)):
                step_nodes, step_factors = nodes_factors_by_step[step]
                fig, ax = plt.subplots()
                ax.plot(odom_x, odom_y, '-', c='0.8')
                added_vars += step_nodes
                added_arr += [var2truth[var] for var in step_nodes]
                step_file_prefix = f"{fig_dir}/step{step}"
                plot_2d_samples(ax=ax, samples_array=np.array([np.hstack((added_arr))]),variable_ordering=added_vars,
                                show_plot=False, equal_axis=False,
                                # truth={variable: pose for variable, pose in
                                #        truth.items() if variable in order},
                                # truth_factors={factor for factor in cur_factors},
                                truth=var2truth,
                                truth_factors=factors,
                                file_name=f"{step_file_prefix}.png", title=f'Ground Truth (step {step})',
                                plot_all_meas=False,
                                plot_meas_give_pose=[var for var in step_nodes if var.type == VariableType.Pose],
                                rbt_traj_no_samples=True,
                                truth_R2=True,
                                truth_SE2=False,
                                truth_odometry_color='k',
                                truth_landmark_markersize=15)
                plt.close()
            # if (abs(odo_dx) > dx_threshold or abs(odo_dth) > dth_threshold):
            #     odo_pose = SE2Pose(x=odo_dx, y=0, theta=odo_dth)
            #     acc_odo = acc_odo * odo_pose
            #     tmp_pose_cnt += 1
            #     if tmp_pose_cnt == sample_every_n_step:
            #         last_pose_node = SE2Variable(f"X{cur_pose_id}")
            #         cur_pose_id += 1
            #         cur_pose_node = SE2Variable(f"X{cur_pose_id}")
            #         model.add_node(cur_pose_node)
            #         odo_factor = SE2RelativeGaussianLikelihoodFactor(
            #             var1=last_pose_node,
            #             var2=cur_pose_node,
            #             observation=acc_odo,
            #             covariance=np.diag(odo_noise_diag))
            #         model.add_factor(odo_factor)
            #
            #         if add_range:
            #             while (range_cnt < range_data.shape[0]):
            #                 if cur_dr_time >= range_data[range_cnt,0]:
            #                     range_cnt += 1
            #                     if abs(cur_dr_time - range_data[range_cnt,0])<sync_time_tol:
            #                         cur_lmk_id, cur_range = range_data[range_cnt, 2:4]
            #                         cur_lmk_id = int(cur_lmk_id)
            #                         cur_lmk_node = R2Variable(f"L{cur_lmk_id}", variable_type=VariableType.Landmark)
            #                         if cur_lmk_node not in lmk_node_list:
            #                             model.add_node(cur_lmk_node)
            #                             lmk_node_list.append(cur_lmk_node)
            #                         if cur_range > range_err_std:
            #                             sigma = range_err_std
            #                         else:
            #                             sigma = cur_range/3.0
            #                         range_factor = SE2R2RangeGaussianLikelihoodFactor(
            #                             var1=cur_pose_node,
            #                             var2=cur_lmk_node,
            #                             observation=cur_range,
            #                             sigma=sigma)
            #                         model.add_factor(range_factor)
            #                         print("add range factor between ",f"{cur_pose_node.name}",
            #                               " to ", f"{cur_lmk_node.name}.")
            #                 else:
            #                     break
            #
            #         if cur_pose_id % inc_step == 0:
            #             step_list.append(i)
            #             step_file_prefix = f"{run_dir}/step{i}"
            #             detailed_timer = []
            #             clique_dim_timer = []
            #             start = time.time()
            #             model.update_physical_and_working_graphs(timer=detailed_timer)
            #             cur_sample = model.incremental_inference(timer=detailed_timer)
            #             end = time.time()
            #
            #             step_timer.append(end - start)
            #             print(f"step {i}/{step_num} time: {step_timer[-1]} sec, "
            #                   f"total time: {sum(step_timer)}")
            #
            #             file = open(f"{step_file_prefix}_ordering", "w+")
            #             file.write(" ".join([var.name for var in model.elimination_ordering]))
            #             file.close()
            #
            #             file = open(f"{step_file_prefix}_split_timing", "w+")
            #             file.write(" ".join([str(t) for t in detailed_timer]))
            #             file.close()
            #
            #             posterior_sampling_timer.append(detailed_timer[-1])
            #             fitting_timer.append(sum(detailed_timer[1:-1]))
            #             # if cur_pose_id % (inc_step * 4) == 0:
            #             #     X = np.hstack([cur_sample[var] for var in model.physical_vars])
            #             #     np.savetxt(fname=step_file_prefix, X=X)
            #
            #             X = np.hstack([cur_sample[var] for var in model.elimination_ordering])
            #             np.savetxt(fname=step_file_prefix, X=X)
            #
            #             #clique dim and timing
            #             np.savetxt(fname=step_file_prefix+'_dim_time',X=np.array(clique_dim_timer))
            #
            #             model.plot2d_mean_rbt_only(title=f"step {i} posterior",if_legend=True,fname=f"{step_file_prefix}.png",front_size=f_size)
            #             file = open(f"{run_dir}/step_timing", "w+")
            #             file.write(" ".join(str(t) for t in step_timer))
            #             file.close()
            #             file = open(f"{run_dir}/step_list", "w+")
            #             file.write(" ".join(str(s) for s in step_list))
            #             file.close()
            #
            #             plt.figure()
            #             plt.plot(step_list, step_timer, 'go-', label='Total')
            #             plt.plot(step_list, posterior_sampling_timer, 'ro-', label='Posterior sampling')
            #             plt.plot(step_list, fitting_timer, 'bd-', label='Fitting flows')
            #             # axes[0].set_xlabel("Step", fontsize=f_size)
            #             plt.ylabel(f"Time (sec)")
            #             plt.legend()
            #
            #             plt.savefig(f"{run_dir}/step_timing.png", bbox_inches="tight")
            #             plt.show()
            #             plt.close()
            #
            #             # plt.figure()
            #             # plt.plot(step_list, step_timer,'go-')
            #             # plt.xlabel("Step",fontsize=f_size)
            #             # plt.ylabel(f"Time for {inc_step * sample_every_n_step} steps (sec)",fontsize=f_size)
            #             # plt.savefig(f"{run_dir}/step_timing.png",bbox_inches="tight")
            #             # plt.show()
            #         tmp_pose_cnt = 0
            #         acc_odo = SE2Pose()
            #     # if i < 10:
            #     #     plot_2d_samples(samples_mapping=cur_sample, show_plot=True,
            #     #                     title=f"step {i} posterior",
            #     #                     has_orientation=False, legend_on=True,
            #     #                     equal_axis=False,
            #     #                     file_name=f"{step_file_prefix}.png"
            #     #                     )
            #     #     plt.close()
            #     # elif i < 550:
            #     #     plot_2d_samples(samples_mapping=cur_sample, show_plot=True,
            #     #                     title=f"step {i} posterior",
            #     #                     has_orientation=False, legend_on=False,
            #     #                     equal_axis=False,
            #     #                     file_name=f"{step_file_prefix}.png"
            #     #                     )
            #     #     plt.close()
            #     # else:
            #     #     plt.figure()
            #     #     model.plot2d_mean_points(title=f"step {i} mean points")
            #     #     plt.savefig(f"{step_file_prefix}_mean.png")