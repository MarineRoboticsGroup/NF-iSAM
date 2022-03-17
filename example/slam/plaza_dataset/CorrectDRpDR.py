import json
import os
import time
import numpy as np

from geometry.TwoDimension import SE2Pose
from slam.Variables import R2Variable, SE2Variable, VariableType
from factors.Factors import UnarySE2ApproximateGaussianPriorFactor, SE2RelativeGaussianLikelihoodFactor, \
    SE2R2RangeGaussianLikelihoodFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt
from copy import deepcopy
from scipy.io import loadmat, savemat
from scipy.spatial import cKDTree
from scipy.stats import linregress, norm
from utils.Functions import theta_to_pipi
import json

# m_s = 12
# f_size = 24
# plt.rc('xtick', labelsize=f_size)
# plt.rc('ytick', labelsize=f_size)

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
    plot_raw_data = False
    correct_data = True
    correct_range_data = True
    data_dir = "RangeOnlyDataset"
    case_name = "Plaza1"
    prefix = ''
    data_path = f"{data_dir}/{case_name}.mat"
    correct_dir = f"{data_dir}/correction_files"
    if not os.path.exists(correct_dir):
        os.mkdir(correct_dir)

    data = loadmat(data_path)
    pose_gt = data[prefix+'GT'] # time (sec), x (m), y (m), heading (theta)
    lmk_gt = data[prefix+'TL'] # lmk ID, x (m), y (m)
    odo_data = data[prefix+'DR'] # time (sec), dx (x), dtheta (rad)
    odo_path_data = data[prefix+'DRp'] # time (sec), x (m), y (m), heading (rad)
    range_data = data[prefix+'TD'] # time (sec), receiver ID (useless), lmk ID, range (m)

    gt_heading_offset = 0.0
    if case_name[:6] == 'Plaza2':
        gt_heading_offset = np.pi

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

        #odom trajectory
        # if case_name[:6] == "Plaza2":
        #     heading_offset = np.pi
        pose0_data = odo_path_data[0,:]
        odom_dr = DR2DRp(odo_data, pose0_data)
        dr_ax.plot(odom_dr[:,1],odom_dr[:,2],label='DR data')
        dr_ax.plot(odo_path_data[:,1], odo_path_data[:,2], color='r',label='DRp data')
        dr_ax.legend()
        fig.savefig(f"{correct_dir}/{case_name}_raw.png", bbox_inches="tight", dpi = 300)
        fig.show()
        plt.close()

    if correct_data:
        # correct ground truth heading
        pose_gt[:,-1] += gt_heading_offset
        pose_gt[:,-1] = theta_to_pipi(pose_gt[:,-1])

        # correct odometry data
        DRpbyGT = alignDRp(pose_gt)
        DRbyGT = DRp2DR(DRpbyGT)
        gt_dr = DR2DRp(DRbyGT, DRpbyGT[0,:])

        DRp_c = alignDRp(odo_path_data)
        DR = DRp2DR(DRp_c)
        pose0_data = DRp_c[0,:]
        odom_dr = DR2DRp(DR, pose0_data)
        fig = plt.figure()
        axs = fig.subplots(2,1)
        axs[0].plot(gt_dr[:,1],gt_dr[:,2],label='GT_DRp')
        axs[0].plot(odom_dr[:,1],odom_dr[:,2],label='DRp2DR')
        axs[0].plot(DRp_c[:,1],DRp_c[:,2],label='DRpc')
        axs[0].plot(odo_path_data[:,1],odo_path_data[:,2],label='DRp')
        axs[0].legend()
        axs[1].plot(theta_to_pipi(odom_dr[:,3]), label="DRpc2DR_th")
        axs[1].plot(theta_to_pipi(DRp_c[:,3]), label="DRpc_th")
        axs[1].plot(theta_to_pipi(odo_path_data[:,3]), label="DRp_th")
        axs[1].legend()
        fig.savefig(f"{correct_dir}/{case_name}_DRp2DR.png", bbox_inches="tight", dpi = 300)
        fig.show()
        plt.close()

        if correct_range_data:
            sync_time_tol = .1
            # correct range data
            lmkID2gt = {}
            for i in range(lmk_gt.shape[0]):
                lmkID2gt[int(lmk_gt[i,0])] = lmk_gt[i,1:]
            added_range_data = []

            range_times = range_data[:,0:1]
            gt_times = pose_gt[:,0:1]
            mytree = cKDTree(gt_times)
            dist, indexes = mytree.query(range_times)
            synced_range_indices = np.where(dist < sync_time_tol)[0]
            sync_pose_indices = indexes[synced_range_indices]
            if len(sync_pose_indices) < len(indexes):
                print("Filtered asynchronized range measurements: ", list(set(np.arange(len(dist))) - set(synced_range_indices)))

            gt_lmks = np.array([lmkID2gt[int(i)] for i in range_data[synced_range_indices,2]])
            gt_dist = np.linalg.norm(pose_gt[sync_pose_indices, 1:3] - gt_lmks, axis=1)
            err_dist = range_data[synced_range_indices,-1] - gt_dist
            # slope, intercept, r_value, p_value, std_err = linregress(range_data[:,-1],err_dist)
            slope, intercept, r_value, p_value, std_err = linregress(gt_dist,err_dist)
            range_bias = gt_dist * slope + intercept
            calib_err = err_dist-range_bias
            range_err_mu, range_err_std = norm.fit(calib_err)
            range_err_calib = {"slope": slope,"intercept": intercept, "r_value": r_value,"p_value": p_value,
                               "std_err": std_err,"range_err_mu": range_err_mu,"range_err_std":range_err_std,
                               "sync_time_tol": sync_time_tol}
            with open(f"{correct_dir}/{case_name}_range_err_calib.txt", 'w') as fp:
                json.dump(range_err_calib, fp)
            print(range_err_calib)

            fig = plt.figure(figsize=(5,10))
            axs = fig.subplots(2)
            fig.subplots_adjust(wspace=.1, hspace=.2)
            axs[0].scatter(gt_dist, err_dist,c='blue', label='Raw Data')
            tmp_x = np.arange(0, max(gt_dist), .1)
            axs[0].plot(tmp_x, tmp_x * slope + intercept,color='red', label="Least Squares")
            axs[0].legend()
            axs[0].set_xlabel("True Distance (m)")
            axs[0].set_ylabel("Error (m)")
            axs[0].yaxis.set_label_coords(-0.1, .5)
            f_s, f_i, f_r = "{:.3f}".format(slope), "{:.3f}".format(intercept), "{:.3f}".format(r_value)
            axs[0].text(np.mean(gt_dist), .5, f"Slope: {f_s}")
            axs[0].text(np.mean(gt_dist), .0, f"Intercept: {f_i}")
            axs[1].hist(calib_err,label="Calibrated Data",color='red', bins=50,density=True)
            xmin, xmax = axs[1].get_xlim()
            x = np.linspace(xmin, xmax, 100)
            p = norm.pdf(x, range_err_mu, range_err_std)
            format_mu = "{:.2f}".format(range_err_mu)
            format_std = "{:.2f}".format(range_err_std)
            axs[1].plot(x, p, 'k', linewidth=2, label=f"Fitted PDF N({format_mu},{format_std})")
            axs[1].set_xlabel("Calibrated Error (m)")
            axs[1].set_ylabel("Density")
            axs[1].yaxis.set_label_coords(-0.1, .5)
            axs[1].legend()
            fig.savefig(f"{correct_dir}/{case_name}_range_err.png",bbox_inches="tight", dpi=300)
            fig.show()
            range_data[synced_range_indices,-1] = range_data[synced_range_indices,-1] - range_bias
            range_data = range_data[synced_range_indices]

        corrected_data = {}
        corrected_data[prefix + 'GT'] = pose_gt  # time (sec), x (m), y (m), heading (theta)
        corrected_data[prefix + 'TL'] = lmk_gt
        corrected_data[prefix + 'DR'] = DR
        corrected_data[prefix + 'DRp'] = DRp_c
        corrected_data[prefix + 'TD'] = range_data
        savemat(f"{data_dir}/{case_name}_c.mat", corrected_data)