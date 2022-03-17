import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    setup_folder = os.path.dirname(os.path.abspath(__file__))
    case_folder = "case1"
    gtsam_folder = "gtsam"
    nf_folder = "run1"
    mmisam_folder = "mmisam"

    m_s = 12
    f_size = 20
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    gtsam_dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"
    mmisam_dir = f"{setup_folder}/{case_folder}/{mmisam_folder}"
    nf_dir = f"{setup_folder}/{case_folder}/{nf_folder}"

    gtsam_time = np.loadtxt(f"{gtsam_dir}/mmd").flatten()
    nf_time = np.loadtxt(f"{nf_dir}/mmd").flatten()
    plt.figure()
    plt.plot(nf_time, "-o", markersize=m_s, c='#1f77b4')
    plt.plot(gtsam_time, "-x", markersize=m_s, c='#ff7f0e')
    plt.plot(np.loadtxt(f"{nf_dir}/marginal_mmd").flatten(), "--o", markersize=m_s, c='#1f77b4')
    plt.plot(np.loadtxt(f"{mmisam_dir}/marginal_mmd").flatten(), "--s", markersize=m_s, c='#2ca02c')
    plt.plot(np.loadtxt(f"{gtsam_dir}/marginal_mmd").flatten(), "--x", markersize=m_s, c='#ff7f0e')
    plt.legend(['NF-iSAM (joint)', 'GTSAM (joint)', 'NF-iSAM (marginal)', 'Caesar.jl (marginal)', 'GTSAM (marginal)'], fontsize=f_size - 7)
    ax = plt.gca()
    ax.set_yscale('log')
    plt.xlim((-0.5, 5.5))
    plt.ylim((0.005, 1.3))
    plt.tight_layout()
    plt.xlabel('Step', fontsize=f_size)
    plt.ylabel('Maximum mean discrepancy', fontsize=f_size)
    plt.savefig(f"{setup_folder}/{case_folder}/small_case_mmd.png",dpi =300, bbox_inches='tight')
    plt.show()