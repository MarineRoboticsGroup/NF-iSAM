import matplotlib.pyplot as plt
import numpy as np
import os


if __name__ == '__main__':
    setup_folder = os.path.dirname(os.path.abspath(__file__))
    case_folder = "case1"
    gtsam_folder = "gtsam"
    reference_folder = "reference"
    nf_folder = "run1"
    mm_folder = "mmisam"

    m_s = 12
    f_size = 20
    plt.rc('xtick', labelsize=f_size)
    plt.rc('ytick', labelsize=f_size)

    ref_dir = f"{setup_folder}/{case_folder}/{reference_folder}"
    gtsam_dir = f"{setup_folder}/{case_folder}/{gtsam_folder}"
    nf_dir = f"{setup_folder}/{case_folder}/{nf_folder}"
    mm_dir = f"{setup_folder}/{case_folder}/{mm_folder}"

    gtsam_time = np.loadtxt(f"{gtsam_dir}/timing").flatten()
    ref_time = np.loadtxt(f"{ref_dir}/timing").flatten()
    nf_time = np.loadtxt(f"{nf_dir}/timing").flatten()
    mm_time = np.loadtxt(f"{mm_dir}/timing").flatten()
    plt.figure()
    log_time = True
    if log_time:
        plt.semilogy(nf_time, "-o",markersize=m_s)
        plt.semilogy(gtsam_time, "-x",markersize=m_s)
        plt.semilogy(mm_time, "-+",markersize=m_s)
        plt.semilogy(ref_time, "-^",markersize=m_s)
    else:
        plt.plot(nf_time, "-o",markersize=m_s)
        plt.plot(gtsam_time, "-x",markersize=m_s)
        plt.plot(mm_time, "-+",markersize=m_s)
        plt.plot(ref_time, "-^",markersize=m_s)
    plt.legend(['NF-iSAM (Python)','GTSAM (C++)','Caesar.jl (Julia)','dynesty (Python)'],fontsize=f_size-4,bbox_to_anchor=(0.6, 0.15),loc='lower center')
    plt.tight_layout()
    plt.xlabel('Step', fontsize=f_size)
    if log_time:
        plt.ylabel('Time (sec)', fontsize=f_size)
    else:
        plt.ylabel('Time (sec)', fontsize=f_size)
    # plt.axis("equal")
    plt.savefig(f"{setup_folder}/{case_folder}/small_case_timing.png", dpi=300, bbox_inches='tight')
    plt.show()