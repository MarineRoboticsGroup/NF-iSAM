import numpy as np
import os

if __name__ == '__main__':
    case_folder = "journal_paper"
    # case_folder = "/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/manhattan_world_with_range/random_4x4/res"
    case_list = [case_folder + '/' + dir for dir in os.listdir(case_folder) if os.path.isdir(case_folder + '/' + dir)]
    caesar_folder = "caesar1"
    for case_dir in case_list:
        caesar_dir = f"{case_dir}/{caesar_folder}"
        if os.path.exists(caesar_dir):
            cur_idx = 0
            sample_file = f"{caesar_dir}/step{cur_idx}.npz"
            while os.path.exists(sample_file):
                res = np.load(sample_file)
                np.savetxt(f'{caesar_dir}/step{cur_idx}', X=res.T)
                cur_idx += 1
                sample_file = f"{caesar_dir}/step{cur_idx}.npz"
            time_file = f"{caesar_dir}/timing.npz"
            if os.path.exists(time_file):
                res = np.load(time_file)
                np.savetxt(f'{caesar_dir}/timing', X=res)