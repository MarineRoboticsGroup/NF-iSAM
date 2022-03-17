import os
import time

import numpy as np
from slam.Variables import SE2Variable
from factors.Factors import SE2RelativeGaussianLikelihoodFactor, UnarySE2ApproximateGaussianPriorFactor
from slam.NFiSAM import NFiSAM
import matplotlib.pyplot as plt
import multiprocessing as mp
from geometry.TwoDimension import SE2Pose
from utils.Visualization import plot_2d_samples


def plot_data(x,i=0,j=1, **kwargs):
    plt.scatter(x[:,i], x[:,j], marker=".", **kwargs)

if __name__ == '__main__':

    #this is an example similar to Pose2SLAMExample of GTSAM, p13
    #https://borg.cc.gatech.edu/sites/edu.borg/files/downloads/gtsam.pdf

    # define the name of the directory to be created
    path = "five_pose_incremental"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    node_x1 = SE2Variable('x1')
    node_x2 = SE2Variable('x2')
    node_x3 = SE2Variable('x3')
    node_x4 = SE2Variable('x4')
    node_x5 = SE2Variable('x5')


    #results with low theta std
    move = 2
    correlated_R_t = True
    prior_noise = np.diag([0.3**2,0.3**2,0.1**2])
    odometry_noise = np.diag([0.2**2,0.2**2,0.1**2])

    prior_factor_x1 = UnarySE2ApproximateGaussianPriorFactor(
        var=node_x1,
        prior_pose = SE2Pose(x=0, y=0, theta=0),
        covariance = prior_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x1_x2 =  SE2RelativeGaussianLikelihoodFactor(
        var1 = node_x1,
        var2 = node_x2,
        observation = SE2Pose(x=move, y=0, theta=0),
        covariance = odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x2_x3 =  SE2RelativeGaussianLikelihoodFactor(
        var1 = node_x2,
        var2 = node_x3,
        observation = SE2Pose(x=move, y=0, theta=np.pi / 2),
        covariance = odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x3_x4 =  SE2RelativeGaussianLikelihoodFactor(
        var1 = node_x3,
        var2 = node_x4,
        observation = SE2Pose(x=move, y=0, theta=np.pi / 2),
        covariance = odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x4_x5 =  SE2RelativeGaussianLikelihoodFactor(
        var1 = node_x4,
        var2 = node_x5,
        observation = SE2Pose(x=move, y=0, theta=np.pi / 2),
        covariance = odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x5_x2 =  SE2RelativeGaussianLikelihoodFactor(
        var1 = node_x5,
        var2 = node_x2,
        observation = SE2Pose(x=move, y=0, theta=np.pi / 2),
        covariance = odometry_noise,
        correlated_R_t=correlated_R_t
    )

    mp.set_start_method("spawn")
    model = NFiSAM()

    #settings
    poster_sample_num = 1000
    flow_number = 1
    flow_type = "NSF_AR"
    flow_iterations = 190
    cuda_training = True
    store_clique_samples = True
    local_sample_num = 1000
    fig_format = 'png'
    sampling_method = "nested"

    step_list = []
    time_list = []
    #step 0
    step = 0
    step_str = str(step)
    model.add_node(node_x1)
    model.add_factor(prior_factor_x1)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')

    #step 1
    step += 1
    step_str = str(step)
    model.add_node(node_x2)
    model.add_factor(llfactor_x1_x2)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')


    #step 2
    step += 1
    step_str = str(step)
    model.add_node(node_x3)
    model.add_factor(llfactor_x2_x3)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')


    #step 3
    step += 1
    step_str = str(step)
    model.add_node(node_x4)
    model.add_factor(llfactor_x3_x4)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')

    #step 4
    step += 1
    step_str = str(step)
    model.add_node(node_x5)
    model.add_factor(llfactor_x4_x5)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')

    #step 5
    step += 1
    step_str = str(step)
    model.add_factor(llfactor_x5_x2)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()
    end = time.time()
    print("Time for step "+step_str+ " inference " + str(end - start) + " sec")
    step_list.append(step)
    time_list.append(end - start)
    plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/'+str(step)+'.svg')

    plt.figure()
    plt.plot(step_list, time_list, marker='x')
    plt.xlabel('Step')
    plt.ylabel('Time (second)')
    plt.savefig(fname = path+"/Time_performance."+fig_format, format = fig_format)
    plt.show()


    # import shelve
    # filename = 'eight_node_data'
    # my_shelf = shelve.open(filename, 'n')  # 'n' for new
    #
    # for key in dir():
    #     try:
    #         my_shelf[key] = globals()[key]
    #     except TypeError:
    #         #
    #         # __builtins__, my_shelf, and imported modules can not be shelved.
    #         #
    #         print('ERROR shelving: {0}'.format(key))
    # my_shelf.close()
