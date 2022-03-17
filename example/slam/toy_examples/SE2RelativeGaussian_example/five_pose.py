import time

import numpy as np
from slam.Variables import SE2Variable
from factors.Factors import UnarySE2ApproximateGaussianPriorFactor, SE2RelativeGaussianLikelihoodFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt
from geometry.TwoDimension import SE2Pose
from sampler.NestedSampling import GlobalNestedSampler
from utils.Visualization import plot_2d_samples
import os

def plot_data(x,i=0,j=1, **kwargs):
    plt.scatter(x[:,i], x[:,j], marker=".", **kwargs)

if __name__ == '__main__':

    #this is an example similar to Pose2SLAMExample of GTSAM, p13
    #https://borg.cc.gatech.edu/sites/edu.borg/files/downloads/gtsam.pdf

    # define the name of the directory to be created
    path = "five_pose"

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

    nodes = [node_x1,node_x2,node_x3,node_x4,node_x5]

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
    factors = [prior_factor_x1,llfactor_x1_x2,llfactor_x2_x3,
               llfactor_x3_x4,llfactor_x4_x5,llfactor_x5_x2]
    # mp.set_start_method("spawn")
    args = NFiSAMArgs()
    model = NFiSAM(args)

    #settings
    poster_sample_num = 200
    flow_number = 1
    flow_type = "NSF_AR"
    flow_iterations = 190
    cuda_training = False
    store_clique_samples = False
    local_sample_num = 1000
    fig_format = 'png'
    sampling_method = "nested"
    save_res = False

    step_list = []
    time_list = []
    #step 0
    model.add_node(node_x1)
    model.add_factor(prior_factor_x1)
    model.add_node(node_x2)
    model.add_factor(llfactor_x1_x2)
    model.add_node(node_x3)
    model.add_factor(llfactor_x2_x3)
    model.add_node(node_x4)
    model.add_factor(llfactor_x3_x4)
    model.add_node(node_x5)
    model.add_factor(llfactor_x4_x5)
    model.add_factor(llfactor_x5_x2)
    model.update_physical_and_working_graphs()
    res = model.incremental_inference()
    # fig = model.plot2d_posterior(title="Posterior samples")
    fig = plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/NFSLAM.svg')

    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)

    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=200)
    end = time.time()
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    fig = plot_2d_samples(samples_array=samples,variable_ordering=nodes,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Reference solution",file_name=path+'/Reference.svg')

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