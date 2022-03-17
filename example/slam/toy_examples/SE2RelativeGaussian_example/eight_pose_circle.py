import os
import time

import numpy as np
from slam.Variables import SE2Variable
from factors.Factors import SE2RelativeGaussianLikelihoodFactor, \
    UnarySE2ApproximateGaussianPriorFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt
from geometry.TwoDimension import SE2Pose
from sampler.NestedSampling import GlobalNestedSampler
from utils.Visualization import plot_2d_samples


def plot_data(x, i=0, j=1, **kwargs):
    plt.scatter(x[:, i], x[:, j], marker=".", **kwargs)


if __name__ == '__main__':
    # this is an example similar to Pose2SLAMExample of GTSAM, p13
    # https://borg.cc.gatech.edu/sites/edu.borg/files/downloads/gtsam.pdf

    path = "eight_pose_circle"

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
    node_x6 = SE2Variable('x6')
    node_x7 = SE2Variable('x7')
    node_x8 = SE2Variable('x8')

    nodes = [node_x1, node_x2,
             node_x3, node_x4,
             node_x5, node_x6,
             node_x7, node_x8]

    # results with low theta std
    move = 4
    correlated_R_t = True
    prior_noise = np.diag([0.3 ** 2, 0.3 ** 2, 0.1 ** 2])
    odometry_noise = np.diag([0.2 ** 2, 0.2 ** 2, 0.1 ** 2])

    prior_factor_x1 = UnarySE2ApproximateGaussianPriorFactor(
        var=node_x1,
        prior_pose=SE2Pose(x=0, y=0, theta=np.pi / 8),
        covariance=prior_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x1_x2 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x1,
        var2=node_x2,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x2_x3 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x2,
        var2=node_x3,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x3_x4 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x3,
        var2=node_x4,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x4_x5 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x4,
        var2=node_x5,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x5_x6 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x5,
        var2=node_x6,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x6_x7 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x6,
        var2=node_x7,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x7_x8 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x7,
        var2=node_x8,
        observation=SE2Pose(x=move, y=0, theta=2 * np.pi / 8),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )
    llfactor_x8_x1 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x8,
        var2=node_x1,
        observation=SE2Pose(x=move, y=0, theta=1 * np.pi / 4),
        covariance=odometry_noise,
        correlated_R_t=correlated_R_t
    )

    factors = [prior_factor_x1, llfactor_x1_x2,
               llfactor_x2_x3, llfactor_x3_x4,
               llfactor_x4_x5, llfactor_x5_x6,
               llfactor_x6_x7,llfactor_x7_x8,
               llfactor_x8_x1]

    # mp.set_start_method("spawn")
    args = NFiSAMArgs(cuda_training=True)
    model = NFiSAM(args)

    # settings


    step_list = []
    time_list = []

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
    model.add_node(node_x6)
    model.add_factor(llfactor_x5_x6)
    model.add_node(node_x7)
    model.add_factor(llfactor_x6_x7)
    model.add_node(node_x8)
    model.add_factor(llfactor_x7_x8)
    model.update_physical_and_working_graphs()
    start = time.time()
    res = model.incremental_inference()

    end = time.time()
    print(
        "Time for inference " + str(end - start) + " sec")
    fig = plot_2d_samples(samples_mapping=res,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/NFSLAM(no loop).svg')
    # fig.savefig(fname="8node_circle_flow_no_closure_solution." + fig_format,
    #             format=fig_format)

    # ground truth samples
    sampler = GlobalNestedSampler(nodes=nodes, factors=factors[:-1])
    samples = sampler.sample(live_points=args.posterior_sample_num)
    end = time.time()
    print("Time for reference samples " + str(end - start) + " sec")
    fig = plot_2d_samples(samples_array=samples,variable_ordering=nodes,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Reference solution",file_name=path+'/Reference(no loop).svg')
    # fig.savefig(fname="8node_circle_flow_no_closure_reference." + fig_format,
    #             format=fig_format)

    # Add loop closure
    model.add_factor(llfactor_x8_x1)
    model.update_physical_and_working_graphs()
    start = time.time()
    res2 = model.incremental_inference()
    end = time.time()
    print("Time for inference " + str(end - start) + " sec")
    plot_2d_samples(samples_mapping=res2,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Posterior estimation",file_name=path+'/NFSLAM(one loop).svg')

    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)

    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=args.posterior_sample_num)
    end = time.time()
    print("Time for inference " + str(end - start) + " sec")
    plot_2d_samples(samples_array=samples,variable_ordering=nodes,
                    show_plot=True, equal_axis=False, legend_on=True,has_orientation=True,
                    title="Reference solution",file_name=path+'/Reference(one loop).svg')
