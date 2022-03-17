import os
import time
import numpy as np

from geometry.TwoDimension import SE2Pose
from slam.Variables import R2Variable, SE2Variable, VariableType
from factors.Factors import UnaryR2RangeGaussianPriorFactor, UnarySE2ApproximateGaussianPriorFactor, SE2RelativeGaussianLikelihoodFactor, \
    SE2R2RangeGaussianLikelihoodFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt

from utils.Visualization import plot_2d_samples


if __name__ == '__main__':
    # define the name of the directory to be created
    path = "five_node_range_res"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    node_x0 = SE2Variable('x0')
    node_x1 = SE2Variable('x1')
    node_x2 = SE2Variable('x2')
    node_l1 = R2Variable('l1',variable_type=VariableType.Landmark)
    node_l2 = R2Variable('l2',variable_type=VariableType.Landmark)
    nodes = [node_x0, node_x1, node_x2, node_l1, node_l2]
    dist_x0_l1 = 5 * np.sqrt(2)
    dist_l1_x1 = 10
    pose_x0 = SE2Pose(x=0,y=0,theta=-np.pi/4)
    tf_x0_x1 = SE2Pose(x=5*np.sqrt(2),y=0,theta=np.pi/2)
    tf_x1_x2 = SE2Pose(x=5*np.sqrt(2),y=0,theta=0.0)
    dist_l2_x2 = 5
    sigma = 0.5
    pose_cov = np.identity(3) * 0.5
    pose_cov[2,2] = 0.01

    prior_factor_x0 = UnarySE2ApproximateGaussianPriorFactor(var = node_x0,
                                                 prior_pose = pose_x0,
                                                 covariance=pose_cov
                                                 )
    prior_factor_l2 = UnaryR2RangeGaussianPriorFactor(
        var = node_l2,
        center = np.array([10, 0]),
        mu = 5,
        sigma=sigma)
    llfactor_x0_l1 = SE2R2RangeGaussianLikelihoodFactor(
        var1=node_x0,
        var2=node_l1,
        observation=dist_x0_l1,
        sigma=sigma)#warning: this sigma is standard deviation
    llfactor_l1_x1 = SE2R2RangeGaussianLikelihoodFactor(
        var1=node_l1,
        var2=node_x1,
        observation=dist_l1_x1,
        sigma=sigma)
    llfactor_x0_x1 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x0,
        var2=node_x1,
        observation=tf_x0_x1,
        covariance=pose_cov)
    llfactor_x1_x2 = SE2RelativeGaussianLikelihoodFactor(
        var1=node_x1,
        var2=node_x2,
        observation=tf_x1_x2,
        covariance=pose_cov)
    llfactor_l2_x2 = SE2R2RangeGaussianLikelihoodFactor(
        var1=node_l2,
        var2=node_x2,
        observation=dist_l2_x2,
        sigma=sigma)

    factors = [prior_factor_x0, prior_factor_l2, llfactor_x0_x1, llfactor_x0_l1,
               llfactor_l1_x1, llfactor_x1_x2, llfactor_l2_x2]

    args = NFiSAMArgs(posterior_sample_num=1000,
                      flow_type="NSF_AR",
                      flow_number=1,
                      flow_iterations=200,
                      local_sample_num=500,
                      cuda_training=True,
                      store_clique_samples=False,
                      num_knots=5)
    model = NFiSAM(args)

    model.add_node(node_x0)
    model.add_node(node_x1)
    model.add_node(node_l1)
    model.add_node(node_x2)
    model.add_node(node_l2)

    model.add_factor(prior_factor_x0)
    model.add_factor(prior_factor_l2)
    model.add_factor(llfactor_x0_l1)
    model.add_factor(llfactor_x0_x1)
    model.add_factor(llfactor_x1_x2)
    model.add_factor(llfactor_l1_x1)
    model.add_factor(llfactor_l2_x2)

    model.update_physical_and_working_graphs()
    start = time.time()

    samples = model.incremental_inference()
    end = time.time()
    print("Time for phase 1 inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_mapping=samples, show_plot=True, file_name=path+'/NFSLAM.svg',
                    legend_on=True,title='Posterior estimation',equal_axis=False,has_orientation=True)

    # sampler = GlobalNestedSampler(nodes=nodes, factors=factors)
    #
    # start = time.time()
    # #an alternative sampling method is dynamic_nested, which may be a bit unstable
    # samples2 = sampler.sample(live_points=300, downsampling=True)
    # end = time.time()
    # print("Time for inference " + str(end - start) + " sec")
    # plt.figure()
    # plot_2d_samples(samples_array=samples2, variable_ordering=nodes,
    #                 show_plot=True, legend_on=True,equal_axis=False,
    #                 title="Reference solution",has_orientation=True, file_name=path+'/Reference.svg')



