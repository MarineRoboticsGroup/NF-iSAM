import os
import time
import numpy as np
from slam.Variables import R2Variable
from factors.Factors import R2RelativeGaussianLikelihoodFactor, R2RangeGaussianLikelihoodFactor, UnaryR2GaussianPriorFactor, \
    UnaryR2RangeGaussianPriorFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt

from sampler.NestedSampling import GlobalNestedSampler
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

    node_x0 = R2Variable('x0')
    node_x1 = R2Variable('x1')
    node_x2 = R2Variable('x2')
    node_l1 = R2Variable('l1')
    node_l2 = R2Variable('l2')
    nodes = [node_x0, node_x1, node_x2, node_l1, node_l2]
    dist_x0_l1 = 5 * np.sqrt(2)
    dist_l1_x1 = 10
    disp_x0_x1 = np.array([5, -5])
    disp_x1_x2 = np.array([5, 5])
    dist_l2_x2 = 5
    sigma = 0.5

    prior_factor_x0 = UnaryR2GaussianPriorFactor(var = node_x0,
                                                 mu = np.array([0.0, 0.0]),
                                                 covariance=np.identity(2) * sigma ** 2
                                                 )
    prior_factor_l2 = UnaryR2RangeGaussianPriorFactor(
        var = node_l2,
        center = np.array([10, 0]),
        mu = 5,
        sigma=sigma)
    llfactor_x0_l1 = R2RangeGaussianLikelihoodFactor(
        var1=node_x0,
        var2=node_l1,
        observation=dist_x0_l1,
        sigma=sigma)#warning: this sigma is standard deviation
    llfactor_l1_x1 = R2RangeGaussianLikelihoodFactor(
        var1=node_l1,
        var2=node_x1,
        observation=dist_l1_x1,
        sigma=sigma)
    llfactor_x0_x1 = R2RelativeGaussianLikelihoodFactor(
        var1=node_x0,
        var2=node_x1,
        observation=disp_x0_x1,
        precision=np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x1_x2 = R2RelativeGaussianLikelihoodFactor(
        var1=node_x1,
        var2=node_x2,
        observation=disp_x1_x2,
        precision=np.array([[10, 0.0], [0.0, 10]]))
    llfactor_l2_x2 = R2RangeGaussianLikelihoodFactor(
        var1=node_l2,
        var2=node_x2,
        observation=dist_l2_x2,
        sigma=sigma)

    factors = [prior_factor_x0, prior_factor_l2, llfactor_x0_x1, llfactor_x0_l1,
               llfactor_l1_x1, llfactor_x1_x2, llfactor_l2_x2]

    args = NFiSAMArgs(posterior_sample_num = 1000,
                      flow_number=1,
                      flow_type="NSF_AR",
                      flow_iterations=200,
                      local_sample_num=500,
                      cuda_training=True,
                      store_clique_samples=False,
                      num_knots=5)
    model = NFiSAM(args)

    model.add_node(node_l1)
    model.add_node(node_l2)
    model.add_node(node_x0)
    model.add_node(node_x1)
    model.add_node(node_x2)

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
                    legend_on=True,title='Posterior estimation',equal_axis=False)

    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)

    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=300)
    end = time.time()
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_array=samples, variable_ordering=nodes,
                    show_plot=True, legend_on=True,equal_axis=False,
                    title="Reference solution", file_name=path+'/Reference.svg')



