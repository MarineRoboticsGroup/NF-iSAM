import time

import numpy as np
from slam.Variables import R2Variable
from factors.Factors import R2RelativeGaussianLikelihoodFactor, UnaryR2GaussianPriorFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt
import multiprocessing as mp

from sampler.NestedSampling import GlobalNestedSampler
from utils.Visualization import plot_2d_samples


def plot_data(x,i=0,j=1, **kwargs):
    plt.scatter(x[:,i], x[:,j], marker="x", **kwargs)
    plt.xlim((-10, 15))
    plt.ylim((-10, 15))

if __name__ == '__main__':
    node_x0 = R2Variable('x0')
    node_x1 = R2Variable('x1')
    node_x2 = R2Variable('x2')
    node_x3 = R2Variable('x3')
    node_x4 = R2Variable('x4')
    node_x5 = R2Variable('x5')
    node_l1 = R2Variable('l1')
    node_l2 = R2Variable('l2')

    nodes = [node_x0,node_x1,node_x2,node_x3,node_x4,node_x5,node_l1,node_l2]

    disp_x0_l1 = np.array([5, 5])
    disp_l1_x1 = np.array([0, -10])
    disp_x0_x1 = np.array([5, -5])
    disp_x1_x2 = np.array([5, 5])
    disp_l2_x2 = np.array([0, -5])
    disp_x2_x3 = np.array([5, 5])
    disp_x3_x4 = np.array([-5, 5])
    disp_x4_x5 = np.array([-5, 0])

    disp_l2_x3 = np.array([5, 0])
    disp_l2_x4 = np.array([0, 5])
    disp_l1_x5 = np.array([0, 5])
    prior_factor_l1 = UnaryR2GaussianPriorFactor(var = node_l1,
                                                 mu = np.array([5.0, 5.0]),
                                                 covariance=np.identity(2) * 0.5
                                                 )
    prior_factor_l2 = UnaryR2GaussianPriorFactor(var = node_l2,
                                                 mu = np.array([10.0, 5.0]),
                                                 covariance=np.identity(2) * 0.5
                                                 )
    llfactor_x0_l1 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_x0,
        var2 = node_l1,
        observation = disp_x0_l1,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_l1_x1 =  R2RelativeGaussianLikelihoodFactor(
        var1 = node_l1,
        var2 = node_x1,
        observation = disp_l1_x1,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x0_x1 =  R2RelativeGaussianLikelihoodFactor(
        var1 = node_x0,
        var2 = node_x1,
        observation = disp_x0_x1,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x1_x2 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_x1,
        var2 = node_x2,
        observation = disp_x1_x2,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_l2_x2 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_l2,
        var2 = node_x2,
        observation = disp_l2_x2,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x2_x3 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_x2,
        var2 = node_x3,
        observation = disp_x2_x3,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x3_x4 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_x3,
        var2 = node_x4,
        observation = disp_x3_x4,
        precision= np.array([[10, 0.0], [0.0, 10]]))
    llfactor_x4_x5 = R2RelativeGaussianLikelihoodFactor(
        var1 = node_x4,
        var2 = node_x5,
        observation = disp_x4_x5,
        precision= np.array([[10, 0.0], [0.0, 10]]))

    llfactor_l2_x3 = R2RelativeGaussianLikelihoodFactor(
        var1=node_l2,
        var2=node_x3,
        observation=disp_l2_x3,
        precision=np.array([[10, 0.0], [0.0, 10]]))
    llfactor_l2_x4 = R2RelativeGaussianLikelihoodFactor(
        var1=node_l2,
        var2=node_x4,
        observation=disp_l2_x4,
        precision=np.array([[10, 0.0], [0.0, 10]]))
    llfactor_l1_x5 = R2RelativeGaussianLikelihoodFactor(
        var1=node_l1,
        var2=node_x5,
        observation=disp_l1_x5,
        precision=np.array([[10, 0.0], [0.0, 10]]))

    factors = [prior_factor_l1, prior_factor_l2,llfactor_l1_x1,
               llfactor_x0_l1,llfactor_x0_x1,llfactor_x1_x2,llfactor_l2_x2,
               llfactor_x2_x3,llfactor_x3_x4,llfactor_x4_x5,llfactor_l2_x3,llfactor_l2_x4,
               llfactor_l1_x5]

    args = NFiSAMArgs(posterior_sample_num = 100,
                      flow_number= 1,
                      flow_type="NSF_AR",
                      flow_iterations=50,
                      local_sample_num=200,
                      cuda_training=False,
                      store_clique_samples=False,
                      local_sampling_method="nested")
    model = NFiSAM(args)

    model.add_node(node_l1)
    model.add_node(node_l2)
    model.add_node(node_x0)
    model.add_node(node_x1)
    model.add_node(node_x2)
    model.add_node(node_x3)
    model.add_node(node_x4)
    model.add_node(node_x5)

    model.add_factor(prior_factor_l1)
    model.add_factor(prior_factor_l2)
    model.add_factor(llfactor_x0_l1)
    model.add_factor(llfactor_x0_x1)
    model.add_factor(llfactor_x1_x2)
    model.add_factor(llfactor_l1_x1)
    model.add_factor(llfactor_l2_x2)
    model.add_factor(llfactor_x2_x3)
    model.add_factor(llfactor_x3_x4)
    model.add_factor(llfactor_x4_x5)

    mp.set_start_method("spawn")


    model.update_physical_and_working_graphs()
    start = time.time()
    model.incremental_inference()
    end = time.time()
    print("Time for phase 1 inference " + str(end - start) + " sec")
    plt.figure()
    model.plot2d_posterior(title="Phase 1 posterior samples")

    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)

    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=300)
    end = time.time()
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_array=samples, variable_ordering=nodes,
                    show_plot=True, equal_axis=True, legend_on=True,
                    title="Reference solution")
