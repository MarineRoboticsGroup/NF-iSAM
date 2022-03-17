import os
import time
import numpy as np
from slam.Variables import R2Variable
from factors.Factors import R2RelativeGaussianLikelihoodFactor, R2RangeGaussianLikelihoodFactor, GaussianPriorFactor
from slam.NFiSAM import NFiSAM, NFiSAMArgs
import matplotlib.pyplot as plt

from sampler.NestedSampling import GlobalNestedSampler
from utils.Visualization import plot_2d_samples


if __name__ == '__main__':
    path = "five_node_range_incremental_res"

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

    prior_factor_l1 = GaussianPriorFactor(var = node_l1,
                                          mean = np.array([5.0, 5.0]),
                                          covariance = np.identity(2) * 0.5
                                          )
    prior_factor_l2 = GaussianPriorFactor(var = node_l2,
                                          mean = np.array([10.0, 5.0]),
                                          covariance = np.identity(2) * 0.5
                                          )

    llfactor_x0_l1 = R2RangeGaussianLikelihoodFactor(
        var1=node_x0,
        var2=node_l1,
        observation=dist_x0_l1,
        sigma=sigma)
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

    factors = [prior_factor_l1, prior_factor_l2, llfactor_x0_x1, llfactor_x0_l1, llfactor_l1_x1,
               llfactor_x1_x2, llfactor_l2_x2]
    nodes0 = [node_x0, node_l1]
    nodes1 = [node_x0, node_l1, node_x1]
    factors0 = [prior_factor_l1, llfactor_x0_l1]
    factors1 = [prior_factor_l1, llfactor_x0_x1, llfactor_x0_l1, llfactor_l1_x1]

    local_sample_num = 1000
    draw_samples = 500
    flow_iteration = 800
    flow_number = 1
    knot = 15
    gpu_training = True

    args = NFiSAMArgs(posterior_sample_num=draw_samples,
                      flow_type="NSF_AR",
                      flow_number=flow_number,
                      flow_iterations=flow_iteration,
                      local_sample_num=local_sample_num,
                      cuda_training=gpu_training,
                      store_clique_samples=False,
                      num_knots=knot)
    model = NFiSAM(args)

    model = NFiSAM()
    step_list = []
    time_list = []
    step = 0
    step_list.append(step)
    model.add_node(node_l1)
    model.add_node(node_x0)
    model.add_factor(prior_factor_l1)
    model.add_factor(llfactor_x0_l1)
    model.update_physical_and_working_graphs()
    start = time.time()

    samples = model.incremental_inference(timer=[start])
    end = time.time()
    time_list.append(end - start)
    print("Time for phase "+str(step)+" inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_mapping=samples, show_plot=True, file_name=path+'/step'+str(step)+'.svg',
                    legend_on=True,title='Posterior estimation (step '+str(step)+')',equal_axis=False)


    step = step + 1
    step_list.append(step)
    model.add_node(node_x1)
    model.add_factor(llfactor_x0_x1)
    model.add_factor(llfactor_l1_x1)
    model.update_physical_and_working_graphs()
    start = time.time()
    samples = model.incremental_inference(timer=[start])
    end = time.time()
    time_list.append(end - start)
    print("Time for phase "+str(step)+" inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_mapping=samples, show_plot=True, file_name=path+'/step'+str(step)+'.svg',
                    legend_on=True,title='Posterior estimation (step '+str(step)+')',equal_axis=False)


    step = step + 1
    step_list.append(step)
    model.add_node(node_x2)
    model.add_node(node_l2)
    model.add_factor(llfactor_x1_x2)
    model.add_factor(llfactor_l2_x2)
    model.add_factor(prior_factor_l2)
    model.update_physical_and_working_graphs()
    start = time.time()
    samples = model.incremental_inference(timer=[start])
    end = time.time()
    time_list.append(end - start)
    print("Time for phase "+str(step)+" inference " + str(end - start) + " sec")
    plt.figure()
    plot_2d_samples(samples_mapping=samples, show_plot=True, file_name=path+'/step'+str(step)+'.svg',
                    legend_on=True,title='Posterior estimation (step '+str(step)+')',equal_axis=False)

    plt.figure()
    plt.plot(step_list, time_list)
    plt.xlabel("Step")
    plt.ylabel("Time (sec)")
    plt.savefig(path+'/timing(NFSLAM).svg')
    plt.show()

    fig_format = "svg"
    step_list = []
    time_list = []
    step = 0
    step_list.append(step)
    sampler = GlobalNestedSampler(nodes=nodes0, factors=factors0)
    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=draw_samples)
    end = time.time()
    time_list.append(end-start)
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    fig = plot_2d_samples(samples_array=samples, variable_ordering=nodes0,
                    show_plot=True, equal_axis=False, legend_on=True,
                    title="Reference solution "+str(step), file_name=path+'/ref_step_'+str(step)+'.svg')
    # fig.savefig(fname="Reference solution "+str(step) + fig_format,
    #              format="png")
    step = step+1
    step_list.append(step)
    sampler = GlobalNestedSampler(nodes=nodes1, factors=factors1)
    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=draw_samples)
    end = time.time()
    time_list.append(end-start)
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    fig2 = plot_2d_samples(samples_array=samples, variable_ordering=nodes1,
                    show_plot=True, equal_axis=True, legend_on=True,
                    title="Reference solution "+str(step), file_name=path+'/ref_step_'+str(step)+'.svg')
    # fig2.savefig(fname="Reference solution "+str(step) + fig_format,
    #              format="png")
    step = step+1
    step_list.append(step)
    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)
    start = time.time()
    #an alternative sampling method is dynamic_nested, which may be a bit unstable
    samples = sampler.sample(live_points=draw_samples)
    end = time.time()
    time_list.append(end-start)
    print("Time for inference " + str(end - start) + " sec")
    plt.figure()
    fig3 = plot_2d_samples(samples_array=samples, variable_ordering=nodes,
                    show_plot=True, equal_axis=True, legend_on=True,
                    title="Reference solution "+str(step), file_name=path+'/ref_step_'+str(step)+'.svg')
    # fig3.savefig(fname="Reference solution "+str(step) + fig_format,
    #              format="png")

    plt.figure()
    plt.plot(step_list, time_list)
    plt.xlabel("Step")
    plt.ylabel("Time (sec)")
    plt.savefig(path+'/timing(ref).svg')
    plt.show()
