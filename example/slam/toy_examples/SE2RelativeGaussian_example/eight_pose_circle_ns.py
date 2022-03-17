import os
import random
import time

import numpy as np
from slam.Variables import SE2Variable
from factors.Factors import SE2RelativeGaussianLikelihoodFactor, \
    UnarySE2ApproximateGaussianPriorFactor
import matplotlib.pyplot as plt
from geometry.TwoDimension import SE2Pose
from sampler.NestedSampling import GlobalNestedSampler
from utils.Visualization import plot_2d_samples


def plot_data(x, i=0, j=1, **kwargs):
    plt.scatter(x[:, i], x[:, j], marker=".", **kwargs)


if __name__ == '__main__':

    path = "eight_pose_circle"

    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    folder = 'ns_informative_prior'
    case_path = f"{path}/{folder}"
    if not os.path.exists(case_path):
        os.mkdir(case_path)

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

    sampler = GlobalNestedSampler(nodes=nodes, factors=factors)
    step_time_list = []
    run_num = 10
    for i in range(run_num):
        start = time.time()

        #an alternative sampling method is dynamic_nested, which may be a bit unstable
        sample_num = 500
        samples = sampler.sample(live_points=sample_num, downsampling=False)
        end = time.time()
        print("Time for inference " + str(end - start) + " sec")
        step_time_list.append(end - start)
        run_count = 1
        while os.path.exists(f"{case_path}/run_{run_count}"):
            run_count += 1

        file = open(f"{case_path}/run_{run_count}_ordering", "w+")
        file.write(" ".join([var.name for var in nodes]))
        file.close()

        np.savetxt(fname=f"{case_path}/run_{run_count}", X=samples)

        mean = np.mean(samples, axis=0, keepdims=True)
        cov = np.cov(samples, rowvar=False)
        np.savetxt(fname=f"{case_path}/run_{run_count}_mean", X=mean)
        np.savetxt(fname=f"{case_path}/run_{run_count}_cov", X=cov)

        sampled_idx = random.sample(list(range(samples.shape[0])), sample_num)
        down_samples = samples[sampled_idx, :]

        plt.figure()
        ax = plot_2d_samples(samples_array=down_samples,variable_ordering=nodes,
                        show_plot=False, equal_axis=False, legend_on=True, has_orientation=True)
        # for node in nodes:
        #     x, y = truth[node]
        #     color = "r" if node.type == VariableType.Landmark else \
        #         "b"
        #     ax.plot([x], [y], c=color, markersize=12, marker="x")
        #     ax.text(x+2, y, s=node.name)
        # for factor in factors:
        #     if not isinstance(factor, PriorFactor):
        #         var1, var2 = factor.vars
        #         x1, y1 = truth[var1]
        #         x2, y2 = truth[var2]
        #         ax.plot([x1, x2], [y1, y2], c='k', linewidth=1)
        plt.savefig(f"{case_path}/run_{run_count}.png")
        plt.show()
    np.savetxt(fname=f"{case_path}/batch_time_list",
               X=np.array(step_time_list))
