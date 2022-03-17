import time
import numpy as np
from slam.Variables import VariableType
from factors.Factors import PriorFactor, BinaryFactor, KWayFactor
import matplotlib.pyplot as plt
from utils.Visualization import plot_2d_samples
from slam.FactorGraphSimulator import read_factor_graph_from_file
from sampler.NestedSampling import GlobalNestedSampler
import os


def generate_pose_first_ordering(nodes) -> None:
    """
    Generate the ordering by which nodes are added and lmk eliminated later
    """
    pose_list = []
    lmk_list = []
    for node in nodes:
        if node._type == VariableType.Landmark:
            lmk_list.append(node)
        else:
            pose_list.append(node)
    return pose_list + lmk_list

def generate_reference_samples_for_numerical_example_1(
        setup_name: str, case_name: str, sample_num: int):
    case_dir = f"{setup_name}/{case_name}"
    nodes, truth, factors = read_factor_graph_from_file(
        f"{case_dir}/factor_graph")
    ref_cnt = 1
    ref_dir = f"{case_dir}/reference{ref_cnt}"
    while ref_cnt<100:
        if not os.path.exists(ref_dir):
            os.mkdir(ref_dir)
            break
        else:
            ref_cnt += 1
            ref_dir = f"{case_dir}/reference{ref_cnt}"

    # prior_node = [node for node in nodes if node.name == "X0"][0]
    remaining_nodes = nodes.copy()
    # remaining_nodes.remove(prior_node)
    remaining_factors = factors.copy()
    current_nodes = []
    current_factors = []

    start_time = time.time()
    step = 0
    step_timer = []
    while remaining_nodes:
        print(f"Time step {step} sampling starts")
        new_node = remaining_nodes.pop(0)
        current_nodes.append(new_node)
        new_factors = []
        for factor in remaining_factors:
            if not set.intersection(set(remaining_nodes), set(factor.vars)):
                new_factors.append(factor)
            elif isinstance(factor, BinaryFactor):
                var1, var2 = factor.vars
                if var1 not in remaining_nodes and var2.type == \
                        VariableType.Landmark:
                    new_factors.append(factor)
                    remaining_nodes.remove(var2)
                    current_nodes.append(var2)
                elif var2 not in remaining_nodes and var1.type == \
                        VariableType.Landmark:
                    new_factors.append(factor)
                    remaining_nodes.remove(var1)
                    current_nodes.append(var1)
        for factor in new_factors:
            remaining_factors.remove(factor)
            current_factors.append(factor)

        #an important step avoiding sampling SE2Pose from Point2 samples
        current_nodes = generate_pose_first_ordering(current_nodes)
        start = time.time()
        sampler = GlobalNestedSampler(nodes=current_nodes,
                                      factors=current_factors)
        samples = sampler.sample(live_points=sample_num)
        end = time.time()
        step_timer.append(end - start)
        print(f"Step {step} time for inference {end - start} sec")

        np.savetxt(fname=f"{ref_dir}/step_{step}",
                   X=samples)

        file = open(f"{ref_dir}/step_{step}_ordering", "w+")
        file.write(" ".join([str(var.name) for var in current_nodes]))
        file.close()

        plt.figure()
        ax = plot_2d_samples(samples_array=samples,
                             variable_ordering=current_nodes,
                             show_plot=False, equal_axis=True)
        for node in nodes:
            if node not in remaining_nodes:
                x, y = truth[node][:2]
                color = "r" if node.type == VariableType.Landmark else \
                    "b"
                ax.plot([x], [y], c=color, markersize=12, marker="x")
                ax.text(x, y - 1, s=node.name)
        for factor in factors:
            if not isinstance(factor, PriorFactor):
                if factor not in remaining_factors:
                    if isinstance(factor, BinaryFactor):
                        var1, var2 = factor.vars
                        x1, y1 = truth[var1][:2]
                        x2, y2 = truth[var2][:2]
                        ax.plot([x1, x2], [y1, y2], c='k', linewidth=1)
                    elif isinstance(factor, KWayFactor):
                        var1 = factor.root_var
                        for var2 in factor.child_vars:
                            x1, y1 = truth[var1][:2]
                            x2, y2 = truth[var2][:2]
                            ax.plot([x1, x2], [y1, y2], c='k', linewidth=1)
        plt.axis("equal")
        plt.xlim((-15, 35))
        plt.ylim((-10, 30))
        plt.savefig(f"{ref_dir}/step_{step}.png")

        step += 1
    end_time = time.time()
    print(f"{end_time - start_time} sec used for reference solution")
    np.savetxt(fname=f"{ref_dir}/timing",
               X=np.array(step_timer))


if __name__ == '__main__':
    setups = ['icra_paper']
    # cases = ['case1_da','case1_da2','case1']
    cases = ['case_da']
    for setup in setups:
        for case in cases:
            generate_reference_samples_for_numerical_example_1(
                setup_name=setup, case_name=case,
                sample_num=500)
