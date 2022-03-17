import time
import numpy as np
from slam.Variables import Variable, VariableType
from slam.NFiSAM import NFiSAM, NFiSAMArgs
from utils.Visualization import plot_2d_samples
from slam.FactorGraphSimulator import read_factor_graph_from_file
from slam.RunBatch import group_nodes_factors_incrementally
from utils.Statistics import mmd
import os
from typing import List
import json
import matplotlib.pylab as plt


def reorder_samples(ref_order: List[Variable], sample_order: List[Variable], samples: np.ndarray) -> np.ndarray:
    res = np.zeros_like(samples)
    cur_dim = 0
    for var in ref_order:
        var_idx = sample_order.index(var)
        if var_idx == 0:
            sample_dim = 0
        else:
            sample_dim = np.sum([it.dim for it in sample_order[:var_idx]])
        res[:, cur_dim:cur_dim+var.dim] = samples[:, sample_dim:sample_dim+var.dim]
        cur_dim += var.dim
    return res


def run(setup_name: str, case_name: str, show_plot: bool = False,
        compute_mmd: bool = False, incremental_step: int = None,
        ordering:str = "natural", **params) -> None:

    case_dir = f"{setup_name}/{case_name}"
    factor_graph_file = f"{case_dir}/factor_graph"
    nodes, truth, factors = read_factor_graph_from_file(factor_graph_file)
    nodes_factors_by_step = group_nodes_factors_incrementally(
        nodes=nodes, factors=factors, incremental_step=incremental_step)

    if "poster_sample_num" not in params:
        params["poster_sample_num"] = 1000
    if "local_sample_num" not in params:
        params["local_sample_num"] = 500
    if "flow_iterations" not in params:
        params["flow_iterations"] = 200
    if "flow_type" not in params:
        params["flow_type"] = "NSF_AR"
    if "flow_number" not in params:
        params["flow_number"] = 1
    if "cuda_training" not in params:
        params["cuda_training"] = True
    if "learning_rate" not in params:
        params["learning_rate"] = 0.015
    if "num_knots" not in params:
        params["num_knots"] = 5

    run_count = 1
    while os.path.exists(f"{case_dir}/run{run_count}"):
        run_count += 1

    os.mkdir(f"{case_dir}/run{run_count}")
    run_dir = f"{case_dir}/run{run_count}"

    file = open(f"{run_dir}/parameters", "w+")
    params_to_save = params.copy()
    params_to_save["incremental_steps"] = incremental_step
    file.write(json.dumps(params_to_save))
    file.close()

    args = NFiSAMArgs(
        elimination_method=ordering,
        posterior_sample_num=params["poster_sample_num"],
        flow_number=params["flow_number"],
        flow_type=params["flow_type"],
        flow_iterations=params["flow_iterations"],
        local_sample_num=params["local_sample_num"],
        cuda_training=params["cuda_training"],
        learning_rate=params["learning_rate"],
        num_knots=params["num_knots"],
        store_clique_samples=False)


    model = NFiSAM(args)
    num_batches = len(nodes_factors_by_step)
    timer = []
    observed_nodes = []
    for i in range(num_batches):
        batch_count = i + 1
        batch_timer = []
        step_nodes, step_factors = nodes_factors_by_step[i]
        for node in step_nodes:
            model.add_node(node)
        for factor in step_factors:
            model.add_factor(factor)
        observed_nodes += step_nodes
        model.update_physical_and_working_graphs(timer=batch_timer)

        print("variable elimination order is ", [var.name for var in model.elimination_ordering])

        start = time.time()
        samples = model.incremental_inference(timer=batch_timer)
        end = time.time()

        timer.append(sum(batch_timer))
        print(f"Batch {batch_count}/{num_batches} time: {end - start} sec, "
              f"total time: {sum(timer)}")

        file = open(f"{run_dir}/batch_{batch_count}_ordering", "w+")
        file.write(" ".join([var.name for var in model.elimination_ordering]))
        file.close()

        file = open(f"{run_dir}/batch_{batch_count}_timing", "w+")
        file.write(" ".join([str(t) for t in batch_timer]))
        file.close()

        X = np.hstack([samples[var] for var in model.elimination_ordering])
        np.savetxt(fname=f"{run_dir}/batch{batch_count}", X=X)

        plot_2d_samples(samples_mapping=samples, show_plot=show_plot,
                        equal_axis=True,
                        truth={variable: pose for variable, pose in
                               truth.items() if variable in observed_nodes},
                        truth_factors={factor for factor in factors if
                                       set(factor.vars).issubset(observed_nodes
                                                                 )},
                        truth_label_offset = (0, -2),
                        file_name=f"{run_dir}/batch{batch_count}.png")
        plt.close()

    file = open(f"{run_dir}/timing", "w+")
    file.write(" ".join(str(t) for t in timer))
    file.close()

    if compute_mmd:
        nodes = []
        print("Computing MMD")
        for batch in range(num_batches):
            batch_count = batch + 1
            print(f"Batch {batch_count}")
            new_nodes, _ = nodes_factors_by_step[batch]
            nodes += new_nodes
            step = len([var for var in nodes if var.type == VariableType.Pose]) - 1
            comp_samples = np.loadtxt(f"{run_dir}/batch{batch_count}")[:params["poster_sample_num"], :]
            ref_samples = np.loadtxt(f"{case_dir}/reference/step_{step}")[:params["poster_sample_num"], :]
            m = mmd(comp_samples, ref_samples)
            file = open(f"{run_dir}/batch_{batch_count}_mmd", "w+")
            file.write(str(m))
            file.close()


if __name__ == '__main__':
    knots = [12]
    iters = [600]
    nums_samples = [1000]
    parameters = {
        "flow_number": 1,
        "num_knots": 5,
        "local_sample_num": 1000,
        "flow_iterations": 80,
        "cuda_training": False,
        "poster_sample_num": 1000,
        "learning_rate": 0.02
    }
    for i, num_knots in enumerate(knots):
        parameters["num_knots"] = num_knots
        for j, num_samples in enumerate(nums_samples):
            parameters["local_sample_num"] = num_samples
            for k, flow_iterations in enumerate(iters):
                print("=======================")
                print(knots[i], nums_samples[j], iters[k])
                parameters["flow_iterations"] = flow_iterations
                run(setup_name=".", case_name="case1", show_plot=True,
                    compute_mmd=False, incremental_step=1, ordering="pose_first", **parameters)
