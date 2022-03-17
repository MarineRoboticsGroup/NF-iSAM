#!/bin/bash
dir="/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/manhattan_world_with_range/random_4x4/res"

for d in $dir/*/ ; do
    echo "$d"
    efg="${d}factor_graph.fg"
    echo "$efg"
    output_dir="${d}gtsam"
    echo "$output_dir"
    # the last three entries in the following command (please see src/external/gtsam/README.md for details)
    # incremental_step, artificial_prior_sigma, groud_truth_initialization
    /home/chad/codebase/NF-iSAM/src/external/gtsam/build/gtsam_solution "$efg" "$output_dir" 1 -1 0
done
