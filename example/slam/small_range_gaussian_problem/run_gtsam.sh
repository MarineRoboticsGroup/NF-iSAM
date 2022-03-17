#!/bin/bash
#dir="/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/manhattan_world_with_range/test_samplers/RO_NF_NoisyOdom_STD1/"
dir="/home/chad/codebase/NF-iSAM/example/slam/problems_for_paper/small_range_gaussian_problem/data_association/"
#dir="/home/chad/Research/optimalTransport/CouplingSLAM/example/slam/manhattan_world_with_range/random_4x4/res"

for d in $dir/*/ ; do
    echo "$d"
    efg="${d}factor_graph.fg"
    echo "$efg"
    output_dir="${d}gtsam"
    echo "$output_dir"
    /home/chad/codebase/NF-iSAM/src/external/gtsam/build/gtsam_solution "$efg" "$output_dir" 1 -1 0
done
