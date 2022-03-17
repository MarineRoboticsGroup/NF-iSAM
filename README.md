# Normalizing flow incremental smoothing and mapping (NF-iSAM)
This is the codebase of NF-iSAM. We also put together here some examples and scripts for testing the performance of NF-iSAM and other solvers (mm-iSAM, GTSAM, and nested sampling) that are in comparison.

The following instruction was tested on Ubuntu 18.04 with Miniconda.

## Requirements on Ubuntu
```
sudo apt-get install gcc libc6-dev
sudo apt-get install gfortran libgfortran3
sudo apt-get install libsuitesparse-dev
```
We recommend to install NF-iSAM using conda environment. The default env name in the environment.yml is NFiSAM.

## Installation
```
git clone git@github.com:MarineRoboticsGroup/NF-iSAM.git
cd nsfg
conda env create -f environment.yml
conda activate nsfg
pip3 install --upgrade TransportMaps
pip3 install -r requirements.txt
python setup.py install
```

## Examples
### NF-iSAM
In `example/slam`, the file folder `toy_examples` contains Python scripts that call NF-iSAM with hardcoded factor graphs in the scripts. The other folders store factor graphs by files `factor_graph.fg` and you can find `run_nfisam.py` in those folders and solve corresponding factor graphs. All results will be stored in the file folder that contains the factor graph file `factor_graph.fg`. For example, to run one of the minimal examples in this codebase, you can
```
cd example/slam/small_range_gaussian_problem
python run_nfisam.py
```

### Nested sampling
To use nested sampling to directly draw samples from posteriors encoded in the factor graph of the minimal example, you can run
```
cd example/slam/small_range_gaussian_problem
python run_nested_sampling.py
```

### GTSAM
We also prepared C++ scripts to parse our factor graphs for [GTSAM](https://github.com/borglab/gtsam). To solve our factor graphs using GTSAM, you have to install [GTSAM](https://github.com/borglab/gtsam) first and then build our parser by running
```
cd src/external/gtsam
mkdir build
cd build
cmake ..
make
```
Now you should see `gtsam_solution` in the build folder. Go back to the example folder and modify the path of `gtsam_solution` accordingly in the bash script `run_gtsam.sh`. If all steps are completed for the small example used above, to get GTSAM solutions, you can run
```
cd example/slam/small_range_gaussian_problem
source run_gtsam.sh
```

### mm-iSAM
To solve our factor graphs using mm-iSAM provided by [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/), please follow the instruction in [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/) to install `Julia` and [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/) first. And then you need to modify the `parent_path` variable in `src/external/caesar/fg2caesar.jl` to adapt the directory on your computer. Now you should be ready to solve the minimal example using [Caesar.jl](https://juliarobotics.org/Caesar.jl/latest/) by running
```
cd src/external/caesar
julia fg2caesar.jl
```
