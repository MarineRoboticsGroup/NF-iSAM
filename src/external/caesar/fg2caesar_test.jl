# Copyright 2021 Massachusetts Institute of Technology
# @file R2RANGE example
# @author: Qiangqiang Huang

# constructor for variables and factors
struct my_var
  name
  type
  manifold
  gt_arr
end

struct f_unary
  var
  f_type
  obs
  cov_arr
end

struct f_bin
  var1
  var2
  f_type
  obs
  cov_arr
end

struct f_kary
  var1
  var2s
  binary
  weights
  obs
  cov_arr
  f_type
end

struct incVarFactor
  vars
  factors
end

function get_kw_idx(str_arr, idx, keyword)
  kw_idx = 0
  for i = idx:length(str_arr)
    if cmp(str_arr[i], keyword) == 0
      kw_idx = i
      break
    end
  end
  return kw_idx
end

# strings for parser
str_var = "Variable"
str_factor = "Factor"
str_rbt = "Pose"
str_lmk = "Landmark"
str_f_p2p = "SE2RelativeGaussianLikelihoodFactor"
str_f_range = "SE2R2RangeGaussianLikelihoodFactor"
str_f_ada = "AmbiguousDataAssociationFactor"
str_f_prior = "UnarySE2ApproximateGaussianPriorFactor"

using Caesar
using Gadfly

# using RoME, Distributions
# using GraphPlot
using RoMEPlotting
Gadfly.set_default_plot_size(35cm,20cm)

# using Rebugger
using NPZ
using MAT

# # add multiproc support
using Distributed
addprocs(8)
@everywhere using Caesar, RoME

# # 90% precompile
for i in 1:5
  warmUpSolverJIT()
end

kde_sample = 100
post_sample = 1000
inc_step = 1
case_f = "/home/chad/codebase/NF-iSAM/example/slam/problems_for_paper/small_range_gaussian_problem/data_association/test_caesar"

fg_path = string(case_f, "/factor_graph.fg")
println(fg_path)
cnt=1
output_dir = string(case_f,"/caesar",cnt)
while true
    if !isdir(output_dir)
        mkpath(output_dir)
        break
    else
        println(output_dir)
        cnt = cnt + 1
        output_dir = string(case_f,"/caesar",cnt)
    end
end

name2var = Dict()
lmknames = []
rbtnames = []

# read factor graph
lines = readlines(fg_path)

println(lines)

factor_arr = []
p2p_idx = []
range_idx = []
ada_idx = []
nh_idx = []
prior_idx = []

factor_idx = 0
for l in lines
    str_arr = split(l,' ')
    if cmp(str_arr[1], str_var) == 0
    # parsing vriables
    gt_arr = [parse(Float64, str_arr[i]) for i in 5:length(str_arr)]
    tmp_var = my_var(str_arr[4],str_arr[2],str_arr[3],gt_arr)
    merge!(name2var, Dict(tmp_var.name=>tmp_var))
    if cmp(tmp_var.type, str_rbt) == 0
        push!(rbtnames, tmp_var.name)
    elseif cmp(tmp_var.type, str_lmk) == 0
        push!(lmknames, tmp_var.name)
    end
    elseif cmp(str_arr[1], str_factor) == 0
    factor_idx += 1
    n_f = 0
    if cmp(str_arr[2], str_f_p2p) == 0
        first_obs_idx = 5
        cov_idx = get_kw_idx(str_arr, 4, "covariance")
        obs_arr = [parse(Float64, str_arr[i]) for i in first_obs_idx:cov_idx-1]
        cov_arr = [parse(Float64, str_arr[i]) for i in cov_idx+1:length(str_arr)]
        n_f = f_bin(str_arr[3],str_arr[4], str_arr[2], obs_arr, cov_arr)
        push!(p2p_idx, factor_idx)
    elseif cmp(str_arr[2], str_f_prior) == 0
        cov_idx = get_kw_idx(str_arr, 4, "covariance")
        obs_arr = [parse(Float64, str_arr[i]) for i in 4:cov_idx-1]
        cov_arr = [parse(Float64, str_arr[i]) for i in cov_idx+1:length(str_arr)]
        n_f = f_unary(str_arr[3], str_f_prior, obs_arr, cov_arr)
        push!(prior_idx, factor_idx)
    elseif cmp(str_arr[2], str_f_range) == 0
        first_obs_idx = 5
        obs_arr = [parse(Float64, str_arr[first_obs_idx])]
        cov_arr = [parse(Float64, str_arr[first_obs_idx+1])^2]
        n_f = f_bin(str_arr[3],str_arr[4], str_arr[2], obs_arr, cov_arr)
        push!(range_idx, factor_idx)
    elseif cmp(str_arr[2], str_f_ada) == 0
        va1_idx = get_kw_idx(str_arr, 1, "Observer")
        va2_idx = get_kw_idx(str_arr, 1, "Observed")
        w_idx = get_kw_idx(str_arr, 1, "Weights")
        f_idx = get_kw_idx(str_arr, 1, "Binary")
        obs_idx = get_kw_idx(str_arr, 1, "Observation")
        sig_idx = get_kw_idx(str_arr, 1, "Sigma")
        var2_arr = [str_arr[i] for i=va2_idx+1:w_idx-1]
        w_arr = [parse(Float64, str_arr[i]) for i=w_idx+1:f_idx-1]
        obs_arr = [parse(Float64, str_arr[i]) for i=obs_idx+1:sig_idx-1]
        if cmp(str_arr[f_idx+1], str_f_range) == 0
        cov_arr = [parse(Float64, str_arr[i])^2 for i=sig_idx+1:length(str_arr)]
        else
        cov_arr = [parse(Float64, str_arr[i]) for i=sig_idx+1:length(str_arr)]
        end
        n_f = f_kary(str_arr[va1_idx+1],var2_arr, str_arr[f_idx+1],w_arr, obs_arr, cov_arr, str_f_ada)
        push!(ada_idx, factor_idx)
    else
        throw(ErrorException("Unknown factor type"))
    end
    push!(factor_arr, n_f)
    end
end



# group factors and variables incrementally
# //return values and indices of factors
incVarFactorPairs = []
# vector<pair<Values, vector<int>>> res;

if (inc_step == 0 || inc_step > length(rbtnames))
    inc_step = rbtnames.size();
    println("Reset inc_step to ", inc_step)
end

newVars = []
newFactors = []
addedRbts = []
addedLmks = []
# incremental updates with robot vars
for i in 1:length(rbtnames)
    push!(newVars, name2var[rbtnames[i]])
    push!(addedRbts, rbtnames[i])

    tmp_factor_idx = []
    for j in prior_idx
        if cmp(factor_arr[j].var, rbtnames[i]) == 0
            push!(tmp_factor_idx, j)
            println("Push a prior factor at ", rbtnames[i])
        end
    end
    prior_idx = setdiff(prior_idx, tmp_factor_idx)
    union!(newFactors, tmp_factor_idx)

    tmp_factor_idx = []
    for j in p2p_idx
        tmp_vars = [factor_arr[j].var1, factor_arr[j].var2]
        if issubset(tmp_vars, addedRbts)
            push!(tmp_factor_idx, j)
            println("Push a p2p factor between ", factor_arr[j].var1, factor_arr[j].var2)
        end
    end
    if length(tmp_factor_idx) == 0 && length(addedRbts) > 1
        throw(ErrorException("No pose2pose factors for the newly added robot variable."))
    else
        p2p_idx = setdiff(p2p_idx, tmp_factor_idx)
        union!(newFactors, tmp_factor_idx)
    end

    tmp_factor_idx = []
    for j in range_idx
        lmk_var = setdiff([factor_arr[j].var1, factor_arr[j].var2], [rbtnames[i]])
        len_lmk_var = length(lmk_var)
        if len_lmk_var == 1
            lmk_var = lmk_var[1]
            if ~(lmk_var in addedLmks)
                push!(newVars, name2var[lmk_var])
                push!(addedLmks, lmk_var)
            end
            push!(tmp_factor_idx, j)
            println("Push a range factor between ", factor_arr[j].var1, factor_arr[j].var2)
        elseif len_lmk_var == 2
            ;
        else
            throw(ErrorException("Not a binary factor."))
        end
    end
    range_idx = setdiff(range_idx, tmp_factor_idx)
    union!(newFactors, tmp_factor_idx)

    tmp_factor_idx = []
    for j in ada_idx
        if cmp(factor_arr[j].var1, rbtnames[i]) == 0
            if ~(issubset(factor_arr[j].var2s, addedRbts) || issubset(factor_arr[j].var2s, addedLmks))
                throw(ErrorException("Unobserved var is associated."))
            end
            push!(tmp_factor_idx, j)
            println("Push an ADA factor between ", factor_arr[j].var1, factor_arr[j].var2s)
        end
    end
    ada_idx = setdiff(ada_idx, tmp_factor_idx)
    union!(newFactors, tmp_factor_idx)

    # TODO: add null hypothesis

    if( i%inc_step == 0)
        inc_data = incVarFactor(deepcopy(newVars), deepcopy(newFactors))
        push!(incVarFactorPairs, inc_data)
        empty!(newVars)
        empty!(newFactors)
        println("New batch loaded.")
    end
end
println("There are ", length(incVarFactorPairs), " pairs of vars and factors.")

N = post_sample
kde_N = kde_sample
# start with an empty factor graph object
fg = initfg()
init_tree = false
tree = 1
# tree = wipeBuildNewTree!(fg)
# fieldnames(typeof(getSolverParams(fg)))
# getSolverParams(fg).graphinit = true  # init on graph first then solve on tree (default)
# getSolverParams(fg).graphinit = false # init and solve on tree
# getSolverParams(fg).useMsgLikelihoods = true
getSolverParams(fg).N = kde_N

timing = zeros(0)
added_syms = Vector{Symbol}(undef, 0)
for i in 1:length(incVarFactorPairs)
    cur_vars = incVarFactorPairs[i].vars
    cur_factors = incVarFactorPairs[i].factors
    # Add at a fixed location Prior to pin :x0 to a starting location (0,0,0)
    s_time = time_ns()
    for var in cur_vars
        if ~(var in added_syms)
        push!(added_syms, Symbol(var.name))
        if cmp(var.manifold, "SE2") == 0
            addVariable!(fg, Symbol(var.name), Pose2)
        elseif cmp(var.manifold, "R2") == 0
            addVariable!(fg, Symbol(var.name), Point2)
        else
            throw(ErrorException("Unknown manifold"))
        end
        end
    end

    for j in cur_factors
        f = factor_arr[j]
        if cmp(f.f_type, str_f_prior) == 0
            addFactor!(fg, [Symbol(f.var)], PriorPose2( MvNormal(f.obs, reshape(f.cov_arr,3,3)) ))
        elseif cmp(f.f_type, str_f_range) == 0
            addFactor!(fg, [Symbol(f.var1), Symbol(f.var2)], Pose2Point2Range(MvNormal(f.obs, diagm(f.cov_arr))))
        elseif cmp(f.f_type, str_f_p2p) == 0
            addFactor!(fg, [Symbol(f.var1), Symbol(f.var2)], Pose2Pose2(MvNormal(f.obs, reshape(f.cov_arr,3,3))))
        elseif cmp(f.f_type, str_f_ada) == 0
            symbols = [Symbol(f.var1)]
            hypos = [1.0]
            for (v_i, v) in enumerate(f.var2s)
                push!(symbols, Symbol(v))
                push!(hypos, f.weights[v_i])
            end
            # push!(symbols, Symbol("o$(j)"))
            # push!(hypos, 1.0)
            if cmp(f.binary, str_f_p2p) == 0
                # addVariable!(fg, Symbol("o$(j)"), Pose2)
                addFactor!(fg, symbols, Pose2Pose2(MvNormal(f.obs, reshape(f.cov_arr,3,3))), multihypo=hypos)
            elseif cmp(f.binary, str_f_range) == 0
                # addVariable!(fg, Symbol("o$(j)"), ContinuousScalar)
                addFactor!(fg, symbols, Pose2Point2Range(MvNormal(f.obs, diagm(f.cov_arr))), multihypo=hypos)
            else
                throw(ErrorException("Unknown binary type."))
            end
        else
            throw(ErrorException("Unknown factor types."))
        end
    end

    ## Perform inference
    if !init_tree
        tree, smt, hist = solveTree!(fg)
        init_tree = true
    else
        tree, smt, hist = solveTree!(fg, tree)
    end
    e_time = time_ns()
    elapsed = (e_time - s_time)/10^9
    append!(timing, elapsed)
    println("elapsed time: ", elapsed)

    # TODO: save variable order
    pl = plotKDE(fg, added_syms,dims=[1;2], levels=4)
    Gadfly.draw(PDF(joinpath(output_dir,"step$(i-1).pdf"), 20cm, 10cm),pl)
    res = rand(getBelief(fg, added_syms[1]) ,N)
    var_names = [string(added_syms[1])]
    if length(added_syms)>1
        for s in added_syms[2:end]
        res = vcat(res, rand(getBelief(fg, s) ,N))
        push!(var_names, string(s))
        end
    end
    npzwrite(joinpath(output_dir,"step$(i-1).npz"),res)
    outfile = joinpath(output_dir,"step$(i-1)_ordering")

    open(outfile, "w") do io
        for i in var_names # or for note in notes
            println(io, i)
        end
    end
end
npzwrite(joinpath(output_dir,"timing.npz"),timing)
