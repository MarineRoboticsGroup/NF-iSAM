using RoME, Distributions


# [OPTIONAL] add more julia processes to speed up inference
using Distributed
nprocs() < 3 ? addprocs(4-nprocs()) : nothing

@everywhere using RoME

# number of particles
N = 100

# start with an empty factor graph object
fg = initfg()


# Add the first pose :x0
addVariable!(fg, :x0, Pose2, N=N)
# Add a few more poses
for i in 1:5
  addVariable!(fg, Symbol("x$(i)"), Pose2, N=N)
end
# Add landmarks
for i in 1:2
  addVariable!(fg, Symbol("l$(i)"), Point2, N=N)
end
# Add at a fixed location Prior to pin :x0 to a starting location (0,0,0)
addFactor!(fg, [:x0], PriorPose2( MvNormal([0; 0; 0], Matrix(Diagonal([0.3;0.3;0.3].^2)) )))

# Step 0
ppr = Pose2Point2Range(MvNormal([7.323699045815659], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [:x0; :l1], ppr )

# Step 1
i = 1
pp = Pose2Pose2(MvNormal([9.82039106655709;-0.040893643507463134;0.7851856362659602], Matrix(Diagonal([0.3;0.3;0.05].^2))))
ppr = Pose2Point2Range(MvNormal([6.783355788017825], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [Symbol("x$(i-1)"); Symbol("x$(i)")], pp )
addFactor!(fg, [Symbol("x$(i)"); Symbol("l1")], ppr )

# Step 2
i = i + 1
pp = Pose2Pose2(MvNormal([9.824301116341609;-0.1827443713759503;0.7586281983933238], Matrix(Diagonal([0.3;0.3;0.05].^2))))
ppr = Pose2Point2Range(MvNormal([6.768864478545091], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [Symbol("x$(i-1)"); Symbol("x$(i)")], pp )
addFactor!(fg, [Symbol("x$(i)"); Symbol("l2")], ppr )

# Step 3
i = i + 1
pp = Pose2Pose2(MvNormal([9.776502885351334; -0.010587078502017132; 1.5591793408311467], Matrix(Diagonal([0.3;0.3;0.05].^2))))
ppr = Pose2Point2Range(MvNormal([7.401417053438512], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [Symbol("x$(i-1)"); Symbol("x$(i)")], pp )
addFactor!(fg, [Symbol("x$(i)"); Symbol("l2")], ppr )

# Step 4
i = i + 1
pp = Pose2Pose2(MvNormal([9.644657137571507; 0.5847494762836476; 0.7422440549101994], Matrix(Diagonal([0.3;0.3;0.05].^2))))
ppr = Pose2Point2Range(MvNormal([7.331883435735532], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [Symbol("x$(i-1)"); Symbol("x$(i)")], pp )
addFactor!(fg, [Symbol("x$(i)"); Symbol("l2")], ppr )

# Step 5
i = i + 1
pp = Pose2Pose2(MvNormal([9.725096752125593; 0.6094800276622434; 0.7750400402527422], Matrix(Diagonal([0.3;0.3;0.05].^2))))
ppr = Pose2Point2Range(MvNormal([6.476782344345081], Matrix(Diagonal([0.3].^2))))
addFactor!(fg, [Symbol("x$(i-1)"); Symbol("x$(i)")], pp )
addFactor!(fg, [Symbol("x$(i)"); Symbol("l1")], ppr )


using Cairo, RoMEPlotting
Gadfly.set_default_plot_size(35cm, 20cm)


# draw initialized
plotSLAM2D(fg, contour=false)


for i in 1:10
# solve factor graph
tree,_,_ = solveTree!(fg, storeOld=true);


# after solve
plotSLAM2D(fg, contour=false)


pl1 = plotSLAM2DPoses(fg, to=0, drawhist=true);
X0 = IncrementalInference.getBelief(fg, :x0) |> getPoints;
# pl2 = Gadfly.layer(x=X0[1,:],y=X0[2,:], Geom.hexbin);
# you can also do
# push!(pl1.layers, pl2[1])
# pl1 # to show
# pl1 |> PDF("/tmp/test.pdf", 12cm,8cm)
  # plH = hstack(pl1, pl2)


# convert to file
p1H |> PNG("/home/chad/Pictures/julia_x0_step0.png")

#drawPoses(fg)
# If you have landmarks, you can instead call
plotSLAM2D(fg)

# # Draw the KDE for x0
# plotKDE(fg, :x0)
# # Draw the KDE's for x0 and x1
# plotKDE(fg, [:x1])
# plotKDE(fg, [:x2])
# plotKDE(fg, [:x3])
# plotKDE(fg, [:x4])
# plotKDE(fg, [:x5])
# plotKDE(fg, [:l1])
# plotKDE(fg, [:l2])
