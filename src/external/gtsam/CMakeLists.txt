cmake_minimum_required(VERSION 2.8.3)
project(gtsam_solution)


find_package(GTSAM REQUIRED)

include_directories(
  include
  )
  
add_executable(gtsam_solution gtsam_solution.cpp)

target_link_libraries(gtsam_solution gtsam)

target_include_directories(gtsam_solution PUBLIC ${GTSAM_INCLUDE_DIR})

