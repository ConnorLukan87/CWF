cmake_minimum_required(VERSION 3.17)

enable_language(CUDA)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(CMAKE_VERBOSE_MAKEFILE ON)
add_executable(MAIN main.cpp ${PROJECT_SOURCE_DIR}/src/CVTLike/kernels.cu)

target_link_libraries(MAIN Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike cuda)
target_include_directories(MAIN PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)



add_executable(cwf cwf.cpp ${PROJECT_SOURCE_DIR}/src/CVTLike/kernels.cu)
target_link_libraries(cwf Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike cuda)
target_include_directories(cwf PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
