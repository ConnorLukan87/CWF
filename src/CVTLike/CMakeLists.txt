cmake_minimum_required(VERSION 3.17)

enable_language(CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)

set(BGAL_CVTLike_SRC
		CPD.cpp
		CVT.cpp kernels.cu)

set_source_files_properties(kernels.cu PROPERTIES LANGUAGE CUDA)
# Get static lib

add_library(CVTLike STATIC ${BGAL_CVTLike_SRC})

set_target_properties(CVTLike PROPERTIES CUDA_SEPARABLE_COMPILATION ON 
CUDA_STANDARD 14
CUDA_ARCHITECTURES "60;70;75;80")

target_link_libraries(CVTLike Algorithm BaseShape Model Tessellation2D Tessellation3D Optimization OpenMP::OpenMP_CXX CGAL::CGAL ${Boost_LIBRARIES} CUDA::cudart)
set_target_properties(CVTLike PROPERTIES VERSION ${VERSION})
set_target_properties(CVTLike PROPERTIES CLEAN_DIRECT_OUTPUT 1)


add_definitions(-DNOMINMAX)

target_include_directories(CVTLike PUBLIC
	$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
	$<INSTALL_INTERFACE:include>)
