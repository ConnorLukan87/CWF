cmake_minimum_required(VERSION 3.17)
# set(CMAKE_TOOLCHAIN_FILE C:/dev/vcpkg/scripts/buildsystems/vcpkg.cmake)
project(BGAL LANGUAGES CXX CUDA)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_BUILD_TYPE "Debug")

# Default Build_type RELEASE
set(CMAKE_BUILD_TYPE RELEASE)
# CXX STANDARD
set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# Set Version
set(VERSION 1.0)
# Set lib output directory
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${PROJECT_SOURCE_DIR}/lib)
#set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

option(CMAKE_INSTALL_PREFIX "BGAL install PATH" /usr/local)
# include header from project_source_dir
include_directories(${PROJECT_SOURCE_DIR}/include)

find_path( LIBIGL_INCLUDE "igl/ZERO.h")
include_directories( ${LIBIGL_INCLUDE})

# Get Eigen3
find_package(Eigen3 REQUIRED)
if (Eigen3_FOUND)
    message(STATUS "${EIGEN3_VERSION_STRING} FOUND")
endif ()

# Get Boost
find_package(Boost REQUIRED)
if (Boost_FOUND)
    message(STATUS "BOOST FOUND")
    include_directories(${Boost_INCLUDE_DIRS})
endif ()

# Get CGAL
find_package(CGAL REQUIRED)
if (CGAL_FOUND)
    message(STATUS "CGAL FOUND")
else ()
    message("ERROR: this program requires CGAL and will not be compiled.")
endif ()

find_package(OpenMP REQUIRED COMPONENTS CXX)

set(CUDAToolkit_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "60;70;75;80")


if(WIN32 AND NOT MINGW)
  if(NOT DEFINED CMAKE_DEBUG_POSTFIX)
    set(CMAKE_DEBUG_POSTFIX "d")
  endif()
  if(NOT DEFINED CMAKE_RELWITHDEBINFO_POSTFIX)
    set(CMAKE_RELWITHDEBINFO_POSTFIX "rd")
  endif()
  if(NOT DEFINED CMAKE_MINSIZEREL_POSTFIX)
    set(CMAKE_MINSIZEREL_POSTFIX "s")
  endif()
endif()

# sub_directory
add_subdirectory(include)
add_subdirectory(src)
#target_link_libraries(${PROJECT_NAME}  Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D ${Boost_LIBRARIES})
add_subdirectory(MAIN)

#set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")


install(TARGETS Algorithm BaseShape Draw Geodesic Integral Model Optimization PointCloudProcessing PQP Reconstruction Tessellation2D Tessellation3D CVTLike
	EXPORT ${PROJECT_NAME}Targets
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib
	RUNTIME DESTINATION bin
	PUBLIC_HEADER DESTINATION include
	)
install(DIRECTORY ./include/ DESTINATION include
        FILES_MATCHING PATTERN "*.h")
configure_file(${PROJECT_NAME}-config.cmake.in ${PROJECT_NAME}-config.cmake @ONLY)
install(EXPORT ${PROJECT_NAME}Targets
	FILE ${PROJECT_NAME}-config.cmake
	NAMESPACE ${PROJECT_NAME}::
	DESTINATION lib/cmake/${PROJECT_NAME}
	)
