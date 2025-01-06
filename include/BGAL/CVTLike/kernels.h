#ifndef KERNELS_H
#define KERNELS_H
#include <cuda_runtime.h>

void compute_triangle_wise4(int* triangleID_to_vertex_indices, double* vertexIndice_to_vertex, double* area_vec, double* normals, double* rs, double* sites, int* triangleID_to_site, double epsilon, double lambda, int START_TRIANGLE, int END_TRIANGLE, int gridSize, int blockSize);

void get_launch_params(int* grid_size, int* block_size);

#endif