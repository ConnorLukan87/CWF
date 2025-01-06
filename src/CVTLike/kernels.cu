#include <cuda_runtime.h>
#include "BGAL/CVTLike/kernels.h"

__global__ void compute_triangle_stuff4(double weight, double w_1, double w_2, double w_3, int* triangleID_to_vertex_indices, double* vertexIndice_to_vertex, double* area_vec, double* normals, double* rs, double* sites, int* triangleID_to_sites, double epsilon, double lambda, int START_TRIANGLE, int END_TRIANGLE)
{
    int threadId = (blockDim.x * blockIdx.x) + threadIdx.x; // 5*Numtriangles many

    int NUM_TRIANGLES = END_TRIANGLE - START_TRIANGLE;
    if (threadId >= 5 * NUM_TRIANGLES) return;

    int triangleID = threadId % NUM_TRIANGLES; // there will be 5 threads with the same triangleID
    int triangleID_index = START_TRIANGLE + triangleID; // used for index in the triangleID_to_vertex_indices array

    int vertex_i0 = triangleID_to_vertex_indices[3 * triangleID_index];
    int vertex_i1 = triangleID_to_vertex_indices[3 * triangleID_index + 1];
    int vertex_i2 = triangleID_to_vertex_indices[3 * triangleID_index + 2];

    double pt0_x = vertexIndice_to_vertex[3 * vertex_i0];
    double pt0_y = vertexIndice_to_vertex[3 * vertex_i0 + 1];
    double pt0_z = vertexIndice_to_vertex[3 * vertex_i0 + 2];

    double pt1_x = vertexIndice_to_vertex[3 * vertex_i1];
    double pt1_y = vertexIndice_to_vertex[3 * vertex_i1 + 1];
    double pt1_z = vertexIndice_to_vertex[3 * vertex_i1 + 2];

    double pt2_x = vertexIndice_to_vertex[3 * vertex_i2];
    double pt2_y = vertexIndice_to_vertex[3 * vertex_i2 + 1];
    double pt2_z = vertexIndice_to_vertex[3 * vertex_i2 + 2];

    int site_id = triangleID_to_sites[triangleID_index];

    double x_pt = (pt0_x * w_1) + (pt1_x * w_2) + (pt2_x * w_3);
    double y_pt = (pt0_y * w_1) + (pt1_y * w_2) + (pt2_y * w_3);
    double z_pt = (pt0_z * w_1) + (pt1_z * w_2) + (pt2_z * w_3);

    double x_norm = normals[3 * triangleID_index];
    double y_norm = normals[3 * triangleID_index + 1];
    double z_norm = normals[3 * triangleID_index + 2];
 
    double v_p_x = x_pt - sites[3 * site_id];
    double v_p_y = y_pt - sites[(3 * site_id) + 1];
    double v_p_z = z_pt - sites[(3 * site_id) + 2];

    int r_index = threadId / NUM_TRIANGLES;
    if (r_index == 0) // r(0)
    {
        rs[threadId] += area_vec[triangleID_index] * weight * epsilon * ((v_p_x * v_p_x) + (v_p_y * v_p_y) + (v_p_z * v_p_z));
    }
    else if (r_index == 1) // r(1)
    {
        rs[threadId] += area_vec[triangleID_index] * weight * ((lambda * ((x_norm * v_p_x) + (y_norm * v_p_y) + (z_norm * v_p_z)) * ((x_norm * v_p_x) + (y_norm * v_p_y) + (z_norm * v_p_z))) + (epsilon * ((v_p_x * v_p_x) + (v_p_y * v_p_y) + (v_p_z * v_p_z))));
    }
    else if (r_index == 2) // r(2)
    {
        rs[threadId] += -2 * area_vec[triangleID_index] * weight * ((lambda * x_norm * ((x_norm * v_p_x) + (y_norm * v_p_y) + (z_norm * v_p_z))) + (epsilon * v_p_x));
    }
    else if (r_index == 3) // r(3)
    {
        rs[threadId] += -2 * area_vec[triangleID_index] * weight * ((lambda * y_norm * ((x_norm * v_p_x) + (y_norm * v_p_y) + (z_norm * v_p_z))) + (epsilon * v_p_y));
    }
    else if (r_index == 4) // r(4)
    {
        rs[threadId] += -2 * area_vec[triangleID_index] * weight * ((lambda * z_norm * ((x_norm * v_p_x) + (y_norm * v_p_y) + (z_norm * v_p_z))) + (epsilon * v_p_z));
    }
}

void compute_triangle_wise4(int* triangleID_to_vertex_indices, double* vertexIndice_to_vertex, double* area_vec, double* normals, double* rs, double* sites, int* triangleID_to_site, double epsilon, double lambda, int START_TRIANGLE, int END_TRIANGLE, int gridSize, int blockSize)
{
    double w_1 = .5, w_2 = .5, w_3 = .5;
    double weight = 1.0 / 30.0;
    // r1
    w_1 = 0;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);
    // r2
    w_1 = .5;
    w_3 = 0;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);
    // r3
    w_2 = 0;
    w_3 = .5;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);
    weight *= 9.0;
    // r4
    w_1 = 1.0 / 6.0;
    w_2 = 1.0 / 6.0;
    w_3 = 2.0 / 3.0;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);
    // r5
    w_2 = 2.0 / 3.0;
    w_3 = 1.0 / 6.0;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);
    // r6
    w_1 = 2.0 / 3.0;
    w_2 = 1.0 / 6.0;
    compute_triangle_stuff4 << <gridSize, blockSize >> > (weight, w_1, w_2, w_3, triangleID_to_vertex_indices, vertexIndice_to_vertex, area_vec, normals, rs, sites, triangleID_to_site, epsilon, lambda, START_TRIANGLE, END_TRIANGLE);

}

void get_launch_params(int* grid_size, int* block_size)
{
    cudaOccupancyMaxPotentialBlockSize(grid_size, block_size, compute_triangle_stuff4);
}
