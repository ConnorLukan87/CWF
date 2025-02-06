#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <time.h>
#include <cuda_runtime.h>

#include "BGAL/CVTLike/CVT.h"
#include "BGAL/CVTLike/kernels.h"
#include "BGAL/Algorithm/BOC/BOC.h"
#include "BGAL/Integral/Integral.h"
#include "BGAL/Optimization/LinearSystem/LinearSystem.h"


#include <CGAL/Simple_cartesian.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/IO/OBJ.h>

#include <igl/gaussian_curvature.h>
#include <igl/massmatrix.h>
#include <igl/invert_diag.h>
#include <igl/readOFF.h>
#include <igl/writeOBJ.h>
#include <igl/writeOFF.h>
#include <igl/avg_edge_length.h>
#include <igl/cotmatrix.h>
#include <igl/invert_diag.h>
#include <igl/massmatrix.h>
#include <igl/parula.h>
#include <igl/per_corner_normals.h>
#include <igl/per_face_normals.h>
#include <igl/per_vertex_normals.h>
#include <igl/principal_curvature.h>
#include <igl/read_triangle_mesh.h>



typedef CGAL::Simple_cartesian<double> K_T;
typedef K_T::FT FT;
typedef K_T::Point_3 Point_T;

typedef K_T::Segment_3 Segment;
typedef CGAL::Polyhedron_3<K_T> Polyhedron;
typedef CGAL::AABB_face_graph_triangle_primitive<Polyhedron> Primitive;
typedef CGAL::AABB_traits<K_T, Primitive> Traits;
typedef CGAL::AABB_tree<Traits> Tree;
typedef Tree::Point_and_primitive_id Point_and_primitive_id;
double gamma = 0.00000000000001;
struct MyPoint
{
	MyPoint(Eigen::Vector3d a)
	{
		p = a;

	}

	MyPoint(double a, double b, double c)
	{
		p.x() = a;
		p.y() = b;
		p.z() = c;
	}
	Eigen::Vector3d p;

	bool operator<(const MyPoint& a) const
	{



		double dis = (p - a.p).norm();
		if (dis < gamma)
		{
			return false;
		}

		if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001)
		{
			if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001)
			{
				return (p.z() < a.p.z());
			}
			return (p.y() < a.p.y());
		}
		return (p.x() < a.p.x());



	}
	bool operator==(const MyPoint& a) const
	{
		if ((p.x() - a.p.x()) < 0.00000000001 && (p.x() - a.p.x()) > -0.00000000001)
		{
			if ((p.y() - a.p.y()) < 0.00000000001 && (p.y() - a.p.y()) > -0.00000000001)
			{
				if ((p.z() - a.p.z()) < 0.00000000001 && (p.z() - a.p.z()) > -0.00000000001)
				{
					return 1;
				}
			}

		}
		return 0;
	}
};

struct MyFace
{
	MyFace(Eigen::Vector3i a)
	{
		p = a;
	}
	MyFace(int a, int b, int c)
	{
		p.x() = a;
		p.y() = b;
		p.z() = c;
	}
	Eigen::Vector3i p;
	bool operator<(const MyFace& a) const
	{
		if (p.x() == a.p.x())
		{
			if (p.y() == a.p.y())
			{
				return p.z() > a.p.z();
			}
			return p.y() > a.p.y();
		}
		return p.x() > a.p.x();
	}
};



namespace BGAL
{
	_CVT3D::_CVT3D(const _ManifoldModel& model) : _model(model), _RVD(model), _RVD2(model), _para()
	{
		_rho = [](BGAL::_Point3& p)
		{
			return 1;
		};
		_para.is_show = true;
		_para.epsilon = 1e-30;
		_para.max_linearsearch = 20;
	}
	_CVT3D::_CVT3D(const _ManifoldModel& model, std::function<double(_Point3& p)>& rho, _LBFGS::_Parameter para) : _model(model), _RVD(model), _RVD2(model), _rho(rho), _para(para)
	{

	}
	void OutputMesh(std::vector<_Point3>& sites, _Restricted_Tessellation3D RVD, int num, std::string outpath, std::string modelname, int step)
	{
		const std::vector<std::vector<std::tuple<int, int, int>>>& cells = RVD.get_cells_();
		std::string filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		if (step == 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_RVD.obj";
		}

		if (step > 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_RVD.obj";
		}
		std::cout << "filepath = " << filepath << std::endl;
		std::ofstream out(filepath);
		out << "g 3D_Object\nmtllib BKLineColorBar.mtl\nusemtl BKLineColorBar" << std::endl;
		for (int i = 0; i < RVD.number_vertices_(); ++i)
		{
			out << "v " << RVD.vertex_(i) << std::endl;
		}
		double totarea = 0, parea = 0;
		for (int i = 0; i < cells.size(); ++i)
		{
			double area = 0;
			for (int j = 0; j < cells[i].size(); ++j)
			{
				BGAL::_Point3 p1 = RVD.vertex_(std::get<0>(cells[i][j]));
				BGAL::_Point3 p2 = RVD.vertex_(std::get<1>(cells[i][j]));
				BGAL::_Point3 p3 = RVD.vertex_(std::get<2>(cells[i][j]));
				area += (p2 - p1).cross_(p3 - p1).length_() / 2;
			}
			totarea += area;

			double color = (double)BGAL::_BOC::rand_();
			if (i > cells.size() / 3)
			{
				if (step == 1)
				{
					color = 0;
				}
				//
			}
			else
			{
				parea += area;
			}

			out << "vt " << color << " 0" << std::endl;


			for (int j = 0; j < cells[i].size(); ++j)
			{
				out << "f " << std::get<0>(cells[i][j]) + 1 << "/" << i + 1
					<< " " << std::get<1>(cells[i][j]) + 1 << "/" << i + 1
					<< " " << std::get<2>(cells[i][j]) + 1 << "/" << i + 1 << std::endl;
			}
		}
		out.close();


		filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		if (step == 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Points.xyz";
		}

		if (step > 2)
		{
			filepath = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "_Iter" + std::to_string(step - 3) + "_Points.xyz";
		}

		std::ofstream outP(filepath);

		int outnum = sites.size();
		if (step == 1)
			outnum = sites.size() / 3;

		for (int i = 0; i < outnum; ++i)
		{
			outP << sites[i] << std::endl;
		}
		outP.close();


		if (step >= 2)
		{
			std::string filepath = outpath + "\\Ours_" + std::to_string(num) + "_" + modelname + "_Remesh.obj";


			std::string	filepath1 = outpath + "Ours_" + std::to_string(num) + "_" + modelname + "Iter" + std::to_string(step - 3) + "_Remesh.obj";
			std::ofstream outRDT(filepath);
			std::ofstream outRDT1(filepath1);

			auto Vs = sites;
			auto Edges = RVD.get_edges_();
			std::set<std::pair<int, int>> RDT_Edges;
			std::vector<std::set<int>> neibors;
			neibors.resize(Vs.size());
			for (int i = 0; i < Edges.size(); i++)
			{
				for (auto ee : Edges[i])
				{
					RDT_Edges.insert(std::make_pair(std::min(i, ee.first), std::max(i, ee.first)));
					neibors[i].insert(ee.first);
					neibors[ee.first].insert(i);
					//std::cout << ee.first << std::endl;

				}
			}

			for (auto v : Vs)
			{
				if (step >= 2)
					outRDT << "v " << v << std::endl;
				outRDT1 << "v " << v << std::endl;
			}

			std::set<MyFace> rdtFaces;

			for (auto e : RDT_Edges)
			{
				for (int pid : neibors[e.first])
				{
					if (RDT_Edges.find(std::make_pair(std::min(pid, e.first), std::max(pid, e.first))) != RDT_Edges.end())
					{
						if (RDT_Edges.find(std::make_pair(std::min(pid, e.second), std::max(pid, e.second))) != RDT_Edges.end())
						{
							int f1 = pid, f2 = e.first, f3 = e.second;

							int mid;
							if (f1 != std::max(f1, std::max(f2, f3)) && f1 != std::min(f1, min(f2, f3)))
							{
								mid = f1;
							}
							if (f2 != std::max(f1, std::max(f2, f3)) && f2 != std::min(f1, std::min(f2, f3)))
							{
								mid = f2;
							}
							if (f3 != std::max(f1, max(f2, f3)) && f3 != std::min(f1, min(f2, f3)))
							{
								mid = f3;
							}
							rdtFaces.insert(MyFace(std::max(f1, std::max(f2, f3)), mid, std::min(f1, std::min(f2, f3))));
						}
					}
				}
			}
			for (auto f : rdtFaces)
			{
				if (step >= 2)
					outRDT << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << std::endl;
				outRDT1 << "f " << f.p.x() + 1 << " " << f.p.y() + 1 << " " << f.p.z() + 1 << std::endl;
			}

			outRDT.close();
			outRDT1.close();

		}



	}

void _CVT3D::calculate_(int num_sites, char* modelNamee, char* pointsName)
{
	// pointsName is base path here
	double allTime = 0, RVDtime = 0;
	clock_t start, end;
	clock_t startRVD, endRVD;

	std::string filepath = "../../data/";
	double PI = 3.14159265358;
	std::string modelname = modelNamee;
	Polyhedron polyhedron;
	std::ifstream input(std::string(pointsName) + "MAIN/Temp.off");
	input >> polyhedron;
	Tree tree(faces(polyhedron).first, faces(polyhedron).second, polyhedron);

	int NUM_VERTICES = polyhedron.size_of_vertices();

	double Movement = 0.01;
	std::string inPointsName = std::string(pointsName) + std::string(modelNamee);
	std::ifstream inPoints(inPointsName.c_str());

	std::vector<Eigen::Vector3d> Pts, Nors;

	int count = 0;
	double x, y, z, nx, ny, nz; // if xyz file has normal
	while (inPoints >> x >> y >> z >> nx >> ny >> nz) {
		Pts.push_back(Eigen::Vector3d(x, y, z));
		Nors.push_back(
			Eigen::Vector3d(nx, ny, nz)); // Nors here is useless, if do not have normal, just set it to (1,0,0)
		++count;
	}
	inPoints.close();
	std::cout << "Pts.size(): " << Pts.size() << std::endl;

	if (pointsName != nullptr) {
		num_sites = count;
	}
	// begin step 1.
	int num = Pts.size();

	std::vector<Eigen::Vector3d> Pts3;
	std::cout << "\nBegin CWF.\n" << std::endl;

	int MAX_THREADS_AVAILABLE = omp_get_max_threads();
	int Fnum = 4;
	double alpha = 1.0, eplison = 1, lambda = 1; // eplison is CVT weight,  lambda is qe weight.
	double decay = 0.95;
	
	int grid_size, block_size;
	get_launch_params(&grid_size, &block_size);

	int MAX_THREADS_PER_BATCH = (grid_size * block_size) / 5;
	MAX_THREADS_PER_BATCH *= 5; // make it a multiple of 5
	int THREADS_PER_TRIANGLE = 5;

	int* triangleID_to_vertexIDs_host, * triangleID_to_site_host;
	double* vertexIndice_to_vertex_host, * sites_host, * normals_vec_host, * area_vec_host;
	double* vertexIndice_to_vertex_device, * sites_device, * r_vec_host;
	int* triangleID_to_vertexIDs_device, * triangleID_to_site_device;
	double* normals_vec_device, * area_vec_device, * r_vec_device, * gi_vec;
	std::function<double(const Eigen::VectorXd& X, Eigen::VectorXd& g)> fgm2
		= [&](const Eigen::VectorXd& X, Eigen::VectorXd& g)
		{
			eplison = eplison * decay;
			double lossCVT = 0, lossQE = 0, totalLoss = 0;

			startRVD = clock();
			for (int i = 0; i < num; ++i)
			{
				Point_T query(X(i * 3), X(i * 3 + 1), X(i * 3 + 2)); //project to base surface
				Point_T closest = tree.closest_point(query);
				auto tri = tree.closest_point_and_primitive(query);

				Polyhedron::Face_handle f = tri.second;
				auto p1 = f->halfedge()->vertex()->point();
				auto p2 = f->halfedge()->next()->vertex()->point();
				auto p3 = f->halfedge()->next()->next()->vertex()->point();
				Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
				Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
				Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
				Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
				N.normalize();
				Nors[i] = N;
				BGAL::_Point3 p(closest.x(), closest.y(), closest.z());
				_sites[i] = p;
			}
			_RVD.calculate_(_sites);
			Fnum++;

			if (Fnum % 1 == 0)
			{
				OutputMesh(_sites, _RVD, num_sites, outpath, modelname, Fnum); //output process
			}
			endRVD = clock();
			RVDtime += (double)(endRVD - startRVD) / CLOCKS_PER_SEC;

			const std::vector<std::vector<std::tuple<int, int, int>>>& cells = _RVD.get_cells_();
			const std::vector<std::map<int, std::vector<std::pair<int, int>>>>& edges = _RVD.get_edges_();
			double energy = 0.0;
			g.setZero();
			std::vector<Eigen::Vector3d> gi;
			gi.resize(num);
			int NUM_TRIANGLES = 0;
			for (int i = 0; i < num; ++i)
			{
				gi[i] = Eigen::Vector3d(0, 0, 0);
				NUM_TRIANGLES += cells[i].size();
			}
			NUM_VERTICES = _RVD.number_vertices_();


			area_vec_host = new double[NUM_TRIANGLES];
			normals_vec_host = new double[3 * NUM_TRIANGLES];
			sites_host = new double[num * 3];
			triangleID_to_site_host = new int[NUM_TRIANGLES];
			triangleID_to_vertexIDs_host = new int[3 * NUM_TRIANGLES];
			vertexIndice_to_vertex_host = new double[NUM_VERTICES * 3];

			int triangles_before_here = 0, id0, id1, id2;
			double area;

			for (int i = 0; i < num; i++)
			{
				int NUM_TRIANGLES_HERE = cells[i].size();
				for (int j = 0; j < NUM_TRIANGLES_HERE; j++)
				{
					id0 = std::get<0>(cells[i][j]);
					id1 = std::get<1>(cells[i][j]);
					id2 = std::get<2>(cells[i][j]);
					// put this vertex into the vertexIndice_to_vertex_host
					BGAL::_Point3 p0 = _RVD.vertex_(id0);
					vertexIndice_to_vertex_host[(3 * id0) + 0] = p0.x();
					vertexIndice_to_vertex_host[(3 * id0) + 1] = p0.y();
					vertexIndice_to_vertex_host[(3 * id0) + 2] = p0.z();
					BGAL::_Point3 p1 = _RVD.vertex_(id1);
					vertexIndice_to_vertex_host[(3 * id1) + 0] = p1.x();
					vertexIndice_to_vertex_host[(3 * id1) + 1] = p1.y();
					vertexIndice_to_vertex_host[(3 * id1) + 2] = p1.z();
					BGAL::_Point3 p2 = _RVD.vertex_(id2);
					vertexIndice_to_vertex_host[(3 * id2) + 0] = p2.x();
					vertexIndice_to_vertex_host[(3 * id2) + 1] = p2.y();
					vertexIndice_to_vertex_host[(3 * id2) + 2] = p2.z();
					// get the mapping from threadId to triangle vertices
					triangleID_to_vertexIDs_host[(3 * (triangles_before_here + j)) + 0] = id0;
					triangleID_to_vertexIDs_host[(3 * (triangles_before_here + j)) + 1] = id1;
					triangleID_to_vertexIDs_host[(3 * (triangles_before_here + j)) + 2] = id2;

					triangleID_to_site_host[triangles_before_here + j] = i;
					// get vertex(id1) - vertex(id0)
					// get verrtex(id2) - vertex(id0)

					// compute the cross product
					BGAL::_Point3 cross_prod = (p1 - p0).cross_(p2 - p0);
					// compute area for this triangle
					area = cross_prod.length_();
					area_vec_host[triangles_before_here + j] = .5 * area;
					
					cross_prod.normalized_();
					normals_vec_host[3 * (j + triangles_before_here)] = cross_prod.x();
					normals_vec_host[(3 * (j + triangles_before_here)) + 1] = cross_prod.y();
					normals_vec_host[(3 * (j + triangles_before_here)) + 2] = cross_prod.z();

				}
				triangles_before_here += NUM_TRIANGLES_HERE;
				sites_host[3 * i] = _sites[i].x();
				sites_host[(3 * i) + 1] = _sites[i].y();
				sites_host[(3 * i) + 2] = _sites[i].z();
			}

			cudaMalloc((void**)&vertexIndice_to_vertex_device, sizeof(double) * 3 * NUM_VERTICES);
			cudaMalloc((void**)&sites_device, sizeof(double) * 3 * num);
			cudaMalloc((void**)&triangleID_to_vertexIDs_device, sizeof(int) * 3 * NUM_TRIANGLES);
			cudaMalloc((void**)&normals_vec_device, sizeof(double) * 3 * NUM_TRIANGLES);
			cudaMalloc((void**)&area_vec_device, sizeof(double) * NUM_TRIANGLES);
			cudaMalloc((void**)&triangleID_to_site_device, sizeof(int) * NUM_TRIANGLES);

			cudaMemcpy(triangleID_to_vertexIDs_device, triangleID_to_vertexIDs_host, sizeof(int) * 3 * NUM_TRIANGLES, cudaMemcpyHostToDevice);
			cudaMemcpy(vertexIndice_to_vertex_device, vertexIndice_to_vertex_host, sizeof(double) * 3 * NUM_VERTICES, cudaMemcpyHostToDevice);
			cudaMemcpy(normals_vec_device, normals_vec_host, sizeof(double) * 3 * NUM_TRIANGLES, cudaMemcpyHostToDevice);
			cudaMemcpy(sites_device, sites_host, sizeof(double) * 3 * num, cudaMemcpyHostToDevice);
			cudaMemcpy(area_vec_device, area_vec_host, sizeof(double) * NUM_TRIANGLES, cudaMemcpyHostToDevice);
			cudaMemcpy(triangleID_to_site_device, triangleID_to_site_host, sizeof(int) * NUM_TRIANGLES, cudaMemcpyHostToDevice);

			int current_triangle = 0;
			int current_site;
			cudaMalloc((void**)&r_vec_device, sizeof(double) * MAX_THREADS_PER_BATCH);
			r_vec_host = new double[MAX_THREADS_PER_BATCH];
			gi_vec = new double[5 * num];
			for (int i = 0; i < 5 * num; i++)
			{
				gi_vec[i] = 0.0;
			}

			for (int j = 0; j < ((THREADS_PER_TRIANGLE * NUM_TRIANGLES) + (MAX_THREADS_PER_BATCH - 1)) / MAX_THREADS_PER_BATCH; j++)
			{
				// get the current start triangle id and current end triangle id
				int start = (MAX_THREADS_PER_BATCH * j) / THREADS_PER_TRIANGLE;
				int end = std::min((j + 1) * MAX_THREADS_PER_BATCH / THREADS_PER_TRIANGLE, NUM_TRIANGLES); // min(max_end_triangle, total number of triangles)

				double r_vec[5] = { 0.0 };
				cudaMemset(r_vec_device, 0.0, sizeof(double) * MAX_THREADS_PER_BATCH);
				compute_triangle_wise4(triangleID_to_vertexIDs_device, vertexIndice_to_vertex_device, area_vec_device, normals_vec_device, r_vec_device, sites_device, triangleID_to_site_device, eplison, lambda, start, end, grid_size, block_size);
				cudaMemcpy(r_vec_host, r_vec_device, sizeof(double) * MAX_THREADS_PER_BATCH, cudaMemcpyDeviceToHost);

				for (int triangle_x = start; triangle_x < end; triangle_x++)
				{
					current_site = triangleID_to_site_host[triangle_x];
					for (int z = 0; z < 5; z++)
					{
						gi_vec[(5 * current_site) + z] += alpha * r_vec_host[(z * (end - start)) + triangle_x - start];
					}

				}

			}
			
			omp_set_num_threads(MAX_THREADS_AVAILABLE);
#pragma omp parallel for reduction(+:lossCVT) reduction(+:totalLoss)
			for (int i = 0; i < num; i++)
			{
				double dot0 = 0.0;
				double dot1 = Nors[i].dot(Nors[i]);

				for (int k = 0; k < 3; k++)
				{
					dot0 += gi_vec[(5 * i) + 2 + k] * Nors[i](k);
				}

				for (int k = 0; k < 3; k++)
				{
					gi_vec[(5 * i) + 2 + k] = gi_vec[(5 * i) + 2 + k] - (Nors[i](k) * dot0 / dot1);
					g((3 * i) + k) += gi_vec[(5 * i) + 2 + k];
				}
				lossCVT += gi_vec[5 * i];
				totalLoss += gi_vec[(5 * i) + 1];
			}

			energy += totalLoss;

			std::cout << std::setprecision(7) << "energy: " << energy << " LossCVT: " << lossCVT / eplison << " LossQE: " << totalLoss - lossCVT << " Lambda_CVT: " << eplison << std::endl;

			delete[] sites_host;
			delete[] triangleID_to_site_host;
			delete[] area_vec_host;
			delete[] normals_vec_host;
			delete[] triangleID_to_vertexIDs_host;
			delete[] vertexIndice_to_vertex_host;
			delete[] gi_vec;
			delete[] r_vec_host;
			cudaFree(r_vec_device);
			cudaFree(triangleID_to_vertexIDs_device);
			cudaFree(normals_vec_device);
			cudaFree(area_vec_device);
			cudaFree(triangleID_to_site_device);
			cudaFree(sites_device);
			cudaFree(vertexIndice_to_vertex_device);
			return energy;
		};

	std::vector<Eigen::Vector3d> Pts2;

	Pts2 = Pts;
	num = Pts2.size();
	std::cout << Pts2.size() << "  " << num << std::endl;
	_sites.resize(num);
	_para.max_linearsearch = 20;
	BGAL::_LBFGS lbfgs2(_para);
	Eigen::VectorXd iterX2(num * 3);
	for (int i = 0; i < num; ++i)
	{
		iterX2(i * 3) = Pts2[i].x();
		iterX2(i * 3 + 1) = Pts2[i].y();
		iterX2(i * 3 + 2) = Pts2[i].z();
		_sites[i] = BGAL::_Point3(Pts2[i](0), Pts2[i](1), Pts2[i](2));
	}
	_RVD.calculate_(_sites);
	start = clock();
	lbfgs2.minimize(fgm2, iterX2);
	end = clock();
	allTime += (double)(end - start) / CLOCKS_PER_SEC;
	std::cout << "allTime: " << allTime << " RVDtime: " << RVDtime << " L-BFGS time: " << allTime - RVDtime << std::endl;
	for (int i = 0; i < num; ++i)
	{
		Point_T query(iterX2(i * 3), iterX2(i * 3 + 1), iterX2(i * 3 + 2));
		Point_T closest = tree.closest_point(query);
		auto tri = tree.closest_point_and_primitive(query);

		Polyhedron::Face_handle f = tri.second;
		auto p1 = f->halfedge()->vertex()->point();
		auto p2 = f->halfedge()->next()->vertex()->point();
		auto p3 = f->halfedge()->next()->next()->vertex()->point();
		Eigen::Vector3d v1(p1.x(), p1.y(), p1.z());
		Eigen::Vector3d v2(p2.x(), p2.y(), p2.z());
		Eigen::Vector3d v3(p3.x(), p3.y(), p3.z());
		Eigen::Vector3d N = (v2 - v1).cross(v3 - v1);
		N.normalize();
		Nors[i] = N;

		_sites[i] = BGAL::_Point3(closest.x(), closest.y(), closest.z());

	}
	_RVD.calculate_(_sites);

	OutputMesh(_sites, _RVD, num_sites, outpath, modelname, 2);

}
} // namespace BGAL
