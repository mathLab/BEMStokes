#include "bem_stokes.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_tools.h>

#include <iostream>
#include <fstream>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/utilities.h>
#include <deal.II/fe/mapping_fe_field.h>

#include <mpi.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>

using namespace deal2lkit;
using namespace dealii;
using namespace BEMStokes;

void assemble_single_line_stokes_matrices(FullMatrix<double> &V_matrix_3_line, FullMatrix<double> &K_matrix_3_line, const BEMProblem<3> &bemmy)
{
  V_matrix_3_line.reinit(3,bemmy.dh_stokes.n_dofs());
  K_matrix_3_line.reinit(3,bemmy.dh_stokes.n_dofs());
  FEValues<3-1,3> fe_stokes_v(*bemmy.mappingeul, *bemmy.fe_stokes, bemmy.quadrature,
                              update_values |
                              update_cell_normal_vectors |
                              update_quadrature_points |
                              update_JxW_values);
  std::vector<types::global_dof_index> local_dof_indices(bemmy.fe_stokes->dofs_per_cell);
  FullMatrix<double>    local_single_layer(3, bemmy.fe_stokes->dofs_per_cell);
  FullMatrix<double>    local_double_layer(3, bemmy.fe_stokes->dofs_per_cell);
  std::vector<Point<3> > support_points(bemmy.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<3-1, 3>( *bemmy.mappingeul, bemmy.dh_stokes, support_points);

  typename DoFHandler<3-1,3>::active_cell_iterator
  cell = bemmy.dh_stokes.begin_active(),
  endc = bemmy.dh_stokes.end();
  FullMatrix<double> test_V(3,3);
  FullMatrix<double> test_K(3,3);
  bemmy.get_singular_fe_values(numbers::invalid_unsigned_int);
  for (cell = bemmy.dh_stokes.begin_active(); cell != endc; ++cell)
    {
      fe_stokes_v.reinit(cell);
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i=0; i<1 ; ++i)
        {
          local_single_layer = 0.;
          local_double_layer = 0.;
          bool is_singular = false;
          unsigned int singular_index = numbers::invalid_unsigned_int;
          for (unsigned int j=0; j<bemmy.fe_stokes->dofs_per_cell; ++j)
            {
              unsigned int jdim = bemmy.fe_stokes->system_to_component_index(j).first;
              if (jdim == 0 && local_dof_indices[j] == i)
                {
                  is_singular = true;
                  singular_index = j;
                }
              if (is_singular)
                break;
            }
          FEValues<3-1,3> * internal_fe_v;
          if (is_singular == true)
            {
              Assert(singular_index != numbers::invalid_unsigned_int,
                     ExcInternalError());
              internal_fe_v  = & bemmy.get_singular_fe_values(singular_index);// should be correct since they should be 3 equal support_points
              internal_fe_v->reinit(cell);
            }
          else
            {
              internal_fe_v = & fe_stokes_v;
            }

          const std::vector<Point<3> > &q_points = internal_fe_v->get_quadrature_points();
          const std::vector<Tensor<1, 3> > &normals  = internal_fe_v->get_all_normal_vectors();

          unsigned int n_q_points = q_points.size();
          for (unsigned int q=0; q<n_q_points; ++q)
            {
              const Tensor<1,3> R = q_points[q] - support_points[i];
              Tensor<2,3> G = bemmy.stokes_kernel.value_tens(R);
              Tensor<3,3> W = bemmy.stokes_kernel.value_tens2(R);
              Tensor<2, 3> singular_ker = bemmy.compute_singular_kernel(normals[q], W);
              for (unsigned int idim=0; idim < 3; ++idim)
                {
                  Assert(support_points[i].distance(support_points[i*3+idim])<1e-10, ExcInternalError());
                  for (unsigned int j=0; j<bemmy.fe_stokes->dofs_per_cell; ++j)
                    {

                      unsigned int jdim = bemmy.fe_stokes->system_to_component_index(j).first;
                      local_single_layer(idim, j) += ( G[idim][jdim] *
                                                       internal_fe_v->shape_value(j,q)     *
                                                       internal_fe_v->JxW(q)       );
                      local_double_layer(idim, j) -= (singular_ker[idim][jdim] * //(R[jdim] *
                                                      (internal_fe_v->shape_value(j,q)) *
                                                      internal_fe_v->JxW(q));
                    }
                  for (unsigned int jdim=0; jdim<3; ++jdim)
                    {
                      test_V.add(idim, jdim, G[idim][jdim] * //(R[jdim] *
                                 internal_fe_v->JxW(q)       );
                    }
                  for (unsigned int jdim=0; jdim<3; ++jdim)
                    {
                      test_K.add(idim, jdim, singular_ker[idim][jdim] * //(R[jdim] *
                                 internal_fe_v->JxW(q)       );
                    }

                }
            }
          // for(unsigned int jjj = 0; jjj<local_dof_indices.size(); ++jjj)
          //  std::cout<<jjj<<" "<<local_dof_indices[jjj]<<" ";
          // std::cout<<std::endl;

          std::vector<types::global_dof_index> local_dof_indices_row(3);
          for (unsigned int idim=0; idim<3; ++idim)
            {
              local_dof_indices_row[idim]=idim;
            }
          bemmy.constraints.distribute_local_to_global(local_single_layer,local_dof_indices_row,local_dof_indices,V_matrix_3_line);
          bemmy.constraints.distribute_local_to_global(local_double_layer,local_dof_indices_row,local_dof_indices,K_matrix_3_line);
        }
    }

  bemmy.pcout<<"Check on V_matrix"<<std::endl;
  test_V.print_formatted(std::cout);
  bemmy.pcout<<"Check on K_matrix"<<std::endl;
  test_K.print_formatted(std::cout);
  // for(unsigned int i=0; i<dim; ++i)
  //   bemmy.pcout<<"component i = "<<i<<" : Check value = "<<reference_integral_K[i]<<std::endl;
}

int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-8;
  const unsigned int dim = 3;
  BEMProblem<dim> bem_problem_3d_1, bem_problem_3d_2;
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d_1.convert_bool_parameters();
  bem_problem_3d_2.convert_bool_parameters();
  bem_problem_3d_1.pcout<<"Minimum Test for the assemble of the double layer and the solid angle alpha"<<std::endl;
  std::string fe_name_1="FESystem<2,3>[FE_Q<2,3>(2)^3]";
  bem_problem_3d_1.fe_stokes =  std::unique_ptr<FiniteElement<dim-1, dim> >(FETools::get_fe_by_name<dim-1,dim>(fe_name_1));
  bem_problem_3d_1.fe_map =  std::unique_ptr<FiniteElement<dim-1, dim> >(FETools::get_fe_by_name<dim-1,dim>(fe_name_1));
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d_1.input_grid_path=mesh_filename_path;
  bem_problem_3d_1.input_grid_base_name="sphere_coarse_";
  bem_problem_3d_1.input_grid_format="inp";
  bem_problem_3d_1.create_box_bool=false;
  bem_problem_3d_2.create_box_bool=false;
  for (unsigned int i = 0 ; i<bem_problem_3d_1.wall_bool.size(); ++i)
    {
      bem_problem_3d_1.wall_bool[i]=false;
      bem_problem_3d_2.wall_bool[i]=false;
    }
  bem_problem_3d_1.read_domain();
  bem_problem_3d_1.tria.set_manifold(0,*bem_problem_3d_1.manifold);
  bem_problem_3d_1.dh_stokes.distribute_dofs(*bem_problem_3d_1.fe_stokes);
  bem_problem_3d_1.map_dh.distribute_dofs(*bem_problem_3d_1.fe_stokes);
  bem_problem_3d_1.euler_vec.reinit(bem_problem_3d_1.map_dh.n_dofs());
  VectorTools::get_position_vector(bem_problem_3d_1.map_dh,bem_problem_3d_1.euler_vec);
  bem_problem_3d_1.mappingeul = std::make_shared<MappingFEField<2, 3> >(bem_problem_3d_1.map_dh,bem_problem_3d_1.euler_vec);
  FullMatrix<double> V_matrix_3_line_1, K_matrix_3_line_1;
  FullMatrix<double> V_matrix_3_line_3, K_matrix_3_line_3;
  std::vector<Vector<double> > versors_1(3, Vector<double> (bem_problem_3d_1.dh_stokes.n_dofs()));
  std::vector<Vector<double> > check_1(3, Vector<double> (3));
  for (unsigned int idim=0; idim<3; ++idim)
    for (unsigned int i=0; i<bem_problem_3d_1.dh_stokes.n_dofs()/3; ++i)
      versors_1[idim][i*3+idim]=-1.;
  assemble_single_line_stokes_matrices(V_matrix_3_line_1,K_matrix_3_line_1,bem_problem_3d_1);
  std::cout<<"Test on V "<<std::endl;
  for (unsigned int idim=0; idim<3; ++idim)
    {
      V_matrix_3_line_1.vmult(check_1[idim],versors_1[idim]);
    }
  for (unsigned int idim=0; idim<3; ++idim)
    {
      for (unsigned int jdim=0; jdim<3; ++jdim)
        std::cout<<-check_1[jdim][idim]<<" ";
      std::cout<<std::endl;
    }
  std::cout<<"Test on K "<<std::endl;
  for (unsigned int idim=0; idim<3; ++idim)
    {
      K_matrix_3_line_1.vmult(check_1[idim],versors_1[idim]);
    }
  for (unsigned int idim=0; idim<3; ++idim)
    {
      for (unsigned int jdim=0; jdim<3; ++jdim)
        std::cout<<check_1[jdim][idim]<<" ";
      std::cout<<std::endl;
    }
  std::cout<<bem_problem_3d_1.map_dh.n_dofs()<<std::endl;
  std::string fe_name_2 = "FESystem<2,3>[FE_Q<2,3>(2)^3]";
  bem_problem_3d_2.fe_stokes = std::unique_ptr<FiniteElement<dim-1, dim> >(FETools::get_fe_by_name<dim-1,dim>(fe_name_2));
  bem_problem_3d_2.fe_map =  std::unique_ptr<FiniteElement<dim-1, dim> >(FETools::get_fe_by_name<dim-1,dim>(fe_name_2));
  bem_problem_3d_2.input_grid_path=mesh_filename_path;
  bem_problem_3d_2.input_grid_base_name="sphere_coarse_";
  bem_problem_3d_2.input_grid_format="inp";
  bem_problem_3d_2.read_domain();
  bem_problem_3d_2.tria.set_manifold(0,*bem_problem_3d_2.manifold);
  bem_problem_3d_2.dh_stokes.distribute_dofs(*bem_problem_3d_2.fe_stokes);
  bem_problem_3d_2.map_dh.distribute_dofs(*bem_problem_3d_2.fe_stokes);
  std::vector<types::global_dof_index >   new_numbers(bem_problem_3d_2.dh_stokes.n_dofs());
  FullMatrix<double> permutation_matrix(bem_problem_3d_2.dh_stokes.n_dofs(), bem_problem_3d_2.dh_stokes.n_dofs());
  for (unsigned int i = 0; i < new_numbers.size()/3; ++i)
    {
      if (i == 2)
        {
          for (unsigned int idim=0; idim<3; ++idim)
            {
              new_numbers[3*2+idim]=3*22+idim;
              permutation_matrix.set(3*2+idim,3*22+idim,1.);
            }
        }
      else if (i == 22)
        {
          for (unsigned int idim=0; idim<3; ++idim)
            {
              new_numbers[3*22+idim]=3*2+idim;
              permutation_matrix.set(3*22+idim,3*2+idim,1.);
            }
        }
      else
        {
          for (unsigned int idim=0; idim<3; ++idim)
            {
              new_numbers[3*i+idim]=3*i+idim;
              permutation_matrix.set(3*i+idim,3*i+idim,1.);
            }
        }
    }

  bem_problem_3d_2.dh_stokes.renumber_dofs(new_numbers);
  bem_problem_3d_2.map_dh.renumber_dofs(new_numbers);
  bem_problem_3d_2.euler_vec.reinit(bem_problem_3d_2.map_dh.n_dofs());
  VectorTools::get_position_vector(bem_problem_3d_2.map_dh,bem_problem_3d_2.euler_vec);
  bem_problem_3d_2.mappingeul = std::make_shared<MappingFEField<2, 3> >(bem_problem_3d_2.map_dh,bem_problem_3d_2.euler_vec);
  FullMatrix<double> V_matrix_3_line_2, K_matrix_3_line_2;
  std::vector<Vector<double> > versors_2(3, Vector<double> (bem_problem_3d_2.dh_stokes.n_dofs()));
  std::vector<Vector<double> > check_2(3, Vector<double> (3));
  for (unsigned int idim=0; idim<3; ++idim)
    for (unsigned int i=0; i<bem_problem_3d_2.dh_stokes.n_dofs()/3; ++i)
      versors_2[idim][i*3+idim]=-1.;
  assemble_single_line_stokes_matrices(V_matrix_3_line_2,K_matrix_3_line_2,bem_problem_3d_2);

  V_matrix_3_line_3.reinit(3,bem_problem_3d_2.dh_stokes.n_dofs());
  K_matrix_3_line_3.reinit(3,bem_problem_3d_2.dh_stokes.n_dofs());
  V_matrix_3_line_1.mmult(V_matrix_3_line_3,permutation_matrix);
  K_matrix_3_line_1.mmult(K_matrix_3_line_3,permutation_matrix);

  // for(unsigned int i =0 ; i<3; ++i)
  //   for(unsigned int j =0 ; j<versors_2[0].size(); ++j)
  //   {
  //     if(std::fabs(V_matrix_3_line_1[i,j]-V_matrix_3_line_2[i,j])>tol)
  //     {
  //       std::cout<<"Error on V ";
  //     }
  //     else
  //     {
  //       std::cout<<"OK ";
  //
  //     }
  //   }
  // std::cout<<std::endl;
  // for(unsigned int i =0 ; i<3; ++i)
  //   for(unsigned int j =0 ; j<versors_2[0].size(); ++j)
  //   {
  //     if(std::fabs(K_matrix_3_line_1[i,j]-K_matrix_3_line_2[i,j])>tol)
  //     {
  //       std::cout<<"Error on K ";
  //     }
  //     else
  //     {
  //       std::cout<<"OK ";
  //
  //     }
  //   }
  if (V_matrix_3_line_3 == V_matrix_3_line_2)
    std::cout<<"OK permutation on V matrix"<<std::endl;
  else
    std::cout<<"ERROR permutation on V matrix"<<std::endl;

  if (K_matrix_3_line_3 == K_matrix_3_line_2)
    std::cout<<"OK permutation on K matrix"<<std::endl;
  else
    std::cout<<"ERROR permutation on K matrix"<<std::endl;
  std::ofstream ofs_K_1, ofs_K_2;
  std::string filename_K_1, filename_K_2;
  filename_K_1="K_matrix_1.txt";
  filename_K_2="K_matrix_2.txt";
  ofs_K_1.open (filename_K_1, std::ofstream::out);
  ofs_K_2.open (filename_K_2, std::ofstream::out);

  for (unsigned int i =0 ; i<3; ++i)
    {
      for (unsigned int j =0 ; j<versors_2[0].size(); ++j)
        {
          ofs_K_1<<K_matrix_3_line_1(i,j)<<" ";
          ofs_K_2<<K_matrix_3_line_2(i,j)<<" ";
        }
      ofs_K_1 << std::endl;
      ofs_K_2 << std::endl;
    }
  ofs_K_1.close();
  ofs_K_2.close();
  std::cout<<"Test on V "<<std::endl;
  for (unsigned int idim=0; idim<3; ++idim)
    {
      V_matrix_3_line_2.vmult(check_2[idim],versors_2[idim]);
    }
  for (unsigned int idim=0; idim<3; ++idim)
    {
      for (unsigned int jdim=0; jdim<3; ++jdim)
        std::cout<<-check_2[jdim][idim]<<" ";
      std::cout<<std::endl;
    }
  std::cout<<"Test on K "<<std::endl;
  for (unsigned int idim=0; idim<3; ++idim)
    {
      K_matrix_3_line_2.vmult(check_2[idim],versors_2[idim]);
    }
  for (unsigned int idim=0; idim<3; ++idim)
    {
      for (unsigned int jdim=0; jdim<3; ++jdim)
        std::cout<<check_2[jdim][idim]<<" ";
      std::cout<<std::endl;
    }


  bem_problem_3d_1.tria.set_manifold(0);
  bem_problem_3d_2.tria.set_manifold(0);

  return 0;
}
