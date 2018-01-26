#include "bem_stokes.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_tools.h>

#include <iostream>
#include <fstream>
#include <deal2lkit/error_handler.h>
#include <deal2lkit/parsed_function.h>
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


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-8;
  const unsigned int dim = 3;
  BEMProblem<dim> bem_problem_3d_1, bem_problem_3d_2;
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d_1.convert_bool_parameters();
  bem_problem_3d_2.convert_bool_parameters();
  bem_problem_3d_1.pcout<<"Minimum Test for the assemble of the double layer and the solid angle alpha"<<std::endl;
  std::string fe_name_1="FESystem<2,3>[FE_Q<2,3>(1)^3]";
  bem_problem_3d_1.fe_stokes =  SP(FETools::get_fe_by_name<dim-1,dim>(fe_name_1));
  bem_problem_3d_1.fe_map =  SP(FETools::get_fe_by_name<dim-1,dim>(fe_name_1));
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d_1.input_grid_path=mesh_filename_path;
  bem_problem_3d_1.input_grid_base_name="sphere_half_refined_";
  bem_problem_3d_1.input_grid_format="inp";
  bem_problem_3d_1.create_box_bool=false;
  bem_problem_3d_2.create_box_bool=false;
  for (unsigned int i = 0 ; i<bem_problem_3d_1.wall_bool.size(); ++i)
    {
      bem_problem_3d_1.wall_bool[i]=false;
      bem_problem_3d_2.wall_bool[i]=false;
    }

  bem_problem_3d_1.read_domain();
  bem_problem_3d_1.dh_stokes.distribute_dofs(*bem_problem_3d_1.fe_stokes);
  bem_problem_3d_1.map_dh.distribute_dofs(*bem_problem_3d_1.fe_stokes);
  bem_problem_3d_1.euler_vec.reinit(bem_problem_3d_1.map_dh.n_dofs());
  VectorTools::get_position_vector(bem_problem_3d_1.map_dh,bem_problem_3d_1.euler_vec);
  bem_problem_3d_1.mappingeul = SP(new MappingFEField<2, 3>(bem_problem_3d_1.map_dh,bem_problem_3d_1.euler_vec));

  std::map<unsigned int, Point<dim> > my_map;
  DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem_problem_3d_1.mappingeul, bem_problem_3d_1.dh_stokes, my_map);
  std::string filename_1("no_renumbering.gpl");
  std::ofstream output_1(filename_1.c_str());
  DoFTools::write_gnuplot_dof_support_point_info (output_1, my_map);
  std::string fe_name_2 = "FESystem<2,3>[FE_Q<2,3>(1)^3]";
  bem_problem_3d_2.fe_stokes = SP(FETools::get_fe_by_name<dim-1,dim>(fe_name_2));
  bem_problem_3d_2.fe_map =  SP(FETools::get_fe_by_name<dim-1,dim>(fe_name_2));
  bem_problem_3d_2.input_grid_path=mesh_filename_path;
  bem_problem_3d_2.input_grid_base_name="sphere_half_refined_";
  bem_problem_3d_2.input_grid_format="inp";
  bem_problem_3d_2.read_domain();
  bem_problem_3d_2.dh_stokes.distribute_dofs(*bem_problem_3d_2.fe_stokes);
  bem_problem_3d_2.map_dh.distribute_dofs(*bem_problem_3d_2.fe_stokes);
  DoFRenumbering::component_wise(bem_problem_3d_2.dh_stokes);
  DoFRenumbering::component_wise(bem_problem_3d_2.map_dh);
  bem_problem_3d_2.euler_vec.reinit(bem_problem_3d_2.map_dh.n_dofs());
  VectorTools::get_position_vector(bem_problem_3d_2.map_dh,bem_problem_3d_2.euler_vec);
  bem_problem_3d_2.mappingeul = SP(new MappingFEField<2, 3>(bem_problem_3d_2.map_dh,bem_problem_3d_2.euler_vec));
  std::map<unsigned int, Point<dim> > my_map_2;

  DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem_problem_3d_2.mappingeul, bem_problem_3d_2.dh_stokes, my_map_2);
  std::string filename_2("yes_renumbering.gpl");
  std::ofstream output_2(filename_2.c_str());
  DoFTools::write_gnuplot_dof_support_point_info (output_2, my_map_2);


  bem_problem_3d_1.tria.set_manifold(0);
  bem_problem_3d_2.tria.set_manifold(0);

  return 0;
}
