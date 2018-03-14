#include <bem_stokes.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_generator.h>
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



int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace dealii;
  using namespace BEMStokes;

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-10;
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");
  unsigned int degree = 1;
  BEMProblem<3> bem_problem_3d;

  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
    bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.reflect_kernel=false;
  bem_problem_3d.no_slip_kernel=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();


  bem_problem_3d.grid_type="Real";
  bem_problem_3d.force_pole="Origin";
  bem_problem_3d.monolithic_bool=true;
  bem_problem_3d.solve_directly=false;
  bem_problem_3d.reassemble_preconditoner=true;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="torus_";
  bem_problem_3d.input_grid_format="inp";
  bem_problem_3d.create_box_bool=false;
  // std::cout<< bem_problem_3d.input_grid_path+bem_problem_3d.input_grid_base_name+Utilities::int_to_string(0)+bem_problem_3d.input_grid_format<<std::endl;
  double R,r;

  // R=12.436;
  // r=3.45567;
  // Triangulation<3> foo_tria;
  // Triangulation<2,3> foo_tria_surf;
  // GridGenerator::torus (foo_tria,R,r);
  // TorusManifold<3> manny(R,r);
  // foo_tria.set_all_manifold_ids(0);
  // foo_tria.set_manifold(0,manny);
  // foo_tria.refine_global(2);
  // GridGenerator::extract_boundary_mesh ( foo_tria, foo_tria_surf);
  // std::string filename = "torus.inp";
  //
  //   std::ofstream wall_ofs;
  //   wall_ofs.open(filename, std::ofstream::out);
  //   GridOut go;
  //   go.write_ucd(foo_tria_surf,wall_ofs);
  //
  // bem_problem_3d.tria.refine_global(2);
  // bem_problem_3d.read_domain();
  bem_problem_3d.extra_debug_info=false;
  bem_problem_3d.read_input_mesh_file(0,bem_problem_3d.tria);
  bem_problem_3d.reinit();
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0, true);
  bem_problem_3d.create_wall_body_index_sets();
  bem_problem_3d.mappingeul = SP(new MappingFEField<2, 3> (bem_problem_3d.map_dh,bem_problem_3d.euler_vec));
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);

  Point<3> correct_cg;
  for (unsigned int i=0; i<3; ++i)
    {
      double translation=(i+1)*567.2148;
      correct_cg[i]=translation;
      for (unsigned int j=i*bem_problem_3d.dh_stokes.n_dofs()/3; j<(i+1)*bem_problem_3d.dh_stokes.n_dofs()/3; ++j)
        bem_problem_3d.euler_vec[i]+=translation;
    }
  for (unsigned int i=0; i<3; ++i)
    if (correct_cg.distance(bem_problem_3d.center_of_mass)>tol)
      std::cout<<"OK"<<std::endl;
    else
      std::cout<<"Error "<<bem_problem_3d.center_of_mass<<" instead of "<<correct_cg<<std::endl;




  return 0;
}
