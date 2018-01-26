#include "bem_stokes.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>

int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  const unsigned int degree = 1;
  const unsigned int mapping_degree = 1;
  const unsigned int dim = 2;
  double tol=2e-2;


  BEMProblem<dim> bem_problem_2d;

  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d.prm", "used_foo.prm");
  bem_problem_2d.convert_bool_parameters();
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_2d.create_box_bool=false;
  bem_problem_2d.wall_bool_0=false;
  bem_problem_2d.wall_bool_1=false;
  bem_problem_2d.wall_bool_2=false;
  bem_problem_2d.wall_bool_3=false;
  bem_problem_2d.wall_bool_4=false;
  bem_problem_2d.wall_bool_5=false;
  bem_problem_2d.reflect_kernel=false;
  bem_problem_2d.no_slip_kernel=false;
  bem_problem_2d.use_internal_alpha=false;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_2d.input_grid_path=mesh_filename_path;
  bem_problem_2d.input_grid_base_name="circle_";
  bem_problem_2d.input_grid_format="inp";
  bem_problem_2d.fe_stokes = SP(bem_problem_2d.parsed_fe_stokes());
  bem_problem_2d.fe_map = SP(bem_problem_2d.parsed_fe_mapping());
  bem_problem_2d.read_domain();
  SphericalManifold<dim-1,dim> manifold;
  bem_problem_2d.tria.set_all_manifold_ids(0);
  bem_problem_2d.tria.set_manifold(0, manifold);
  // bem_problem_2d.tria.refine_global(2);
  bem_problem_2d.reinit();
  TrilinosWrappers::MPI::Vector V_x_normals(bem_problem_2d.normal_vector.locally_owned_elements(),bem_problem_2d.normal_vector.get_mpi_communicator());
  TrilinosWrappers::MPI::Vector foo(bem_problem_2d.normal_vector.locally_owned_elements(),bem_problem_2d.normal_vector.get_mpi_communicator());
  bem_problem_2d.compute_euler_vector(bem_problem_2d.euler_vec,0);
  bem_problem_2d.mappingeul = SP(new MappingFEField<dim-1,dim>(bem_problem_2d.map_dh, bem_problem_2d.euler_vec));
  bem_problem_2d.compute_euler_vector(bem_problem_2d.next_euler_vec,1);
  bem_problem_2d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_2d.compute_normal_vector();

  std::cout<<"Test on the assembling of the Single Layer and the tangential projector"<<std::endl;
  bem_problem_2d.assemble_stokes_system(true);
  bem_problem_2d.V_matrix.vmult(V_x_normals, bem_problem_2d.normal_vector);
  // V_matrix.vmult(V_x_normals_body, normal_vector_pure);
  //
  // // foo.sadd(0.,1., V_x_normals,-1., normal_vector) ;
  // pcout << "Check on the V operator Norm post (should be one): " << V_x_normals_body.linfty_norm() << std::endl;
  // foo.reinit(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator);
  // tangential_projector_body(normal_vector_pure, foo);


  if (bem_problem_2d.body_cpu_set == bem_problem_2d.this_cpu_set && bem_problem_2d.this_cpu_set == complete_index_set(bem_problem_2d.this_cpu_set.size()))
    {
      std::cout << "OK Index Sets for the tests" << std::endl;
    }
  else
    {
      std::cout<<"Error ON index sets"<<std::endl;
    }
  if (std::fabs(foo.linfty_norm()) < tol)
    std::cout<<"OK test on tangential projector"<<std::endl;
  else
    std::cout << "Check on the tangential projection (should be zero): " << foo.linfty_norm() << std::endl;
  foo.sadd(0.,1., V_x_normals);
  foo.sadd(1.,-1., bem_problem_2d.normal_vector);
  if (std::fabs((V_x_normals.linfty_norm() - 1)) < tol && std::fabs(foo.linfty_norm()) < tol)
    std::cout<<"OK test on Single Layer kernel post correction"<<std::endl;
  else
    {
      std::cout << "Check on the V operator Norm post (should be one): " << V_x_normals.linfty_norm() << std::endl;
      std::cout << "Check on the V_x_norm post (should be zero): " << foo.linfty_norm() << std::endl;
    }
  std::ofstream out ("grid-3.vtk");
  GridOut grid_out;
  grid_out.write_vtk (bem_problem_2d.tria, out);
  std::cout << "Grid written to grid-3.vtk" << std::endl;
  bem_problem_2d.tria.set_manifold(0);
  return 0;
}
