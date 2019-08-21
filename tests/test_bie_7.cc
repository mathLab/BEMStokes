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

  std::vector<Point<dim> > val_points(1);
  double tol_ext=1e-5;
  double tol_int=7e-2;

  Vector<double> eval_vel(dim);
  BEMProblem<dim> bem_problem_2d;

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d.prm", "used_foo.prm");
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_2d.convert_bool_parameters();
  bem_problem_2d.create_box_bool=false;
  bem_problem_2d.wall_bool_0=false;
  bem_problem_2d.wall_bool_1=false;
  bem_problem_2d.wall_bool_2=false;
  bem_problem_2d.wall_bool_3=false;
  bem_problem_2d.wall_bool_4=false;
  bem_problem_2d.wall_bool_5=false;
  bem_problem_2d.reflect_kernel=false;
  bem_problem_2d.no_slip_kernel=false;
  bem_problem_2d.use_internal_alpha=true;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_2d.input_grid_path=mesh_filename_path;
  bem_problem_2d.input_grid_base_name="circle_";
  bem_problem_2d.input_grid_format="inp";
  bem_problem_2d.fe_stokes = bem_problem_2d.parsed_fe_stokes();
  bem_problem_2d.fe_map = bem_problem_2d.parsed_fe_mapping();
  bem_problem_2d.read_domain();

  // bem_problem_2d.reinit();
  SphericalManifold<dim-1,dim> manifold;
  bem_problem_2d.tria.set_manifold(0, manifold);
  bem_problem_2d.tria.refine_global(2);
  bem_problem_2d.build_sphere_in_deal=true;
  bem_problem_2d.reinit();
  bem_problem_2d.compute_euler_vector(bem_problem_2d.euler_vec,0);
  bem_problem_2d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_2d.mappingeul = SP(new MappingFEField<dim-1,dim>(bem_problem_2d.map_dh, bem_problem_2d.euler_vec));
  bem_problem_2d.compute_euler_vector(bem_problem_2d.next_euler_vec,1);
  Vector<double> dummy_vel(bem_problem_2d.dh_stokes.n_dofs());
  Vector<double> dummy_force(bem_problem_2d.dh_stokes.n_dofs());
  std::vector<Point<dim> > support_points(bem_problem_2d.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<dim-1, dim>( *bem_problem_2d.mappingeul, bem_problem_2d.dh_stokes, support_points);

  std::cout<<"Test on the boundary in 2D for the Double Layer"<<std::endl;

  val_points[0] = support_points[0];

  for (unsigned int i = 0; i<bem_problem_2d.num_rigid; ++i)
    {
      eval_vel.reinit(1*dim);
      dummy_force=0.;//bem_problem_2d.normal_vector;
      dummy_vel=bem_problem_2d.N_rigid[i];

      bem_problem_2d.evaluate_stokes_bie_on_boundary(val_points, dummy_vel, dummy_force, eval_vel);
      std::cout<<"Test on the "<<i<<"th rigid mode"<<std::endl;
      std::cout<<"The rigid mode is = " << dummy_vel(0)<<" "<<dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim)<<std::endl;
      std::cout<<"Boundary Point at "<< val_points[0]<<" , expected ["<< 0.5*dummy_vel(0)<<", "<< 0.5*dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim*1)<<" ]"<<std::endl;
      double foo = 0;
      for (unsigned int k=0; k<dim; ++k)
        foo += std::pow(eval_vel(k)-0.5*dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim*k),2);

      if (std::pow(foo,0.5) < tol_int )
        std::cout<<"OK boundary point"<<std::endl;
      else
        {
          std::cout<<"ux = "<<eval_vel(0)<<std::endl;
          std::cout<<"uy = "<<eval_vel(1)<<" "<<std::pow(foo,0.5)<<std::endl;
        }

    }
  std::ofstream out ("grid-dim.vtk");
  GridOut grid_out;
  grid_out.write_vtk (bem_problem_2d.tria, out);
  std::cout << "Grid written to grid-dim.vtk" << std::endl;
  bem_problem_2d.tria.set_manifold(0);
  return 0;
}
