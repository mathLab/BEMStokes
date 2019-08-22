#include "bem_stokes.h"



int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  const unsigned int degree = 1;
  const unsigned int mapping_degree = 1;
  const unsigned int dim = 2;

  std::vector<Point<2> > val_points(2);
  double tol_ext=1e-5;
  double tol_int=6e-2;

  val_points[0][0] = 0.1;
  val_points[0][1] = 0.1;

  val_points[1][0] = 4.;
  val_points[1][1] = 4.;


  Vector<double> eval_vel(2*dim);
  BEMProblem<dim> bem_problem_2d;

  // bem_problem_2d.read_parameters(SOURCE_DIR "/parameters_test_2d.prm");
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d.prm", "used_foo.prm");
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_2d.convert_bool_parameters();
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
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_2d.read_domain();
  bem_problem_2d.reinit();
  bem_problem_2d.compute_euler_vector(bem_problem_2d.euler_vec,0);
  bem_problem_2d.mappingeul = SP(new MappingFEField<1,2>(bem_problem_2d.map_dh, bem_problem_2d.euler_vec));
  bem_problem_2d.compute_euler_vector(bem_problem_2d.next_euler_vec,1);

  for (auto i : bem_problem_2d.shape_velocities.locally_owned_elements())
    bem_problem_2d.shape_velocities[i] = 1./bem_problem_2d.time_step * (bem_problem_2d.next_euler_vec[i]-bem_problem_2d.euler_vec[i]);
  bem_problem_2d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_2d.compute_normal_vector();

  // bem_problem_2d.compute_normal_vector();
  Vector<double> dummy_vel(bem_problem_2d.dh_stokes.n_dofs());
  Vector<double> dummy_force(bem_problem_2d.dh_stokes.n_dofs());
  dummy_force=bem_problem_2d.normal_vector;
  std::cout<<"Test on DoubleLayer in 2D"<<std::endl;

  for (unsigned int i = 0; i<bem_problem_2d.num_rigid; ++i) //bem_problem_2d.N_rigid.size(); ++i)
    {
      eval_vel.reinit(2*dim);
      dummy_force=0.;//bem_problem_2d.normal_vector;
      dummy_vel = bem_problem_2d.N_rigid[i];
      // dummy_vel.print(std::cout);
      bem_problem_2d.evaluate_stokes_bie(val_points, dummy_vel, dummy_force, eval_vel);
      std::cout<<"Test on the "<<i<<"th rigid mode"<<std::endl;
      std::cout<<"The rigid mode is = " << dummy_vel(0)<<" "<<dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim)<<std::endl;
      std::cout<<"Interior Point at [0.1, 0.1], expected ["<<dummy_vel(0)<<", "<<dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim) <<"]"<<std::endl;
      // std::cout<<"Interior Point at [0.1,0.1], expected [0,0]"<<std::endl;
      // eval_vel.print(std::cout);
      double foo=0.;
      for (unsigned int k=0; k<dim; ++k)
        foo += std::pow(eval_vel(2*k+0)-dummy_vel(bem_problem_2d.dh_stokes.n_dofs()/dim*k+0),2);
      if (std::pow(foo,0.5) < tol_int )
        std::cout<<"OK interior point"<<std::endl;
      else
        {
          std::cout<<"ux = "<<eval_vel(2*0+0)<<std::endl;
          std::cout<<"uy = "<<eval_vel(2*1+0)<<" "<<foo<<std::endl;
        }
      std::cout<<"Exterior Point at [4, 4], expected [0, 0]"<<std::endl;
      foo = 0.;
      for (unsigned int k=0; k<dim; ++k)
        foo += std::pow(eval_vel(2*k+1),2);

      if (std::pow(foo,0.5) < tol_ext )
        std::cout<<"OK exterior point"<<std::endl;
      else
        {
          std::cout<<"ux = "<<eval_vel(2*0+1)<<std::endl;
          std::cout<<"uy = "<<eval_vel(2*1+1)<<" "<<foo<<std::endl;
        }

    }
  bem_problem_2d.tria.reset_manifold(0);
  return 0;
}
