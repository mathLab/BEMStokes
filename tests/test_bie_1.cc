#include "bem_stokes.h"



int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  const unsigned int degree = 1;
  const unsigned int mapping_degree = 1;
  std::vector<Point<2> > val_points(2*3);
  double tol_ext=1e-5;
  double tol_int=1e-3;

  //interior point
  val_points[0][0] = 0.1;
  val_points[0][1] = 0.1;

  val_points[1][0] = 0.1;
  val_points[1][1] = 0.1;

  val_points[2][0] = 0.1;
  val_points[2][1] = 0.1;
  //exterior point
  val_points[3][0] = 4.;
  val_points[3][1] = 4.;

  val_points[4][0] = 4.;
  val_points[4][1] = 4.;

  val_points[5][0] = 4.;
  val_points[5][1] = 4.;

  //boundary point
  // val_points[2][0] = 1.;
  // val_points[2][1] = 0.;
  // val_points[2][2] = 0.;

  Vector<double> eval_vel(2*3);
  BEMProblem<2> bem_problem_2d;

  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d.prm", "used_foo.prm");
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_2d.convert_bool_parameters();
  // bem_problem_2d.use_internal_alpha=false;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_2d.input_grid_path=mesh_filename_path;
  bem_problem_2d.input_grid_base_name="circle_";
  bem_problem_2d.input_grid_format="inp";
  bem_problem_2d.fe_stokes = SP(bem_problem_2d.parsed_fe_stokes());
  bem_problem_2d.fe_map = SP(bem_problem_2d.parsed_fe_mapping());
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
  dummy_vel=0.;
  bem_problem_2d.evaluate_stokes_bie(val_points, dummy_vel, dummy_force, eval_vel);

  std::cout<<"Test on SingleLayer in 2D"<<std::endl;
  std::cout<<"Interior Point at [0.1,0.1,0.1], expected [0,0,0]"<<std::endl;
  if (std::pow((eval_vel(2*0+0)-0.)*(eval_vel(2*0+0)-0.)+
               (eval_vel(2*1+0)-0.)*(eval_vel(2*1+0)-0.),0.5) < tol_int )
    std::cout<<"OK interior point"<<std::endl;
  else
    {
      std::cout<<"ux = "<<eval_vel(2*0+0)<<std::endl;
      std::cout<<"uy = "<<eval_vel(2*1+0)<<std::endl;
    }
  std::cout<<"Exterior Point at [4,4,4], expected [0,0,0]"<<std::endl;
  if (std::pow((eval_vel(2*0+1)-0.)*(eval_vel(2*0+1)-0.)+
               (eval_vel(2*1+1)-0.)*(eval_vel(2*1+1)-0.),0.5) < tol_int )
    std::cout<<"OK exterior point"<<std::endl;
  else
    {
      std::cout<<"ux = "<<eval_vel(2*0+1)<<std::endl;
      std::cout<<"uy = "<<eval_vel(2*1+1)<<std::endl;
    }

  bem_problem_2d.tria.set_manifold(0);
  return 0;
}
