#include "bem_stokes.h"



int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  const unsigned int degree = 1;
  const unsigned int mapping_degree = 1;
  std::vector<Point<3> > val_points(2*3);
  double tol_ext=1e-5;
  double tol_int=1e-3;

  //interior point
  val_points[0][0] = 0.1;
  val_points[0][1] = 0.1;
  val_points[0][2] = 0.1;

  val_points[1][0] = 0.1;
  val_points[1][1] = 0.1;
  val_points[1][2] = 0.1;

  val_points[2][0] = 0.1;
  val_points[2][1] = 0.1;
  val_points[2][2] = 0.1;
  //exterior point
  val_points[3][0] = 4.;
  val_points[3][1] = 4.;
  val_points[3][2] = 4.;

  val_points[4][0] = 4.;
  val_points[4][1] = 4.;
  val_points[4][2] = 4.;

  val_points[5][0] = 4.;
  val_points[5][1] = 4.;
  val_points[5][2] = 4.;

  //boundary point
  // val_points[2][0] = 1.;
  // val_points[2][1] = 0.;
  // val_points[2][2] = 0.;

  Vector<double> eval_vel(2*3);
  BEMProblem<3> bem_problem_3d;

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm", "used_foo.prm");
  // std::cout<<input_grid_base_name<<std::endl;
  bem_problem_3d.create_box_bool=false;
  bem_problem_3d.wall_bool_0=false;
  bem_problem_3d.wall_bool_1=false;
  bem_problem_3d.wall_bool_2=false;
  bem_problem_3d.wall_bool_3=false;
  bem_problem_3d.wall_bool_4=false;
  bem_problem_3d.wall_bool_5=false;
  bem_problem_3d.use_internal_alpha=false;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="sphere_half_refined_";
  bem_problem_3d.input_grid_format="inp";
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();
  bem_problem_3d.read_domain();
  bem_problem_3d.reinit();
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0);
  bem_problem_3d.mappingeul = SP(new MappingFEField<2,3>(bem_problem_3d.map_dh, bem_problem_3d.euler_vec));
  bem_problem_3d.compute_euler_vector(bem_problem_3d.next_euler_vec,1);

  for (auto i : bem_problem_3d.shape_velocities.locally_owned_elements())
    bem_problem_3d.shape_velocities[i] = 1./bem_problem_3d.time_step * (bem_problem_3d.next_euler_vec[i]-bem_problem_3d.euler_vec[i]);
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_normal_vector();

  // bem_problem_2d.compute_normal_vector();
  Vector<double> dummy_vel(bem_problem_3d.dh_stokes.n_dofs());
  Vector<double> dummy_force(bem_problem_3d.dh_stokes.n_dofs());
  dummy_force=bem_problem_3d.normal_vector;
  dummy_vel=0.;
  bem_problem_3d.evaluate_stokes_bie(val_points, dummy_vel, dummy_force, eval_vel);

  std::cout<<"Test on SingleLayer in 3D"<<std::endl;
  std::cout<<"Interior Point at [0.1,0.1,0.1], expected [0,0,0]"<<std::endl;
  if (std::pow((eval_vel(2*0+0)-0.)*(eval_vel(2*0+0)-0.)+
               (eval_vel(2*1+0)-0.)*(eval_vel(2*1+0)-0.)+
               (eval_vel(2*2+0)-0.)*(eval_vel(2*2+0)-0.),0.5) < tol_int )
    std::cout<<"OK interior point"<<std::endl;
  else
    {
      std::cout<<"ux = "<<eval_vel(2*0+0)<<std::endl;
      std::cout<<"uy = "<<eval_vel(2*1+0)<<std::endl;
      std::cout<<"uz = "<<eval_vel(2*2+0)<<std::endl;
    }
  std::cout<<"Exterior Point at [4,4,4], expected [0,0,0]"<<std::endl;
  if (std::pow((eval_vel(2*0+1)-0.)*(eval_vel(2*0+1)-0.)+
               (eval_vel(2*1+1)-0.)*(eval_vel(2*1+1)-0.)+
               (eval_vel(2*2+1)-0.)*(eval_vel(2*2+1)-0.),0.5) < tol_int )
    std::cout<<"OK exterior point"<<std::endl;
  else
    {
      std::cout<<"ux = "<<eval_vel(2*0+1)<<std::endl;
      std::cout<<"uy = "<<eval_vel(2*1+1)<<std::endl;
      std::cout<<"uz = "<<eval_vel(2*2+1)<<std::endl;
    }

  // std::cout<<"Interior Point at [0.1,0.1,0.1], expected [0,0,0]"<<std::endl;
  // std::cout<<"ux = "<<eval_vel(3*0+0)<<std::endl;
  // std::cout<<"uy = "<<eval_vel(3*0+1)<<std::endl;
  // std::cout<<"uz = "<<eval_vel(3*0+2)<<std::endl;
  // std::cout<<"Exterior Point at [4,4,4], expected [0,0,0]"<<std::endl;
  // std::cout<<"ux = "<<eval_vel(3*1+0)<<std::endl;
  // std::cout<<"uy = "<<eval_vel(3*1+1)<<std::endl;
  // std::cout<<"uz = "<<eval_vel(3*1+2)<<std::endl;
  // std::cout<<"Boundary Point at [1,0,0], expected [0,0,0]"<<std::endl;
  // std::cout<<"ux = "<<eval_vel(3*2+0)<<std::endl;
  // std::cout<<"uy = "<<eval_vel(3*2+1)<<std::endl;
  // std::cout<<"uz = "<<eval_vel(3*2+2)<<std::endl;
  bem_problem_3d.tria.reset_manifold(0);
  return 0;
}
