#include <bem_stokes.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
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
  double tol=1e-6;
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  std::cout<<"Test for a bacteria to test that a translation does not alter its rigid velocities"<<std::endl;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");
  unsigned int degree = 1;
  BEMProblem<3> bem_problem_3d;
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_ref_quadrature.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  // for(unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
  //   bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.reflect_kernel=false;
  bem_problem_3d.no_slip_kernel=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();


  bem_problem_3d.use_internal_alpha=false;
  bem_problem_3d.create_box_bool=false;



  bem_problem_3d.wall_bool[0]=false;
  bem_problem_3d.wall_bool[1]=false;
  bem_problem_3d.wall_bool[2]=false;
  bem_problem_3d.wall_bool[3]=false;
  bem_problem_3d.wall_bool[4]=false;
  bem_problem_3d.wall_bool[5]=false;

  bem_problem_3d.wall_spans[0][0]=100;
  bem_problem_3d.wall_spans[0][1]=0;
  bem_problem_3d.wall_spans[0][2]=100;
  bem_problem_3d.wall_spans[1][0]=100;
  bem_problem_3d.wall_spans[1][1]=0;
  bem_problem_3d.wall_spans[1][2]=100;
  bem_problem_3d.wall_spans[2][0]=0;
  bem_problem_3d.wall_spans[2][1]=100;
  bem_problem_3d.wall_spans[2][2]=100;
  bem_problem_3d.wall_spans[3][0]=0;
  bem_problem_3d.wall_spans[3][1]=100;
  bem_problem_3d.wall_spans[3][2]=100;
  bem_problem_3d.wall_spans[4][0]=100;
  bem_problem_3d.wall_spans[4][1]=100;
  bem_problem_3d.wall_spans[4][2]=0;
  bem_problem_3d.wall_spans[5][0]=100;
  bem_problem_3d.wall_spans[5][1]=100;
  bem_problem_3d.wall_spans[5][2]=0;

  bem_problem_3d.wall_positions[0][0]=0.;
  bem_problem_3d.wall_positions[0][1]=100.;
  bem_problem_3d.wall_positions[0][2]=0.;
  bem_problem_3d.wall_positions[1][0]=0.;
  bem_problem_3d.wall_positions[1][1]=-100.;
  bem_problem_3d.wall_positions[1][2]=0.;
  bem_problem_3d.wall_positions[2][0]=100.;
  bem_problem_3d.wall_positions[2][1]=0.;
  bem_problem_3d.wall_positions[2][2]=0.;
  bem_problem_3d.wall_positions[3][0]=-100.;
  bem_problem_3d.wall_positions[3][1]=0.;
  bem_problem_3d.wall_positions[3][2]=0.;
  bem_problem_3d.wall_positions[4][0]=0.;
  bem_problem_3d.wall_positions[4][1]=0.;
  bem_problem_3d.wall_positions[4][2]=100.;
  bem_problem_3d.wall_positions[5][0]=0.;
  bem_problem_3d.wall_positions[5][1]=0.;
  bem_problem_3d.wall_positions[5][2]=-100.;

  bem_problem_3d.wall_types[0]="dirichlet";
  bem_problem_3d.wall_types[1]="dirichlet";
  bem_problem_3d.wall_types[2]="dirichlet";
  bem_problem_3d.wall_types[3]="dirichlet";
  bem_problem_3d.wall_types[4]="dirichlet";
  bem_problem_3d.wall_types[5]="dirichlet";

  bem_problem_3d.grid_type="Real";
  bem_problem_3d.force_pole="Origin";
  bem_problem_3d.extra_debug_info=false;
  bem_problem_3d.monolithic_bool=true;
  bem_problem_3d.solve_directly=false;
  bem_problem_3d.reassemble_preconditoner=true;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="bacterium_";
  bem_problem_3d.input_grid_format="msh";

  bem_problem_3d.read_domain();
  // bem_problem_3d.read_input_mesh_file(0,bem_problem_3d.tria);
  std::cout<<"Computing reference rigid velocities"<<std::endl;
  bem_problem_3d.reinit();
  double exact_omega=-2.*numbers::PI/bem_problem_3d.n_frames/bem_problem_3d.time_step;
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0, true);
  bem_problem_3d.create_wall_body_index_sets();
  bem_problem_3d.mappingeul = SP(new MappingFEField<2, 3> (bem_problem_3d.map_dh,bem_problem_3d.euler_vec));
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_normal_vector();
  bem_problem_3d.compute_euler_vector(bem_problem_3d.next_euler_vec,1, true);
  // bem_problem_3d.project_shape_velocities();
  // bem_problem_3d.shape_velocities.sadd(0.,exact_omega,bem_problem_3d.N_rigid[3]);
  bem_problem_3d.compute_rotational_shape_velocities(bem_problem_3d.shape_velocities,bem_problem_3d.N_rigid[3]);
  bool correction_on_V = true;
  bem_problem_3d.assemble_stokes_system(correction_on_V);
  if (bem_problem_3d.reassemble_preconditoner)
    {
      bem_problem_3d.direct_trilinos_preconditioner.set_up(bem_problem_3d.solver_control);
      bem_problem_3d.direct_trilinos_preconditioner.initialize(bem_problem_3d.monolithic_system_matrix);
    }
  bem_problem_3d.solve_system(bem_problem_3d.monolithic_bool);
  Vector<double> reference_rigid_vel(bem_problem_3d.rigid_velocities);

  bem_problem_3d.update_system_state(true,0,false,false,"Forward");
  bem_problem_3d.output_save_stokes_results(0);



  bem_problem_3d.reinit_for_new_time(0);
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0, true);
  bem_problem_3d.create_wall_body_index_sets();

  Vector<double> translation(3);
  translation[0]=0.0;
  translation[1]=3.54097;
  translation[2]=-8.38246;

  for (unsigned int i=0; i<bem_problem_3d.euler_vec.size()/3; ++i)
    {
      for (unsigned int d=0; d<3; ++d)
        {
          bem_problem_3d.euler_vec[i+d*bem_problem_3d.euler_vec.size()/3]+=translation[d];
        }
    }
  // bem_problem_3d.mappingeul = SP(new MappingFEField<2, 3> (bem_problem_3d.map_dh,bem_problem_3d.euler_vec));
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_normal_vector();
  bem_problem_3d.compute_euler_vector(bem_problem_3d.next_euler_vec,1, true);
  bem_problem_3d.compute_rotational_shape_velocities(bem_problem_3d.shape_velocities,bem_problem_3d.N_rigid[3]);
  std::cout<<exact_omega<<std::endl;

  std::cout<<translation[2]*exact_omega<<" "<<translation[1]*exact_omega<<std::endl;
  for (unsigned int i=0; i<bem_problem_3d.dh_stokes.n_dofs()/3; ++i)
    {
      if (bem_problem_3d.flagellum_cpu_set.is_element(i))
        {
          bem_problem_3d.shape_velocities[i+bem_problem_3d.dh_stokes.n_dofs()/3]+=translation[2]*exact_omega;
          bem_problem_3d.shape_velocities[i+2*bem_problem_3d.dh_stokes.n_dofs()/3]-=translation[1]*exact_omega;
        }
    }
  // bem_problem_3d.shape_velocities.sadd(1.,translation[2]*2.*numbers::PI/bem_problem_3d.time_step/bem_problem_3d.n_frames,bem_problem_3d.N_rigid[1]);
  // bem_problem_3d.shape_velocities.sadd(1.,-translation[1]*2.*numbers::PI/bem_problem_3d.time_step/bem_problem_3d.n_frames,bem_problem_3d.N_rigid[2]);
  bem_problem_3d.assemble_stokes_system(correction_on_V);
  if (bem_problem_3d.reassemble_preconditoner)
    {
      bem_problem_3d.direct_trilinos_preconditioner.set_up(bem_problem_3d.solver_control);
      bem_problem_3d.direct_trilinos_preconditioner.initialize(bem_problem_3d.monolithic_system_matrix);
    }
  bem_problem_3d.solve_system(bem_problem_3d.monolithic_bool);

  bem_problem_3d.rigid_velocities[0] -= bem_problem_3d.rigid_velocities[4] * (0.-translation[2]) - bem_problem_3d.rigid_velocities[5] * (0.-translation[1]);
  bem_problem_3d.rigid_velocities[1] -= bem_problem_3d.rigid_velocities[5] * (0.-translation[0]) - bem_problem_3d.rigid_velocities[3] * (0.-translation[2]);
  bem_problem_3d.rigid_velocities[2] -= bem_problem_3d.rigid_velocities[3] * (0.-translation[1]) - bem_problem_3d.rigid_velocities[4] * (0.-translation[0]);

  for (unsigned int i=0; i<bem_problem_3d.rigid_velocities.size(); ++i)
    {
      if (std::fabs(bem_problem_3d.rigid_velocities[i]-reference_rigid_vel[i])/reference_rigid_vel[i]>tol)
        std::cout<<" Error Reference "<<reference_rigid_vel[i]<<" Vs Translated "<<bem_problem_3d.rigid_velocities[i]<<std::endl;
      else
        std::cout<<" OK !"<<std::endl;
    }
  bem_problem_3d.update_system_state(true,0,false,false,"Forward");
  bem_problem_3d.output_save_stokes_results(99);


  return 0;
}
