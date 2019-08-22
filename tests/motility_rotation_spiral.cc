#include <bem_stokes.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
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



int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace dealii;
  using namespace BEMStokes;

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=6e-3;
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  std::cout<<"Test for the Motility tensor of a sphere in free space"<<std::endl;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");
  unsigned int degree = 1;
  BEMProblem<3> bem_problem_3d;
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  bem_problem_3d.create_box_bool=false;
  for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
    bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.reflect_kernel=false;
  bem_problem_3d.no_slip_kernel=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();
  bem_problem_3d.use_internal_alpha=false;
  bem_problem_3d.grid_type="ImposedVelocity";
  bem_problem_3d.monolithic_bool=true;
  bem_problem_3d.solve_directly=true;
  bem_problem_3d.reassemble_preconditoner=true;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="spiral_";
  bem_problem_3d.input_grid_format="msh";
  bem_problem_3d.read_input_mesh_file(0,bem_problem_3d.tria);
  bem_problem_3d.reinit();
  // VectorTools::get_position_vector(bem_problem_3d.map_dh,bem_problem_3d.euler_vec);
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0);
  bem_problem_3d.create_wall_body_index_sets();
  bem_problem_3d.mappingeul = std::make_shared<MappingFEField<2, 3> > (bem_problem_3d.map_dh,bem_problem_3d.euler_vec);
  for (unsigned int i=3; i<4; ++i)
    {
      bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
      bem_problem_3d.compute_normal_vector();
      bool correction_on_V = true;
      bem_problem_3d.assemble_stokes_system(correction_on_V);
      for (unsigned int j=0; j<6; ++j)
        bem_problem_3d.monolithic_rhs[bem_problem_3d.dh_stokes.n_dofs()+j]=0.;
      bem_problem_3d.monolithic_rhs[bem_problem_3d.dh_stokes.n_dofs()+i]=1.;
      bem_problem_3d.solve_system(bem_problem_3d.monolithic_bool);
      bem_problem_3d.output_save_stokes_results(0);
      auto backup_velocity=bem_problem_3d.rigid_velocities;
      auto backup_forces=bem_problem_3d.rigid_total_forces;
      unsigned int frame=30;
      double angle = -2*numbers::PI * frame/120;
      bem_problem_3d.reinit_for_new_time(frame);
      bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,frame);
      bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
      bem_problem_3d.compute_normal_vector();
      bem_problem_3d.assemble_stokes_system(correction_on_V);
      for (unsigned int j=0; j<6; ++j)
        bem_problem_3d.monolithic_rhs[bem_problem_3d.dh_stokes.n_dofs()+j]=0.;
      bem_problem_3d.monolithic_rhs[bem_problem_3d.dh_stokes.n_dofs()+i]=1.;
      bem_problem_3d.solve_system(bem_problem_3d.monolithic_bool);
      bem_problem_3d.output_save_stokes_results(frame);
      Vector<double> quaternion(4);
      quaternion[0]=cos(angle/2);
      quaternion[1]=sin(angle/2);

      FullMatrix<double> rotation_check_matrix(3,3);
      bem_problem_3d.compute_rotation_matrix_from_quaternion(rotation_check_matrix, quaternion);
      Vector<double> foo_in_v(3), foo_out_v(3) , foo_in_w(3), foo_out_w(3);
      for (unsigned int j=0; j<3; ++j)
        {
          foo_in_v[j]=backup_forces[j];
          foo_in_w[j]=backup_forces[j+3];
        }
      foo_in_v.print(std::cout);
      foo_in_w.print(std::cout);
      rotation_check_matrix.print(std::cout);
      rotation_check_matrix.vmult(foo_out_v, foo_in_v);
      rotation_check_matrix.vmult(foo_out_w, foo_in_w);
      foo_out_v.print(std::cout);
      foo_out_w.print(std::cout);
      for (unsigned int j=0; j<3; ++j)
        {
          backup_forces[j]=foo_out_v[j];
          backup_forces[j+3]=foo_out_w[j];
        }
      for (unsigned int j=0; j<6; ++j)
        {
          std::cout<<bem_problem_3d.rigid_total_forces[j]<<" : "<<backup_forces[j];
          std::cout<<" --- "<<bem_problem_3d.rigid_velocities[j]<<" : "<<backup_velocity[j]<<std::endl;
          // if(j!=i)
          // {
          //   double ratio=std::fabs(bem_problem_3d.rigid_total_forces[j]/bem_problem_3d.rigid_total_forces[i]);
          //   if(ratio < tol)
          //     std::cout<<"OK"<<std::endl;
          //   else
          //     std::cout<<ratio<<std::endl;
          // }
        }


    }

  //
  // std::vector<DataComponentInterpretation::DataComponentInterpretation>
  // data_component_interpretation
  // (3, DataComponentInterpretation::component_is_part_of_vector);
  // DataOut<3-1, DoFHandler<3-1, 3> > dataout;
  //
  // dataout.attach_dof_handler(bem_problem_3d.dh_stokes);
  // dataout.add_data_vector(t1_ex, std::vector<std::string > (3,"G_trace_1_ex"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  // dataout.add_data_vector(t0, std::vector<std::string > (3,"G_velocities_0"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  // dataout.add_data_vector(t1, std::vector<std::string > (3,"G_trace_1_0"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  // dataout.add_data_vector(t1_diff, std::vector<std::string > (3,"G_trace_1_error"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  // dataout.add_data_vector(normal, std::vector<std::string > (3,"normal_vector"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  // dataout.build_patches(*bem_problem_3d.mappingeul,
  //                 degree,
  //                 DataOut<3-1, DoFHandler<3-1, 3> >::curved_inner_cells);
  //
  // std::string filename = ( "G_check_"+ Utilities::int_to_string(degree) +
  //                    ".vtu" );
  // std::ofstream file(filename.c_str());
  //
  // dataout.write_vtu(file);
  //
  //

  return 0;
}
