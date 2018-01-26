#include "bem_stokes.h"
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
using namespace dealii;
using namespace BEMStokes;




int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-8;
  const unsigned int dim = 3;
  BEMProblem<dim> bem_problem_3d;
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  bem_problem_3d.pcout<<"Minimum Test for the preconditioner with interior problem and the monolithic system"<<std::endl;
  bem_problem_3d.pcout<<"We consider the wall to have normal parallel to the i axis"<<std::endl;
  bem_problem_3d.use_internal_alpha=false;
  bem_problem_3d.create_box_bool=false;
  for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
    bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.reflect_kernel=true;
  bem_problem_3d.input_grid_base_name="sphere_half_refined_";
  bem_problem_3d.input_grid_format="inp";
  bem_problem_3d.fe_stokes = SP(bem_problem_3d.parsed_fe_stokes());
  bem_problem_3d.fe_map = SP(bem_problem_3d.parsed_fe_mapping());
  bem_problem_3d.read_domain();
  // bem_problem_3d.tria.refine_global();

  bem_problem_3d.reinit();
  VectorTools::get_position_vector(bem_problem_3d.map_dh,bem_problem_3d.euler_vec);
  bem_problem_3d.mappingeul = SP(new MappingQ<dim-1,dim>(1));//SP(new MappingFEField<2, 3>(bem_problem_2d.map_dh,bem_problem_2d.euler_vec));

  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_normal_vector();
  bool correction_on_V = true;
  bem_problem_3d.assemble_stokes_system(correction_on_V);

  bem_problem_3d.pcout<<"Solving directly the monolithic system"<<std::endl;
  TrilinosWrappers::MPI::Vector reference_monolithic_solution(bem_problem_3d.monolithic_cpu_set, bem_problem_3d.mpi_communicator);
  TrilinosWrappers::SolverDirect solvy(bem_problem_3d.solver_control);
  solvy.initialize(bem_problem_3d.monolithic_system_matrix);
  solvy.solve(reference_monolithic_solution, bem_problem_3d.monolithic_rhs);//monolithic_system_matrix,

  bem_problem_3d.pcout<<"Testing null velocity on the simmetry plane"<<std::endl;
  unsigned int size_length = 20;
  int half_size_length = size_length/2;
  double h_length = 1.;
  std::vector<Point<dim> > test_points(size_length*size_length);
  std::vector<Tensor<2,dim> > val_gradients(test_points.size());
  for (unsigned int i=0; i<size_length; ++i)
    {
      for (unsigned int j=0; j<size_length; ++j)
        {
          test_points[i*size_length+j]=bem_problem_3d.wall_positions[0];
          test_points[i*size_length+j][0]-=h_length*(-half_size_length+(int)i);
          test_points[i*size_length+j][2]-=h_length*(-half_size_length+(int)j);
        }

    }
  // const Vector<double> loc_wall_velocities(bem_problem_3d.wall_velocities);
  // const Vector<double> loc_shape_velocities(bem_problem_3d.shape_velocities);
  // const Vector<double> loc_rigid_puntual_velocities(bem_problem_3d.rigid_puntual_velocities);
  const Vector<double> loc_stokes_forces(bem_problem_3d.stokes_forces);

  const Vector<double> loc_velocities(bem_problem_3d.total_velocities);
  // loc_velocities.sadd(0.,1.,loc_shape_velocities);
  // loc_velocities.sadd(1.,1.,loc_wall_velocities);
  // loc_velocities.sadd(1.,1.,loc_rigid_puntual_velocities);
  Vector<double> val_velocities(dim*test_points.size());
  bem_problem_3d.evaluate_stokes_bie(test_points, loc_velocities, loc_stokes_forces, val_velocities);
  // for(unsigned p = 0; p<test_points.size(); ++p)
  // {
  //   std::cout<<"point : "<<test_points[p]<<" velocity : ";
  //   for(unsigned int pdim =0 ; pdim < dim; ++pdim)
  //     std::cout<<val_velocities[p+pdim*test_points.size()]<<" ";
  //   std::cout<<std::endl;
  // }

  for (unsigned int i = val_velocities.size()/dim; i<val_velocities.size()/dim*2; ++i)
    {

      double foo = std::abs(val_velocities[i]);
      if (foo>tol)
        bem_problem_3d.pcout<<"ERROR, index i : "<<i<<" : "<<foo<<" , instead of : "<<0<<std::endl;
      else
        bem_problem_3d.pcout<<"OK"<<std::endl;
    }

  bem_problem_3d.tria.set_manifold(0);

  return 0;
}
