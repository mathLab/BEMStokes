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
#include <deal.II/fe/fe_tools.h>

#include <mpi.h>

#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_parallel_vector.h>
#include <deal.II/lac/slepc_solver.h>

using namespace deal2lkit;

template<int my_dim>
void impose_G_as_velocity(const BEMStokes::BEMProblem<my_dim> &bem, const Point<my_dim> &source, TrilinosWrappers::MPI::Vector &G_velocities)
{
  std::vector<Point<my_dim> > support_points(bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (*bem.mappingeul,
                                                          bem.dh_stokes, support_points);

  for (unsigned int i=0; i<bem.dh_stokes.n_dofs()/my_dim; ++i)
    {
      if (bem.this_cpu_set.is_element(i))
        {
          const Tensor<1,my_dim> R = support_points[i] - source;
          Tensor<2,my_dim> G = bem.stokes_kernel.value_tens(R) ;
          for (unsigned int jdim=0; jdim<my_dim; ++jdim)
            {
              G_velocities[i+bem.dh_stokes.n_dofs()/my_dim*jdim] = G[jdim][0];
            }
        }
    }
  G_velocities.compress(VectorOperation::insert);

}
template<int my_dim>
void impose_G_as_trace_1(const Point<my_dim> &source, BEMStokes::BEMProblem<my_dim> &bem,  TrilinosWrappers::MPI::Vector &normals, TrilinosWrappers::MPI::Vector &G_trace_1)
{
  std::vector<Point<my_dim> > support_points(bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim>(*bem.mappingeul, //StaticMappingQ1<my_dim-1, my_dim>::mapping,
                                                         bem.dh_stokes, support_points);

  for (unsigned int i=0; i<bem.dh_stokes.n_dofs()/my_dim; ++i)
    {
      if (bem.this_cpu_set.is_element(i))
        {
          const Tensor<1,my_dim> R = support_points[i] - source;
          Point<my_dim> normal;
          for (unsigned int jdim=0; jdim<my_dim; ++jdim)
            normal[jdim] = - normals[i+bem.dh_stokes.n_dofs()/my_dim*jdim];
          Tensor<3,my_dim> W = bem.stokes_kernel.value_tens2(R) ;
          Tensor<2,my_dim> singular_ker = bem.compute_singular_kernel(normal, W) ;
          for (unsigned int jdim=0; jdim<my_dim; ++jdim)
            G_trace_1[i+bem.dh_stokes.n_dofs()/my_dim*jdim] = 1 * singular_ker[jdim][0];
        }
    }
  G_trace_1.compress(VectorOperation::insert);

}


int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  using namespace dealii;
  using namespace BEMStokes;

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=6e-4;
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  std::cout<<"Test that the BEM is able to recover the Fundamental Solution"<<std::endl;
  ParsedFunction<2> exact_solution_trace("PIPPO",2,"0 ; 0");
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");


  const unsigned int dim =2;
  for (unsigned int degree=1; degree<=max_degree; degree++)
    {
      Point<dim> source_point(0.3, 0.3);
      std::cout<< "Testing for degree = "<<degree<<std::endl;
      BEMProblem<2> bem_problem_2d;
      // PETScWrappers::SparseMatrix             V;
      // std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
      // std::vector<Vector<double> > M_mult_eigenfunctions;
      // std::vector<double>                     eigenvalue_tenss;
      // PETScWrappers::MPI::Vector normal_vector_difference;
      ErrorHandler<1> eh("Error 2D","u,u","L2, H1, Linfty; AddUp");
      ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d_Q2.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
      bem_problem_2d.create_box_bool=false;
      for (unsigned int i = 0 ; i<bem_problem_2d.wall_bool.size(); ++i)
        bem_problem_2d.wall_bool[i]=false;
      bem_problem_2d.reflect_kernel=false;
      bem_problem_2d.no_slip_kernel=false;
      bem_problem_2d.use_internal_alpha=false;
      bem_problem_2d.convert_bool_parameters();
      bem_problem_2d.input_grid_path=SOURCE_DIR"/grid_test/";
      bem_problem_2d.input_grid_format="inp";
      bem_problem_2d.input_grid_base_name="circle_";
      std::string fe_name = "FESystem<1,2>[FE_Q<1,2>(2)^2]";
      std::string fe_name_map = "FESystem<1,2>[FE_Q<1,2>(2)^2]";
      bem_problem_2d.fe_stokes = std::unique_ptr(FETools::get_fe_by_name<1,2> (fe_name));
      bem_problem_2d.fe_map = std::unique_ptr(FETools::get_fe_by_name<1,2> (fe_name_map));
      bem_problem_2d.extra_debug_info = false;
      // ParameterAcceptor::initialize("parameters_16.prm", "used_parameters_16.prm");
      bem_problem_2d.read_domain();
      SphericalManifold<1,2> manifold;
      bem_problem_2d.tria.set_all_manifold_ids(0);
      bem_problem_2d.tria.set_manifold(0, manifold);
      for (unsigned int cycle=0; cycle<ncycles; ++cycle)
        {
          bem_problem_2d.pcout<<"Cycle : "<<cycle<<" "<<ncycles<<std::endl;
          bem_problem_2d.reinit();
          // ncycles = bem_problem_2d.n_cycles;
          VectorTools::get_position_vector(bem_problem_2d.map_dh,bem_problem_2d.euler_vec);
          if (cycle == 0)
            bem_problem_2d.mappingeul = SP(new MappingFEField<1, 2>(bem_problem_2d.map_dh,bem_problem_2d.euler_vec));


          bem_problem_2d.compute_center_of_mass_and_rigid_modes(0);

          bem_problem_2d.compute_normal_vector();

          bem_problem_2d.assemble_stokes_system(false);

          TrilinosWrappers::MPI::Vector G_velocities(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator),
                           G_trace_1(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator),
                           G_trace_1_ex(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator),
                           trace_1_vector_difference(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator);


          impose_G_as_velocity(bem_problem_2d, source_point, G_velocities);
          std::cout<<G_velocities.l2_norm()<<" "<<G_velocities.linfty_norm()<<std::endl;
          impose_G_as_trace_1(source_point, bem_problem_2d, bem_problem_2d.normal_vector, G_trace_1_ex);
          // G_trace_1_ex.print(std::cout);
          // bem_problem_2d.assemble_preconditioner();
          bem_problem_2d.dirichlet_to_neumann_operator(G_velocities, G_trace_1);

          TrilinosWrappers::MPI::Vector foo(bem_problem_2d.this_cpu_set, bem_problem_2d.mpi_communicator);
          foo.sadd(0.,1.,G_trace_1_ex);
          bem_problem_2d.tangential_projector(foo, G_trace_1_ex);
          for (unsigned int i=0 ; i<bem_problem_2d.dh_stokes.n_dofs(); ++i)
            if (bem_problem_2d.this_cpu_set.is_element(i))
              {
                trace_1_vector_difference[i] = std::fabs(G_trace_1_ex[i] - G_trace_1[i]);
              }

          // std::cout<<exact_solution_trace.n_components<<std::endl;
          Vector<double> t1_diff(trace_1_vector_difference),
                 t1(G_trace_1),
                 t1_ex(G_trace_1_ex),
                 normal(bem_problem_2d.normal_vector),
                 t0(G_velocities);
          eh.error_from_exact(*bem_problem_2d.mappingeul, bem_problem_2d.dh_stokes, t1_diff, ZeroFunction<dim, double> (dim),0);
          // t1_ex.print(std::cout);
          if (cycle != ncycles-1)
            bem_problem_2d.tria.refine_global(1);
          else if (bem_problem_2d.this_mpi_process==0)
            {
              std::vector<DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation
              (dim, DataComponentInterpretation::component_is_part_of_vector);
              DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;

              dataout.attach_dof_handler(bem_problem_2d.dh_stokes);
              dataout.add_data_vector(t1_ex, std::vector<std::string > (dim,"G_trace_1_ex"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t0, std::vector<std::string > (dim,"G_velocities_0"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t1, std::vector<std::string > (dim,"G_trace_1_0"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t1_diff, std::vector<std::string > (dim,"G_trace_1_error"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(normal, std::vector<std::string > (dim,"normal_vector"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.build_patches(*bem_problem_2d.mappingeul,
                                    bem_problem_2d.fe_stokes->degree,
                                    DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

              std::string filename = ( "G_check_"+ Utilities::int_to_string(degree) +
                                       ".vtu" );
              std::ofstream file(filename.c_str());

              dataout.write_vtu(file);

              std::string file_name1;
              file_name1 = "G_trace_1_" + Utilities::int_to_string(dim) + "d.bin";
              std::ofstream forces (file_name1.c_str());
              t1.block_write(forces);
              std::string file_name2;
              file_name2 = "G_trace_0_" + Utilities::int_to_string(dim) + "d.bin";
              std::ofstream velocities (file_name2.c_str());
              t0.block_write(velocities);
              eh.output_table(std::cout,0);

            }

        }
      bem_problem_2d.tria.set_manifold(0);

    }

  return 0;
}
