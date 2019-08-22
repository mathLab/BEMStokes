#include "bem_stokes.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <iostream>
#include <fstream>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/utilities.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/base/parsed_convergence_table.h>
#include <deal.II/base/parsed_function.h>

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
                                                          bem.map_dh, support_points);

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
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim>(StaticMappingQ1<my_dim-1, my_dim>::mapping,
                                                         bem.map_dh, support_points);

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
  unsigned int ncycles = 2;
  unsigned int max_degree = 1;
  std::cout<<"Test that the BEM is able to recover the Fundamental Solution"<<std::endl;


  const unsigned int dim =3;
  for (unsigned int degree=1; degree<=max_degree; degree++)
    {
      Point<dim> source_point(0.3, 0.3, 0.3);
      std::cout<< "Testing for degree = "<<degree<<std::endl;
      BEMProblem<dim> bem_problem_3d;
      // PETScWrappers::SparseMatrix             V;
      // std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
      // std::vector<Vector<double> > M_mult_eigenfunctions;
      // std::vector<double>                     eigenvalue_tenss;
      // PETScWrappers::MPI::Vector normal_vector_difference;

      ParsedConvergenceTable  eh({"u","u","u"}, {{VectorTools::L2_norm, VectorTools::H1_norm, VectorTools::Linfty_norm}});
      deal2lkit::ParameterAcceptor.prm.enter_subsection("Error 3D");
      eh.add_parameters(deal2lkit::ParameterAcceptor.prm);
      deal2lkit::ParameterAcceptor.prm.leave_subsection("Error 3D");
      // ParsedConvergenceTable eh("Error 3D","u,u,u","L2, H1, Linfty; AddUp; AddUp");
      deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
      bem_problem_3d.create_box_bool=false;
      for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
        bem_problem_3d.wall_bool[i]=false;
      bem_problem_3d.reflect_kernel=false;
      bem_problem_3d.no_slip_kernel=false;
      bem_problem_3d.use_internal_alpha=false;
      bem_problem_3d.input_grid_path=SOURCE_DIR"/grid_test/";
      bem_problem_3d.input_grid_format="inp";
      bem_problem_3d.input_grid_base_name="sphere_";
      bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
      bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();
      bem_problem_3d.read_domain();
      SphericalManifold<2,3> manifold;
      bem_problem_3d.tria.set_all_manifold_ids(0);
      bem_problem_3d.tria.set_manifold(0, manifold);
      for (unsigned int cycle=0; cycle<ncycles; ++cycle)
        {
          bem_problem_3d.pcout<<"Cycle : "<<cycle<<" "<<ncycles<<std::endl;
          bem_problem_3d.reinit();
          // ncycles = bem_problem_3d.n_cycles;
          VectorTools::get_position_vector(bem_problem_3d.map_dh,bem_problem_3d.euler_vec);
          if (cycle == 0)
            bem_problem_3d.mappingeul = std::make_shared<MappingFEField<2, 3> >(bem_problem_3d.map_dh,bem_problem_3d.euler_vec);


          bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);

          bem_problem_3d.compute_normal_vector();

          bem_problem_3d.assemble_stokes_system(false);

          TrilinosWrappers::MPI::Vector G_velocities(bem_problem_3d.this_cpu_set, bem_problem_3d.mpi_communicator),
                           G_trace_1(bem_problem_3d.this_cpu_set, bem_problem_3d.mpi_communicator),
                           G_trace_1_ex(bem_problem_3d.this_cpu_set, bem_problem_3d.mpi_communicator),
                           trace_1_vector_difference(bem_problem_3d.this_cpu_set, bem_problem_3d.mpi_communicator);


          impose_G_as_velocity(bem_problem_3d, source_point, G_velocities);
          std::cout<<G_velocities.l2_norm()<<" "<<G_velocities.linfty_norm()<<std::endl;
          impose_G_as_trace_1(source_point, bem_problem_3d, bem_problem_3d.normal_vector, G_trace_1_ex);
          // G_trace_1_ex.print(std::cout);
          // bem_problem_3d.assemble_preconditioner();
          bem_problem_3d.dirichlet_to_neumann_operator(G_velocities, G_trace_1);

          TrilinosWrappers::MPI::Vector foo(bem_problem_3d.this_cpu_set, bem_problem_3d.mpi_communicator);
          foo.sadd(0.,1.,G_trace_1_ex);
          bem_problem_3d.tangential_projector(foo, G_trace_1_ex);
          for (unsigned int i=0 ; i<bem_problem_3d.dh_stokes.n_dofs(); ++i)
            if (bem_problem_3d.this_cpu_set.is_element(i))
              {
                trace_1_vector_difference[i] = std::fabs(G_trace_1_ex[i] - G_trace_1[i]);
              }

          // std::cout<<exact_solution_trace.n_components<<std::endl;
          Vector<double> t1_diff(trace_1_vector_difference),
                 t1(G_trace_1),
                 t1_ex(G_trace_1_ex),
                 normal(bem_problem_3d.normal_vector),
                 t0(G_velocities);
          eh.error_from_exact(*bem_problem_3d.mappingeul, bem_problem_3d.map_dh, t1_diff, ZeroFunction<3, double> (3),0);
          // t1_ex.print(std::cout);
          if (cycle != ncycles-1)
            bem_problem_3d.tria.refine_global(1);
          else if (bem_problem_3d.this_mpi_process==0)
            {
              std::vector<DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation
              (dim, DataComponentInterpretation::component_is_part_of_vector);
              DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;

              dataout.attach_dof_handler(bem_problem_3d.dh_stokes);
              dataout.add_data_vector(t1_ex, std::vector<std::string > (dim,"G_trace_1_ex"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t0, std::vector<std::string > (dim,"G_velocities_0"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t1, std::vector<std::string > (dim,"G_trace_1_0"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(t1_diff, std::vector<std::string > (dim,"G_trace_1_error"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.add_data_vector(normal, std::vector<std::string > (dim,"normal_vector"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.build_patches(*bem_problem_3d.mappingeul,
                                    bem_problem_3d.fe_stokes->degree,
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
              eh.output_table(std::cout);

            }

        }
      bem_problem_3d.tria.reset_manifold(0);

    }

  return 0;
}
