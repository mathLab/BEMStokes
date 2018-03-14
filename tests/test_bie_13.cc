
#include "bem_stokes.h"
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/base/utilities.h>

#include <iostream>
#include <fstream>
#include <deal2lkit/error_handler.h>
#include <deal2lkit/parsed_function.h>
#include <deal2lkit/parameter_acceptor.h>
#include <deal2lkit/utilities.h>
int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=6e-4;
  unsigned int ncycles = 2;
  unsigned int max_degree = 1;
  const unsigned int dim=2;
  std::cout<<"Test on the Geometry in 3D. We test the convergence of the normal vector"<<std::endl;
  ParsedFunction<dim> exact_solution_id("Exact solution identity",dim,"x ; y");
  ParsedFunction<dim> exact_solution_pos("Exact solution position",dim,
                                         "x / (x*x + y*y)^0.5 ; y / (x*x + y*y)^0.5");

  for (unsigned int degree=1; degree<=max_degree; degree++)
    {
      std::cout<< "Testing for degree = "<<degree<<std::endl;
      ErrorHandler<2> eh("","u,u","L2, H1, Linfty; AddUp");

      ParsedFiniteElement<dim-1,dim> fe_builder23("ParsedFiniteElement<1,2>",
                                                  "FESystem[FE_Q("+Utilities::int_to_string(degree)+")^2]",
                                                  "u,u");

      std::cout<< "Testing for degree = "<<degree<<std::endl;
      BEMProblem<dim> bem_problem_2d;
      ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box_2d.prm", "used_parameters_14.prm");
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
      bem_problem_2d.use_internal_alpha=false;
      std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
      bem_problem_2d.input_grid_path=mesh_filename_path;
      bem_problem_2d.input_grid_base_name="circle_";
      bem_problem_2d.input_grid_format="inp";
      bem_problem_2d.fe_stokes = (fe_builder23());
      bem_problem_2d.fe_map = (fe_builder23());
      bem_problem_2d.read_domain();
      SphericalManifold<dim-1,dim> manifold;
      bem_problem_2d.tria.set_all_manifold_ids(0);
      bem_problem_2d.tria.set_manifold(0, manifold);



      for (unsigned int cycle=0; cycle<ncycles; ++cycle)
        {
          bem_problem_2d.reinit();
          FESystem<dim-1,dim> fee(FE_DGQArbitraryNodes<dim-1,dim> (QGauss<1> (bem_problem_2d.quadrature.size()/(dim-1))),dim);
          DoFHandler<dim-1, dim> dh(bem_problem_2d.tria);

          Vector<double> normal_vector;

          dh.distribute_dofs(fee);
          normal_vector.reinit(dh.n_dofs());
          //std::cout<<bem_problem_2d.map_dh.get_fe().degree<<std::endl;
          VectorTools::get_position_vector(bem_problem_2d.map_dh,bem_problem_2d.euler_vec);
          // for(unsigned int i=0; i<bem_problem_2d.map_dh.n_dofs(); ++i)
          //   if (bem_problem_2d.euler_vec(i) == 0)
          //     std::cout<<"non sense at i :"<<i<<std::endl;
          if (cycle == 0)
            bem_problem_2d.mappingeul = SP(new MappingFEField<dim-1, dim>(bem_problem_2d.map_dh,bem_problem_2d.euler_vec));

          std::vector<types::global_dof_index> local_dof_indices(fee.dofs_per_cell);
          typename DoFHandler<dim-1,dim>::active_cell_iterator
          cell = dh.begin_active(),
          endc = dh.end();
          FEValues<dim-1,dim> fee_v(*bem_problem_2d.mappingeul, fee, fee.get_unit_support_points(),
                                    update_cell_normal_vectors |
                                    update_quadrature_points );


          for (cell = dh.begin_active(); cell != endc; ++cell)
            {
              fee_v.reinit(cell);
              cell->get_dof_indices(local_dof_indices);
              const std::vector<Point<dim> > &q_points = fee_v.get_quadrature_points();

              const std::vector<Tensor<1, dim> > &normals  = fee_v.get_all_normal_vectors();

              unsigned int n_q_points = q_points.size();

              for (unsigned int q=0; q<n_q_points; ++q)
                {
                  normal_vector(local_dof_indices[q]) = -normals[q][fee.system_to_component_index(q).first];
                }


            }

          bem_problem_2d.compute_center_of_mass_and_rigid_modes(0);
          bem_problem_2d.compute_normal_vector();
          Vector<double> normal_foo(bem_problem_2d.normal_vector);
          normal_foo.sadd(1.,0.,normal_foo);
          normal_vector.sadd(-1.,0.,normal_vector);
          eh.error_from_exact(*bem_problem_2d.mappingeul, dh, normal_vector, exact_solution_pos,0);
          eh.error_from_exact(*bem_problem_2d.mappingeul, bem_problem_2d.map_dh, normal_foo, exact_solution_pos,1);

          if (cycle != ncycles-1)
            bem_problem_2d.tria.refine_global(1);
          else
            {
              DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;
              std::vector<DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation
              (dim, DataComponentInterpretation::component_is_part_of_vector);

              dataout.attach_dof_handler(bem_problem_2d.dh_stokes);
              dataout.add_data_vector(bem_problem_2d.normal_vector, std::vector<std::string > (dim,"normal"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataout.build_patches(*bem_problem_2d.mappingeul,
                                    bem_problem_2d.fe_stokes->degree,
                                    DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

              std::string filename = ( "normal_projection_check_"+Utilities::int_to_string(degree)+".vtu");
              std::ofstream file(filename.c_str());

              dataout.write_vtu(file);

              DataOut<dim-1, DoFHandler<dim-1, dim> > dataoutdim;

              dataoutdim.attach_dof_handler(dh);
              dataoutdim.add_data_vector(normal_vector, std::vector<std::string > (dim,"normal"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              dataoutdim.build_patches(*bem_problem_2d.mappingeul,
                                       bem_problem_2d.fe_stokes->degree,
                                       DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

              std::string filenamedim = ( "normal_true_check_"+Utilities::int_to_string(degree)+".vtu");
              std::ofstream filedim(filenamedim.c_str());

              dataoutdim.write_vtu(filedim);

              bem_problem_2d.tria.set_manifold(0);


            }
        }
      std::cout<<"Convergence of the plain deal.II normal"<<std::endl;
      eh.output_table(std::cout,0);
      std::cout<<"Convergence of the deal.II normal with L2 projection"<<std::endl;
      eh.output_table(std::cout,1);


    }

  return 0;
}
