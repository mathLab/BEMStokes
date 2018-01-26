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
int main (int argc, char **argv)
{
  using namespace dealii;
  using namespace BEMStokes;
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=6e-4;
  unsigned int ncycles = 3;
  unsigned int max_degree = 3;
  const unsigned int dim = 2;

  std::cout<<"Test on the Geometry in 3D. We test the convergence of the euler vector"<<std::endl;
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
      bem_problem_2d.build_sphere_in_deal=true;
      bem_problem_2d.fe_stokes = SP(fe_builder23());
      bem_problem_2d.fe_map = SP(fe_builder23());
      bem_problem_2d.read_domain();
      SphericalManifold<dim-1,dim> manifold;
      bem_problem_2d.tria.set_all_manifold_ids(0);
      bem_problem_2d.tria.set_manifold(0, manifold);


      for (unsigned int cycle=0; cycle<ncycles; ++cycle)
        {
          bem_problem_2d.reinit();
          bem_problem_2d.compute_euler_vector(bem_problem_2d.euler_vec,0);
          if (cycle == 0)
            bem_problem_2d.mappingeul = SP(new MappingFEField<dim-1, dim>(bem_problem_2d.map_dh,bem_problem_2d.euler_vec));
          // for(unsigned int i=0; i< bem_problem_2d.euler_vec.size()/dim; ++i)
          // {
          //   double foo = std::pow(bem_problem_2d.euler_vec[i*dim+0]*bem_problem_2d.euler_vec[i*dim+0]+
          //               bem_problem_2d.euler_vec[i*dim+1]*bem_problem_2d.euler_vec[i*dim+1]+
          //               bem_problem_2d.euler_vec[i*dim+2]*bem_problem_2d.euler_vec[i*dim+2],0.5);
          // }
          // std::cout<<type(bem_problem_2d.map_dh)<<std::endl;
          eh.error_from_exact(*bem_problem_2d.mappingeul, bem_problem_2d.map_dh, bem_problem_2d.euler_vec, exact_solution_id,0);
          eh.error_from_exact(*bem_problem_2d.mappingeul, bem_problem_2d.map_dh, bem_problem_2d.euler_vec, exact_solution_pos,1);

          if (cycle != ncycles-1)
            bem_problem_2d.tria.refine_global(1);
          else
            {
              std::vector<DataComponentInterpretation::DataComponentInterpretation>
              data_component_interpretation
              (dim, DataComponentInterpretation::component_is_part_of_vector);
              DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;

              dataout.attach_dof_handler(bem_problem_2d.dh_stokes);
              dataout.add_data_vector(bem_problem_2d.euler_vec, std::vector<std::string > (dim,"euler_vec"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
              // dataout.build_patches(bem_problem_2d.fe_stokes->degree);
              dataout.build_patches(*bem_problem_2d.mappingeul,
                                    bem_problem_2d.fe_stokes->degree,
                                    DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

              std::string filename = ( "euler_check_"+ Utilities::int_to_string(degree) +
                                       ".vtu" );
              std::ofstream file(filename.c_str());

              dataout.write_vtu(file);

              bem_problem_2d.tria.set_manifold(0);
            }
        }
      eh.output_table(std::cout,0);
      eh.output_table(std::cout,1);
    }

  return 0;
}
