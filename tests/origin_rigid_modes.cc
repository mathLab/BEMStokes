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
  double tol=1e-13;
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");
  unsigned int degree = 1;
  BEMProblem<3> bem_problem_3d;
  ParsedFunction<3> exact_rigid_mode_0("Exact rigid mode 0", 3,"1;0;0");
  ParsedFunction<3> exact_rigid_mode_1("Exact rigid mode 1", 3,"0;1;0");
  ParsedFunction<3> exact_rigid_mode_2("Exact rigid mode 2",3, "0;0;1");
  ParsedFunction<3> exact_rigid_mode_3("Exact rigid mode 3", 3,"0;-z;y");
  ParsedFunction<3> exact_rigid_mode_4("Exact rigid mode 4",3, "z;0;-x");
  ParsedFunction<3> exact_rigid_mode_5("Exact rigid mode 5",3, "-y;x;0");

  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
    bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.create_box_bool=false;
  bem_problem_3d.reflect_kernel=false;
  bem_problem_3d.no_slip_kernel=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();


  bem_problem_3d.grid_type="Real";
  bem_problem_3d.force_pole="Origin";
  bem_problem_3d.monolithic_bool=true;
  bem_problem_3d.solve_directly=false;
  bem_problem_3d.reassemble_preconditoner=true;
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="spiral_";
  bem_problem_3d.input_grid_format="msh";

  bem_problem_3d.read_domain();
  // bem_problem_3d.read_input_mesh_file(0,bem_problem_3d.tria);
  bem_problem_3d.reinit();
  bem_problem_3d.compute_euler_vector(bem_problem_3d.euler_vec,0, true);
  bem_problem_3d.create_wall_body_index_sets();
  bem_problem_3d.mappingeul = SP(new MappingFEField<2, 3> (bem_problem_3d.map_dh,bem_problem_3d.euler_vec));
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_euler_vector(bem_problem_3d.next_euler_vec,1, true);



  std::vector<Point<3> > support_points(bem_problem_3d.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<2, 3>( *bem_problem_3d.mappingeul, bem_problem_3d.dh_stokes, support_points);

  std::vector<Vector<double> >values(support_points.size(), Vector<double> (3));

  exact_rigid_mode_0.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<0<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[0][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[0][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[0][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        {
          std::cout<<"Error "<<i<<" "<<test<<std::endl;
          for (unsigned int d=0; d<3; ++d)
            {
              std::cout<<bem_problem_3d.N_rigid[0][i+d*support_points.size()/3]<<" : "<<values[i][d]<<std::endl;
            }
        }
      else
        std::cout<<"OK"<<std::endl;
    }
  exact_rigid_mode_1.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<1<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[1][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[1][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[1][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        std::cout<<"Error "<<i<<" "<<test<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }
  exact_rigid_mode_2.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<2<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[2][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[2][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[2][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        std::cout<<"Error "<<i<<" "<<test<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }
  exact_rigid_mode_3.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<3<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[3][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[3][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[3][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        std::cout<<"Error "<<i<<" "<<test<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }
  exact_rigid_mode_4.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<4<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[4][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[4][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[4][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        std::cout<<"Error "<<i<<" "<<test<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }
  exact_rigid_mode_5.vector_value_list(support_points, values);
  std::cout<<"Testing the "<<5<<"th rigid mode"<<std::endl;
  for (unsigned int i=0; i<support_points.size()/3; ++i)
    {
      double test=std::abs(bem_problem_3d.N_rigid[5][i]-values[i][0])+std::abs(bem_problem_3d.N_rigid[5][i+support_points.size()/3]-values[i][1])+std::abs(bem_problem_3d.N_rigid[5][i+2*support_points.size()/3]-values[i][2]);
      if (test>tol)
        std::cout<<"Error "<<i<<" "<<test<<std::endl;
      else
        std::cout<<"OK"<<std::endl;
    }



  return 0;
}
