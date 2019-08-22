#include "bem_stokes.h"
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
using namespace dealii;
using namespace BEMStokes;




int main (int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  // const unsigned int degree = 1;
  // const unsigned int mapping_degree = 1;
  double tol=1e-5;
  const unsigned int dim = 3;
  BEMProblem<dim> bem_problem_3d;
  deal2lkit::ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  bem_problem_3d.pcout<<"Minimum Test for the rotation with quaternions"<<std::endl;
  bem_problem_3d.use_internal_alpha=false;
  bem_problem_3d.create_box_bool=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();

  double dt=0.000001;
  FullMatrix<double> rotation_matrix(dim,dim);
  Vector<double> P_0(dim),P_ref(dim),P_test(dim);


  for (unsigned int i=0; i<dim; ++i)
    {
      P_0=0.;
      P_0[(i+0)%dim]=1.;
      // P_test = P_0;
      rotation_matrix=0.;
      for (unsigned int j=0; j<dim; ++j)
        rotation_matrix.set(j,j,1.);
      // rotation_matrix.set(0,1,-1.);
      // rotation_matrix.set(1,0,1.);
      // rotation_matrix.set(2,2,1.);
      Vector<double> omega(dim);
      // omega[(i+2)%dim]=numbers::PI/2;//*1/dt;
      // omega.print(std::cout);
      P_0.print(std::cout);
      for (unsigned int j=0; j<(int)(1)/dt; ++j)
        {
          // rotation_matrix.print_formatted(std::cout);
          omega[(i+2)%dim]=cos(2*numbers::PI*j/((int)(1)/dt))*(2*numbers::PI);
          bem_problem_3d.update_rotation_matrix(rotation_matrix,omega,dt);
          // rotation_matrix.print_formatted(std::cout);
          rotation_matrix.vmult(P_test,P_0);
          // P_test.print(std::cout);
          if (j%1000==0)
            {
              Point<dim> P_ref,PP_0;
              for (unsigned int k=0; k<dim; ++k)
                PP_0[k]=P_0[k];
              Point<dim> axis;
              axis[(i+2)%dim]=1.;
              bem_problem_3d.apply_rotation_along_axis(P_ref,PP_0,axis,sin(2*numbers::PI*j/((int)(1)/dt)));
              std::cout<<"Testing j = "<<j<<std::endl;
              for (unsigned int k=0; k<dim; ++k)
                {
                  if (std::abs(P_ref[k]-P_test[k])>tol)
                    {
                      std::cout<<"ERROR !!!"<<std::endl;
                      std::cout<<PP_0<<" : "<<P_ref<<std::endl;
                      P_test.print(std::cout);
                      break;
                    }
                  else
                    {
                      std::cout<<"OK : ";
                    }
                }
              std::cout<<std::endl;

            }
        }


    }



}
