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

void assemble_preconditioner(TrilinosWrappers::SparsityPattern &sp, TrilinosWrappers::SparseMatrix &preccy, const BEMProblem<3> &bem_problem_3d)
{
  sp.reinit(bem_problem_3d.monolithic_cpu_set, bem_problem_3d.mpi_communicator);
  for (auto i : bem_problem_3d.this_cpu_set)
    {
      if (bem_problem_3d.constraints.is_constrained(i)==true)
        {
          std::vector<std::pair<types::global_dof_index, double> > constr_in = *bem_problem_3d.constraints.get_constraint_entries(i);
          for (unsigned int ic=0; ic<constr_in.size(); ic++)
            {
              sp.add(i,i);
              sp.add(i,constr_in[ic].first);
            }

        }
      else
        for (types::global_dof_index j=0; j<bem_problem_3d.this_cpu_set.size(); ++j)
          {
            sp.add(i,j);
          }
      // for(types::global_dof_index j=bem_problem_3d.this_cpu_set.size(); j<bem_problem_3d.this_cpu_set.size()+bem_problem_3d.num_rigid; ++j)
      //   sp.add(i,j);

    }
  if (bem_problem_3d.this_mpi_process==0)
    for (types::global_dof_index i=bem_problem_3d.dh_stokes.n_dofs(); i<bem_problem_3d.dh_stokes.n_dofs()+bem_problem_3d.num_rigid; ++i)
      {
        // for(types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid; ++j)
        sp.add(i,i);
      }

  sp.compress();

  preccy.clear();
  preccy.reinit(sp);
  for (auto i : bem_problem_3d.this_cpu_set)
    {
      for (types::global_dof_index j=0; j<bem_problem_3d.this_cpu_set.size(); ++j)
        {
          if (bem_problem_3d.monolithic_full_sparsity_pattern.exists(i,j))
            preccy.set(i,j,bem_problem_3d.monolithic_system_matrix(i,j));
        }
    }
  if (bem_problem_3d.this_mpi_process==0)
    for (types::global_dof_index i=bem_problem_3d.dh_stokes.n_dofs(); i<bem_problem_3d.dh_stokes.n_dofs()+bem_problem_3d.num_rigid; ++i)
      {
        // for(types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid; ++j)
        preccy.set(i,i,1.);
      }

}

void assemble_band_preconditioner(TrilinosWrappers::SparsityPattern &sp, TrilinosWrappers::SparseMatrix &preccy, unsigned int bandwith, const BEMProblem<3> &bem_problem_3d)
{
  sp.reinit(bem_problem_3d.monolithic_cpu_set, bem_problem_3d.mpi_communicator);
  for (auto i : bem_problem_3d.this_cpu_set)
    {
      types::global_dof_index starting_index=std::max((int)(i - bandwith),0);
      types::global_dof_index end_index=std::min((int)(i +2 * bandwith),(int) bem_problem_3d.this_cpu_set.size());
      for (types::global_dof_index j=starting_index; j<end_index; ++j)
        {
          sp.add(i,j);
        }
      // for(types::global_dof_index j=bem_problem_3d.this_cpu_set.size(); j<bem_problem_3d.this_cpu_set.size()+bem_problem_3d.num_rigid; ++j)
      //   sp.add(i,j);
    }
  if (bem_problem_3d.this_mpi_process==0)
    for (types::global_dof_index i=bem_problem_3d.dh_stokes.n_dofs(); i<bem_problem_3d.dh_stokes.n_dofs()+bem_problem_3d.num_rigid; ++i)
      {
        // for(types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid; ++j)
        sp.add(i,i);
      }

  sp.compress();

  preccy.clear();
  preccy.reinit(sp);
  for (auto i : bem_problem_3d.this_cpu_set)
    {
      types::global_dof_index starting_index=std::max((int)(i - bandwith),0);
      types::global_dof_index end_index=std::min((int)(i +2 * bandwith),(int) bem_problem_3d.this_cpu_set.size());
      for (types::global_dof_index j=starting_index; j<end_index; ++j)
        {
          if (bem_problem_3d.monolithic_full_sparsity_pattern.exists(i,j))
            preccy.set(i,j,bem_problem_3d.monolithic_system_matrix(i,j));
        }
    }
  if (bem_problem_3d.this_mpi_process==0)
    for (types::global_dof_index i=bem_problem_3d.dh_stokes.n_dofs(); i<bem_problem_3d.dh_stokes.n_dofs()+bem_problem_3d.num_rigid; ++i)
      {
        // for(types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid; ++j)
        preccy.set(i,i,1.);
      }

}

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
  bem_problem_3d.use_internal_alpha=true;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();
  std::string mesh_filename_path(SOURCE_DIR "/grid_test/");
  bem_problem_3d.input_grid_path=mesh_filename_path;
  bem_problem_3d.input_grid_base_name="sphere_half_refined_";
  bem_problem_3d.input_grid_format="inp";
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
  bem_problem_3d.pcout<<"Solving using an iterative solver combined with a preconditioner"<<std::endl;
  bem_problem_3d.monolithic_solution=0.;
  TrilinosWrappers::SparsityPattern sp;
  TrilinosWrappers::SparseMatrix my_prec;
  unsigned int bandwith=10;
  assemble_band_preconditioner(sp,my_prec,bandwith,bem_problem_3d);
  // assemble_preconditioner(sp,my_prec,bem_problem_3d);
  bem_problem_3d.pcout<<"ILU"<<std::endl;
  SolverGMRES<TrilinosWrappers::MPI::Vector > monolithic_solver_ILU(bem_problem_3d.solver_control, bem_problem_3d.gmres_additional_data);
  TrilinosWrappers::PreconditionILU monolithic_prec_ILU;
  monolithic_prec_ILU.initialize(my_prec);
  monolithic_solver_ILU.solve(bem_problem_3d.monolithic_system_matrix, bem_problem_3d.monolithic_solution, bem_problem_3d.monolithic_rhs, monolithic_prec_ILU);
  bem_problem_3d.pcout << "Iterations needed to solve monolithic:         " << bem_problem_3d.solver_control.last_step() << std::endl;

  for (auto i : reference_monolithic_solution.locally_owned_elements())
    {
      double foo = std::abs(reference_monolithic_solution[i]-bem_problem_3d.monolithic_solution[i]);
      if (foo>tol)
        std::cout<<"ERROR, index i : "<<i<<" : "<<foo<<" , instead of : "<<0<<std::endl;
    }
  bem_problem_3d.monolithic_solution=0.;
  bem_problem_3d.pcout<<"JACOBI"<<std::endl;
  SolverGMRES<TrilinosWrappers::MPI::Vector > monolithic_solver_Jacobi(bem_problem_3d.solver_control, bem_problem_3d.gmres_additional_data);
  TrilinosWrappers::PreconditionJacobi monolithic_prec_Jacobi;
  monolithic_prec_Jacobi.initialize(my_prec);
  monolithic_solver_Jacobi.solve(bem_problem_3d.monolithic_system_matrix, bem_problem_3d.monolithic_solution, bem_problem_3d.monolithic_rhs, monolithic_prec_Jacobi);
  bem_problem_3d.pcout << "Iterations needed to solve monolithic:         " << bem_problem_3d.solver_control.last_step() << std::endl;

  for (auto i : reference_monolithic_solution.locally_owned_elements())
    {
      double foo = std::abs(reference_monolithic_solution[i]-bem_problem_3d.monolithic_solution[i]);
      if (foo>tol)
        std::cout<<"ERROR, index i : "<<i<<" : "<<foo<<" , instead of : "<<0<<std::endl;
    }
  bem_problem_3d.monolithic_solution=0.;
  bem_problem_3d.pcout<<"AMG"<<std::endl;
  SolverGMRES<TrilinosWrappers::MPI::Vector > monolithic_solver_AMG(bem_problem_3d.solver_control, bem_problem_3d.gmres_additional_data);
  TrilinosWrappers::PreconditionAMG monolithic_prec_AMG;
  // TrilinosWrappers::PreconditionAMG::AdditionalData::AdditionalData add_data_amg;//(false,
  // false,1,false,1e-4,std::vector<std::vector<bool> > (0),
  // 2,0,false,"Jacobi","Amesos-KLU");

  monolithic_prec_AMG.initialize(my_prec); //,add_data_amg);
  monolithic_solver_AMG.solve(bem_problem_3d.monolithic_system_matrix, bem_problem_3d.monolithic_solution, bem_problem_3d.monolithic_rhs, monolithic_prec_AMG);
  bem_problem_3d.pcout << "Iterations needed to solve monolithic:         " << bem_problem_3d.solver_control.last_step() << std::endl;

  for (auto i : reference_monolithic_solution.locally_owned_elements())
    {
      double foo = std::abs(reference_monolithic_solution[i]-bem_problem_3d.monolithic_solution[i]);
      if (foo>tol)
        std::cout<<"ERROR, index i : "<<i<<" : "<<foo<<" , instead of : "<<0<<std::endl;
    }

  bem_problem_3d.tria.set_manifold(0);

  return 0;
}
