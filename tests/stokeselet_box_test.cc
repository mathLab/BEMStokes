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
  unsigned int ncycles = 4;
  unsigned int max_degree = 1;
  std::cout<<"Test that the BEM is able to recover the Fundamental Solution"<<std::endl;
  // ParsedFunction<3,3> exact_solution_eig("Exact solution position",
  //         "x / (x*x + y*y + z*z)^0.5 ; y / (x*x + y*y + z*z)^0.5 ; z / (x*x + y*y + z*z)^0.5");
  unsigned int degree = 1;
  BEMProblem<3> bem_problem_3d;
  ParameterAcceptor::initialize(SOURCE_DIR "/parameters_test_alpha_box.prm","used.prm");//("foo.prm","foo1.prm");//SOURCE_DIR "/parameters_test_3d_boundary.prm"
  bem_problem_3d.convert_bool_parameters();
  bem_problem_3d.pcout<<"Minimum Test for the preconditioner with interior problem and the monolithic system"<<std::endl;
  bem_problem_3d.pcout<<"We consider the wall to have normal parallel to the i axis"<<std::endl;
  for (unsigned int i = 0 ; i<bem_problem_3d.wall_bool.size(); ++i)
    bem_problem_3d.wall_bool[i]=false;
  bem_problem_3d.reflect_kernel=false;
  bem_problem_3d.no_slip_kernel=false;
  bem_problem_3d.fe_stokes = bem_problem_3d.parsed_fe_stokes();
  bem_problem_3d.fe_map = bem_problem_3d.parsed_fe_mapping();


  Point<3> position_0,position_1,position_2,position_3,position_4,position_5;
  bem_problem_3d.wall_spans[0][0]=100;
  bem_problem_3d.wall_spans[0][2]=100;
  bem_problem_3d.wall_spans[1][0]=100;
  bem_problem_3d.wall_spans[1][2]=100;
  bem_problem_3d.wall_spans[2][0]=100;
  bem_problem_3d.wall_spans[2][1]=100;
  bem_problem_3d.wall_spans[3][0]=100;
  bem_problem_3d.wall_spans[3][1]=100;
  bem_problem_3d.wall_spans[4][1]=100;
  bem_problem_3d.wall_spans[4][2]=100;
  bem_problem_3d.wall_spans[5][1]=100;
  bem_problem_3d.wall_spans[5][2]=100;
  position_0[0]=0;
  position_0[1]=0;
  position_0[2]=0;
  position_1[0]=0;
  position_1[1]=-200;
  position_1[2]=0;
  position_2[0]=0;
  position_2[1]=-100;
  position_2[2]=100;
  position_3[0]=0;
  position_3[1]=-100;
  position_3[2]=-100;
  position_4[0]=100;
  position_4[1]=-100;
  position_4[2]=0;
  position_5[0]=-100;
  position_5[1]=-100;
  position_5[2]=0;


  bem_problem_3d.wall_types[0]="no_slip";
  bem_problem_3d.wall_types[1]="dirichlet";
  bem_problem_3d.wall_types[2]="dirichlet";
  bem_problem_3d.wall_types[3]="dirichlet";
  bem_problem_3d.wall_types[4]="dirichlet";
  bem_problem_3d.wall_types[5]="dirichlet";
  bem_problem_3d.use_internal_alpha=true;


  bem_problem_3d.create_box_bool=false;
  bem_problem_3d.wall_bool[0]=true;
  bem_problem_3d.wall_bool[1]=true;
  bem_problem_3d.wall_bool[2]=true;
  bem_problem_3d.wall_bool[3]=true;
  bem_problem_3d.wall_bool[4]=true;
  bem_problem_3d.wall_bool[5]=true;

  bem_problem_3d.refine_distance_from_center=4.;//15.;
  bem_problem_3d.wall_threshold=10.;//4.;
  bem_problem_3d.refinement_center[0]=0.;
  bem_problem_3d.refinement_center[1]=0.;
  bem_problem_3d.refinement_center[2]=0.;
  bem_problem_3d.extra_debug_info=true;
  // for(unsigned int i=0; i<first_index_box; ++i)
  //   {
  //     bem_problem_3d.pcout<<"Creating wall "<<i<<std::endl;
  //     bem_problem_3d.create_wall(bem_problem_3d.tria, bem_problem_3d.wall_types[i], bem_problem_3d.wall_positions[i], bem_problem_3d.wall_spans[i], i, false);
  //   }
  // bem_problem_3d.create_box(bem_problem_3d.tria, 0);
  // bem_problem_3d.refine_walls(bem_problem_3d.tria,bem_problem_3d.refine_distance_from_center,bem_problem_3d.wall_threshold,bem_problem_3d.refinement_center);

  std::cout<<bem_problem_3d.wall_positions.size()<<std::endl;
  Triangulation<3-1, 3> triangulation_wall, triangulation_helper;
  bem_problem_3d.tria.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_0, bem_problem_3d.wall_spans[0], false);
  bem_problem_3d.tria.copy_triangulation(triangulation_wall);
  triangulation_wall.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_1, bem_problem_3d.wall_spans[1], false);
  triangulation_helper.clear();
  triangulation_helper.copy_triangulation(bem_problem_3d.tria);
  bem_problem_3d.tria.clear();
  GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, bem_problem_3d.tria);
  triangulation_wall.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_2, bem_problem_3d.wall_spans[2], false);
  triangulation_helper.clear();
  triangulation_helper.copy_triangulation(bem_problem_3d.tria);
  bem_problem_3d.tria.clear();
  GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, bem_problem_3d.tria);
  triangulation_wall.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_3, bem_problem_3d.wall_spans[3], false);
  triangulation_helper.clear();
  triangulation_helper.copy_triangulation(bem_problem_3d.tria);
  bem_problem_3d.tria.clear();
  GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, bem_problem_3d.tria);
  triangulation_wall.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_4, bem_problem_3d.wall_spans[4], false);
  triangulation_helper.clear();
  triangulation_helper.copy_triangulation(bem_problem_3d.tria);
  bem_problem_3d.tria.clear();
  GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, bem_problem_3d.tria);
  triangulation_wall.clear();
  bem_problem_3d.create_coarse_wall(triangulation_wall, bem_problem_3d.wall_types[0], position_5, bem_problem_3d.wall_spans[5], false);
  triangulation_helper.clear();
  triangulation_helper.copy_triangulation(bem_problem_3d.tria);
  bem_problem_3d.tria.clear();
  GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, bem_problem_3d.tria);
  triangulation_wall.clear();

  typename Triangulation<3-1,3>::active_cell_iterator
  cell = bem_problem_3d.tria.begin_active(),
  endc = bem_problem_3d.tria.end();
  for (cell=bem_problem_3d.tria.begin_active(); cell != endc; ++cell)
    {
      if (std::abs(cell->center()[1])<=1e-5)
        cell->set_material_id(2);
      bem_problem_3d.pcout<<cell->center()[1]<<std::endl;
    }
  bem_problem_3d.tria.refine_global(1);

  bem_problem_3d.refine_walls(bem_problem_3d.tria,bem_problem_3d.refine_distance_from_center,bem_problem_3d.wall_threshold,bem_problem_3d.refinement_center);

  bem_problem_3d.remove_hanging_nodes_between_different_material_id(bem_problem_3d.tria);

  for (cell=bem_problem_3d.tria.begin_active(); cell != endc; ++cell)
    {
      cell->set_material_id(0);
    }

  std::string filename1="try.vtk";
  std::ofstream foo_ofs;
  foo_ofs.open(filename1, std::ofstream::out);
  GridOut go;
  go.write_vtk(bem_problem_3d.tria,foo_ofs);
// remove_hanging_nodes_between_different_material_id


  bem_problem_3d.reinit();
  bem_problem_3d.body_cpu_set.clear();
  bem_problem_3d.body_cpu_set.set_size(bem_problem_3d.dh_stokes.n_dofs());
  bem_problem_3d.body_cpu_set.add_range(0,bem_problem_3d.dh_stokes.n_dofs());
  VectorTools::get_position_vector(bem_problem_3d.map_dh,bem_problem_3d.euler_vec);
  bem_problem_3d.mappingeul = SP(new MappingQ<2,3>(degree));//SP(new MappingFEField<2, 3>(bem_problem_2d.map_dh,bem_problem_2d.euler_vec));
  bem_problem_3d.compute_center_of_mass_and_rigid_modes(0);
  bem_problem_3d.compute_normal_vector();
  bool correction_on_V = true;
  // bem_problem_3d.constraints.clear();
  // DoFTools::make_hanging_node_constraints (bem_problem_3d.dh_stokes,bem_problem_3d.constraints);
  // bem_problem_3d.constraints.close();
  bem_problem_3d.assemble_stokes_system(correction_on_V);

  // bem_problem_3d.V_matrix.print(std::cout);


  Point<3> source_point;
  source_point[0] = 0.5;
  source_point[1] = 15.;
  source_point[2] = 0.5;
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

  Vector<double> t1_diff(trace_1_vector_difference),
         t1(G_trace_1),
         t1_ex(G_trace_1_ex),
         normal(bem_problem_3d.normal_vector),
         t0(G_velocities);
  bem_problem_3d.pcout<<"Error L2 : "<<t1_diff.l2_norm()<<" , error Linfty : "<<t1_diff.linfty_norm()<<std::endl;

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  data_component_interpretation
  (3, DataComponentInterpretation::component_is_part_of_vector);
  DataOut<3-1, DoFHandler<3-1, 3> > dataout;

  dataout.attach_dof_handler(bem_problem_3d.dh_stokes);
  dataout.add_data_vector(t1_ex, std::vector<std::string > (3,"G_trace_1_ex"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  dataout.add_data_vector(t0, std::vector<std::string > (3,"G_velocities_0"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  dataout.add_data_vector(t1, std::vector<std::string > (3,"G_trace_1_0"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  dataout.add_data_vector(t1_diff, std::vector<std::string > (3,"G_trace_1_error"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  dataout.add_data_vector(normal, std::vector<std::string > (3,"normal_vector"), DataOut<3-1, DoFHandler<3-1, 3> >::type_dof_data, data_component_interpretation);
  dataout.build_patches(*bem_problem_3d.mappingeul,
                        degree,
                        DataOut<3-1, DoFHandler<3-1, 3> >::curved_inner_cells);

  std::string filename = ( "G_check_"+ Utilities::int_to_string(degree) +
                           ".vtu" );
  std::ofstream file(filename.c_str());

  dataout.write_vtu(file);

  // std::string file_name1;
  // file_name1 = "G_trace_1_" + Utilities::int_to_string(3) + "d.bin";
  // std::ofstream forces (file_name1.c_str());
  // t1.block_write(forces);
  // std::string file_name2;
  // file_name2 = "G_trace_0_" + Utilities::int_to_string(3) + "d.bin";
  // std::ofstream velocities (file_name2.c_str());
  // t0.block_write(velocities);
  // eh.output_table(std::cout,0);


  return 0;
}
