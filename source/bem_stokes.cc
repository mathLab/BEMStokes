
#include <deal.II/grid/grid_reordering.h>

#include <bem_stokes.h>
#include <operator.h>

#include "Teuchos_TimeMonitor.hpp"
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#ifdef BEM_STOKES_WITH_OCE
#include <deal.II/opencascade/boundary_lib.h>
#include <deal.II/opencascade/utilities.h>
#endif
using Teuchos::Time;
using Teuchos::TimeMonitor;
using Teuchos::RCP;

RCP<Time> RunTime = Teuchos::TimeMonitor::getNewTimer("Run Time");
RCP<Time> AssembleTime = Teuchos::TimeMonitor::getNewTimer("Assemble Time");
RCP<Time> LacSolveTime = Teuchos::TimeMonitor::getNewTimer("LAC Solve Time");
RCP<Time> ReinitTime = Teuchos::TimeMonitor::getNewTimer("BEM Reinitialisation Time");
RCP<Time> RefactTime = Teuchos::TimeMonitor::getNewTimer("DirectPreconditioner Refactorisation");


// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace BEMStokes
{
  using namespace dealii;
  using namespace deal2lkit;

  template<int my_dim>
  void impose_G_as_velocity(const BEMStokes::BEMProblem<my_dim> *bem, const Point<my_dim> &source, TrilinosWrappers::MPI::Vector &G_velocities)
  {
    std::vector<Point<my_dim> > support_points(bem->dh_stokes.n_dofs());
    DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (*(bem->mappingeul),
                                                            bem->map_dh, support_points);

    for (unsigned int i=0; i<bem->dh_stokes.n_dofs()/my_dim; ++i)
      {
        if (bem->this_cpu_set.is_element(i))
          {
            const Tensor<1,my_dim> R = support_points[i] - source;
            Tensor<2,my_dim> G = bem->stokes_kernel.value_tens(R) ;
            for (unsigned int jdim=0; jdim<my_dim; ++jdim)
              {
                G_velocities[i+bem->dh_stokes.n_dofs()/my_dim*jdim] = G[jdim][0];
              }
          }
      }
    G_velocities.compress(VectorOperation::insert);

  }

  template<int my_dim>
  void impose_G_as_trace_1(const Point<my_dim> &source, const BEMStokes::BEMProblem<my_dim> *bem,  TrilinosWrappers::MPI::Vector &normals, TrilinosWrappers::MPI::Vector &G_trace_1)
  {
    std::vector<Point<my_dim> > support_points(bem->dh_stokes.n_dofs());
    DoFTools::map_dofs_to_support_points<my_dim-1, my_dim>(*(bem->mappingeul),//StaticMappingQ1<my_dim-1, my_dim>::mapping
                                                           bem->map_dh, support_points);

    for (unsigned int i=0; i<bem->dh_stokes.n_dofs()/my_dim; ++i)
      {
        if (bem->this_cpu_set.is_element(i))
          {
            const Tensor<1,my_dim> R = support_points[i] - source;
            Point<my_dim> normal;
            for (unsigned int jdim=0; jdim<my_dim; ++jdim)
              normal[jdim] = - normals[i+bem->dh_stokes.n_dofs()/my_dim*jdim];
            Tensor<3,my_dim> W = bem->stokes_kernel.value_tens2(R) ;
            Tensor<2,my_dim> singular_ker = bem->compute_singular_kernel(normal, W) ;
            for (unsigned int jdim=0; jdim<my_dim; ++jdim)
              G_trace_1[i+bem->dh_stokes.n_dofs()/my_dim*jdim] = 1 * singular_ker[jdim][0];
          }
      }
    G_trace_1.compress(VectorOperation::insert);

  }
  void read_eig_vector(Vector<double> &input_vector, std::string filename)
  {
    std::ifstream infile (filename.c_str());
    std::string instring;
    types::global_dof_index i;
    i=0;
    while (infile.good() && i < input_vector.size())
      {
        instring.clear();
        getline ( infile, instring, '\n');
        input_vector[i] = double(atof(instring.c_str()));
        // if(input_vector[i]!=0)
        //   std::cout<<instring<<" "<<input_vector[i]<<" "<<double(atof(instring.c_str()))<<std::endl;
        i=i+1;
      }

  }
// @sect3{Single and double layer operator kernels}

// First, let us define a bit of the boundary integral equation machinery.

// @sect4{BEMProblem::BEMProblem and BEMProblem::read_parameters}

// The constructor initializes the various object in much the same way as
// done in the finite element programs such as step-4 or step-6. The only
// new ingredient here is the ParsedFunction object, which needs, at
// construction time, the specification of the number of components.
//
// For the exact solution the number of vector components is one, and no
// action is required since one is the default value for a ParsedFunction
// object. The wind, however, requires dim components to be
// specified. Notice that when declaring entries in a parameter file for the
// expression of the Functions::ParsedFunction, we need to specify the
// number of components explicitly, since the function
// Functions::ParsedFunction::declare_parameters is static, and has no
// knowledge of the number of components.
  template <>
  BEMProblem<2>::BEMProblem(MPI_Comm mpi_commy)
    :
    map_dh(tria),
    dh_stokes(tria),
    mappingeul(NULL),
    quadrature("Internal Quadrature"),
    quadrature_ext("External Quadrature"),
    parsed_fe_stokes("Finite Element Stokes","FESystem<1,2>[FE_Q<1,2>(1)^2]","u,u",2),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_fe_mapping("Finite Element Mapping","FESystem<1,2>[FE_Q<1,2>(1)^2]","u,u",2),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    preconditioner_bandwith(100),
    rotation_matrix(2,2),
    old_rotation_matrix(2,2),
    wall_bool(8, false),
    flip_normal_wall_bool(8, false),
    wall_types(8),
    wall_positions(8,Point<2>()),
    wall_spans(8,std::vector<double>(3)),
    initial_quaternion(4),
    mpi_communicator (mpi_commy),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout,
          (this_mpi_process
           == 0)),
    dpcout(std::cout)
  {}
  template <>
  BEMProblem<3>::BEMProblem(MPI_Comm mpi_commy)
    :
    map_dh(tria),
    dh_stokes(tria),
    mappingeul(NULL),
    quadrature("Internal Quadrature"),
    quadrature_ext("External Quadrature"),
    parsed_fe_stokes("Finite Element Stokes","FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",3),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    parsed_fe_mapping("Finite Element Mapping","FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",3),//,"FESystem<2,3>[FE_Q<2,3>(1)^3]","u,u,u",dim),
    preconditioner_bandwith(100),
    rotation_matrix(3,3),
    old_rotation_matrix(3,3),
    wall_bool(8, false),
    flip_normal_wall_bool(8, false),
    wall_types(8),
    wall_positions(8,Point<3>()),
    wall_spans(8,std::vector<double>(3)),
    initial_quaternion(4),
    mpi_communicator (mpi_commy),
    n_mpi_processes (Utilities::MPI::n_mpi_processes(mpi_communicator)),
    this_mpi_process (Utilities::MPI::this_mpi_process(mpi_communicator)),
    pcout(std::cout,
          (this_mpi_process
           == 0)),
    dpcout(std::cout,
           (this_mpi_process
            == 0))
  {
    monolithic_preconditioner_sparsity_pattern = NULL;
  }

  template <int dim>
  BEMProblem<dim>::~BEMProblem()
  {
    if (bandwith_preconditioner)
      delete monolithic_preconditioner_sparsity_pattern;
    else if (&monolithic_full_sparsity_pattern == monolithic_preconditioner_sparsity_pattern)
      {
        monolithic_preconditioner_sparsity_pattern = NULL;
      }
    else if (monolithic_preconditioner_sparsity_pattern == NULL)
      {  }
    else
      {
        AssertThrow(false, ExcNotImplemented());
      }
    // Reset all singular FEValues

    get_singular_fe_values(numbers::invalid_unsigned_int);
    if (tria.n_levels()>0)
      {
        auto tria_manifold = tria.get_manifold_ids();
        for (unsigned int i = 0; i<tria_manifold.size(); ++i)
          if (tria_manifold[i]!=numbers::invalid_manifold_id)
            tria.set_manifold(tria_manifold[i]);
      }
    // tria.set_manifold(0);
    // tria.set_manifold(1);
    // tria.set_manifold(2);
  }

  template <int dim>
  void BEMProblem<dim>::declare_parameters (ParameterHandler &prm)
  {
    add_parameter(prm, &num_convergence_cycle, "Number of convergence iterations","3", Patterns::Integer());

    add_parameter(prm, &spheres_distance, "Distance for the 2 sphere analysis","2.2", Patterns::Double());

    add_parameter(prm, &solve_with_torque, "Impose a torque on the flagellum","false", Patterns::Bool());

    add_parameter(prm, &spherical_head, "Impose a spherical head to the bacterium","false", Patterns::Bool());

    add_parameter(prm, &galerkin, "Galerkin Resolution Strategy","false", Patterns::Bool());
    add_parameter(prm, &n_frames, "Total number of frames","120",Patterns::Integer());
    add_parameter(prm, &delta_frame, "Delta between frames","1",Patterns::Integer());
    add_parameter(prm, &gmres_restart, "Gmres restart evert","100",Patterns::Integer());

    add_parameter(prm, &bool_rot, "Consider rigid rotations","true",Patterns::Bool());

    add_parameter(prm, &apply_iges, "Apply iges manifold to reference tria","false",Patterns::Bool());
    add_parameter(prm, &remove_tria_anisotropies, "Load and save refinement history in the code","false",Patterns::Bool());
    add_parameter(prm, &apply_iges_to_frame, "Apply iges manifold to frame tria","false",Patterns::Bool());


    add_parameter(prm, &bool_dipl, "Consider rigid displacement to move the swimmer","false",Patterns::Bool());
    add_parameter(prm, &bool_dipl_x, "Consider rigid displacement x to move the swimmer","false",Patterns::Bool());
    add_parameter(prm, &bool_dipl_y, "Consider rigid displacement y to move the swimmer","false",Patterns::Bool());
    add_parameter(prm, &bool_dipl_z, "Consider rigid displacement z to move the swimmer","false",Patterns::Bool());

    add_parameter(prm, &monolithic_bool, "Monolithic resolurion strategy","true",Patterns::Bool());
    add_parameter(prm, &gradual_wall_refinement, "Wall gradual wall refinement","false",Patterns::Bool());
    add_parameter(prm, &solve_directly, "Use a direct resolution strategy","true",Patterns::Bool());

    add_parameter(prm, &grid_type, "Grid","Real", Patterns::Selection("Real|ImposedForce|Cube|Convergence|ImposedVelocity"));

    add_parameter(prm, &singular_quadrature_type, "Singular quadrature kind","Mixed", Patterns::Selection("Mixed|Duffy|Telles"));

    add_parameter(prm, &force_pole, "Force Pole to be used","Origin", Patterns::Selection("Baricenter|Origin|Point"));

    if (dim == 2)
      add_parameter(prm, &force_arbitrary_point,"Force Pole Point Setting","1,0",
                    Patterns::List(Patterns::Double(),dim,dim));
    else if (dim == 3)
      add_parameter(prm, &force_arbitrary_point,"Force Pole Point Setting","1,0,0",
                    Patterns::List(Patterns::Double(),dim,dim));

    add_parameter(prm, &preconditioner_type, "Type of preconditioner to be used","AMG", Patterns::Selection("AMG|ILU|Jacobi|SOR|SSOR|Direct"));

    add_parameter(prm, &bandwith_preconditioner, "Use a bandwith preconditioner","false", Patterns::Bool());

    add_parameter(prm, &bandwith, "Bandwith for the preconditioner","100", Patterns::Integer());


    add_parameter(prm, &n_cycles, "Number of cycles", "4", Patterns::Integer());

    add_parameter(prm, &use_internal_alpha, "Use alpha for the internal problem", "false",
                  Patterns::Bool());

    add_parameter(prm, &input_grid_path, "Input path to grid", "../debug_grids/",
                  Patterns::Anything());

    add_parameter(prm, &input_iges_file_1, "Iges filename 1", "../iges_files/flagellum_rotated_",
                  Patterns::Anything());


    add_parameter(prm, &input_grid_base_name, "Input grid base name", "sphere_mesh_"+Utilities::int_to_string(dim)+"d_",
                  Patterns::Anything());
    add_parameter(prm, &input_grid_format, "Input grid format", "msh",
                  Patterns::Anything());

    add_parameter(prm, &res_strategy,"Time Integration","Forward",
                  Patterns::Selection("Forward|Heun"));

    add_parameter(prm, &build_sphere_in_deal, "Build the sphere inside the code", "false",
                  Patterns::Bool());
    add_parameter(prm, &internal_sphere_refinements, "Refinement applied to sphere built inside the code", "3",
                  Patterns::Integer());
    add_parameter(prm, &imposed_component, "Imposed Component for Non Real Simulation", "1",
                  Patterns::Integer());

    add_parameter(prm, &singular_quadrature_order, "Singular quadrature order", "5", Patterns::Integer());

    add_parameter(prm, &angle_for_imposed_stuff,"Angle to rotate the imposed stuff","0.",
                  Patterns::Double());

    if (dim == 3)
      {
        add_parameter(prm, &initial_quaternion[0],"Initial quaternion value q[0]","1",//,0,0,0",
                      Patterns::Double());//Patterns::List(Patterns::Double(),4,4));
        add_parameter(prm, &initial_quaternion[1],"Initial quaternion value q[1]","0",//,0,0,0",
                      Patterns::Double());//Patterns::List(Patterns::Double(),4,4));
        add_parameter(prm, &initial_quaternion[2],"Initial quaternion value q[2]","0",//,0,0,0",
                      Patterns::Double());//Patterns::List(Patterns::Double(),4,4));
        add_parameter(prm, &initial_quaternion[3],"Initial quaternion value q[3]","0",//,0,0,0",
                      Patterns::Double());//Patterns::List(Patterns::Double(),4,4));
        add_parameter(prm, &create_box_bool, "Create a bounding box","false",Patterns::Bool());

        add_parameter(prm, &rotation_correction, "Apply a correction for rigid rotations","false",Patterns::Bool());

        add_parameter(prm, &imposed_rotation_as_flagellum_shape, "Impose a rotation for flagellum shape using the first rotation mode","false",Patterns::Bool());

        add_parameter(prm, &imposed_traslation_as_flagellum_shape, "Impose a translation for flagellum shape using the first translation mode","false",Patterns::Bool());


        add_parameter(prm, &first_index_box, "First wall index to create the box","0",Patterns::Integer());


        add_parameter(prm, &cylinder_create_bool,"Cylindrical wall creation","false",
                      Patterns::Bool());

        add_parameter(prm, &cylinder_import_bool,"Cylindrical wall import","false",
                      Patterns::Bool());

        add_parameter(prm, &cylinder_manifold_bool,"Cylinder Apply Manifold descriptor","true",
                      Patterns::Bool());

        add_parameter(prm, &cylinder_flip_normal_bool,"Cylinder flip normal","false",
                      Patterns::Bool());

        add_parameter(prm, &cylinder_radius,"Cylinder Radius","1.",
                      Patterns::Double());
        add_parameter(prm, &time_step,"Time interval between frames","0.1",
                      Patterns::Double());

        add_parameter(prm, &cylinder_heigth,"Cylinder Heigth","2.",
                      Patterns::Double());



        add_parameter(prm, &cylinder_direction,"Cylinder axis orientation","0.,0.,1.",
                      Patterns::List(Patterns::Double(),dim,dim));

        add_parameter(prm, &cylinder_point_on_axis,"Cylinder point on axis","0.,0.,0.",
                      Patterns::List(Patterns::Double(),dim,dim));

        add_parameter(prm, &refinement_center,"Refinement Center For Wall Refinement","0.,0.,0.",
                      Patterns::List(Patterns::Double(),dim,dim));

        add_parameter(prm, &cylinder_wall_type,"Cylinder Wall type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");

        add_parameter(prm, &cylinder_mesh_filename, "Cylinder mesh name", "cylinder",
                      Patterns::Anything());

        add_parameter(prm, &wall_bool_0,"Wall 0 bool","false",
                      Patterns::Bool(),"Bool set to create wall 0.");
        add_parameter(prm, &wall_bool_1,"Wall 1 bool","false",
                      Patterns::Bool(),"Bool set to create wall 1.");
        add_parameter(prm, &wall_bool_2,"Wall 2 bool","false",
                      Patterns::Bool(),"Bool set to create wall 2.");
        add_parameter(prm, &wall_bool_3,"Wall 3 bool","false",
                      Patterns::Bool(),"Bool set to create wall 3.");
        add_parameter(prm, &wall_bool_4,"Wall 4 bool","false",
                      Patterns::Bool(),"Bool set to create wall 4.");
        add_parameter(prm, &wall_bool_5,"Wall 5 bool","false",
                      Patterns::Bool(),"Bool set to create wall 5.");
        add_parameter(prm, &wall_bool_6,"Wall 6 bool","false",
                      Patterns::Bool(),"Bool set to create wall 6.");
        add_parameter(prm, &wall_bool_7,"Wall 7 bool","false",
                      Patterns::Bool(),"Bool set to create wall 7.");

        add_parameter(prm, &flip_normal_wall_bool_0,"Flip normal Wall 0","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_1,"Flip normal Wall 1","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_2,"Flip normal Wall 2","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_3,"Flip normal Wall 3","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_4,"Flip normal Wall 4","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_5,"Flip normal Wall 5","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_6,"Flip normal Wall 6","false",Patterns::Bool());
        add_parameter(prm, &flip_normal_wall_bool_7,"Flip normal Wall 7","false",Patterns::Bool());
        add_parameter(prm, &reflect_kernel,"Reflect the kernel","false",//,0,0,0",
                      Patterns::Bool());//Patterns::List(Patterns::Double(),4,4));

        add_parameter(prm, &no_slip_kernel,"Use no slip kernel","false",//,0,0,0",
                      Patterns::Bool());//Patterns::List(Patterns::Double(),4,4));


        add_parameter(prm, &(wall_spans[0]),"Wall 0 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 0. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[1]),"Wall 1 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 1. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[2]),"Wall 2 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 2. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[3]),"Wall 3 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 3. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[4]),"Wall 4 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 4. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[5]),"Wall 5 spans","10,0,10",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 5. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[6]),"Wall 6 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 6. If -1 we intend infinite.");
        add_parameter(prm, &(wall_spans[7]),"Wall 7 spans","1,1,-1",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the spans of the wall 7. If -1 we intend infinite.");

        add_parameter(prm, &wall_positions[0],"Wall center position wall 0","0,5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[1],"Wall center position wall 1","0,-5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[2],"Wall center position wall 2","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[3],"Wall center position wall 3","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[4],"Wall center position wall 4","0,5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[5],"Wall center position wall 5","0,-5,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[6],"Wall center position wall 6","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");
        add_parameter(prm, &wall_positions[7],"Wall center position wall 7","0,10,0",
                      Patterns::List(Patterns::Double(),dim,dim),"List of the positions of all the centers of the walls.");


        add_parameter(prm, &wall_types[0],"Wall 0 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[1],"Wall 1 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[2],"Wall 2 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[3],"Wall 3 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[4],"Wall 4 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[5],"Wall 5 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[6],"Wall 6 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");
        add_parameter(prm, &wall_types[7],"Wall 7 type","no_slip",
                      Patterns::Selection("no_slip|free_surface|do_nothing|dirichlet|neumann"),"List of walls, only no slip, free surface or do nothing.");

        add_parameter(prm, &wall_threshold,"Wall Refinement Threshold","1.",
                      Patterns::Double(),"Minimum diameter you want for the wall at its center.");

        add_parameter(prm, &refine_distance_from_center,"Distance for wall refinement","2.",
                      Patterns::Double(),"Minimum distance from center you want to refine on the wall.");

        add_parameter(prm, &use_flagellum_handler, "Internal flagellum generation and handling", "false", Patterns::Bool());
      }
    add_parameter(prm, &use_previous_state,"Use state from previous frame","false",//,0,0,0",
                  Patterns::Bool());//Patterns::List(Patterns::Double(),4,4));


    add_parameter(prm, &assemble_scaling,"Scaling for monolithic assembling","1.",
                  Patterns::Double());

    add_parameter(prm, &extra_debug_info, "Print extra debug information", "true", Patterns::Bool());

    // In the solver section, we set all SolverControl parameters. The object
    // will then be fed to the GMRES solver in the solve_system() function.
    prm.enter_subsection("Solver");
    SolverControl::declare_parameters(prm);
    prm.leave_subsection();

    add_parameter(prm, &n_subdivisions, "Number of subdivisions", "0",
                  Patterns::Integer(0), "Set to zero for automatic");
  }


// After declaring all these parameters to the ParameterHandler object,
// let's read an input file that will give the parameters their values. We
// then proceed to extract these values from the ParameterHandler object:
//
  template <int dim>
  void BEMProblem<dim>::parse_parameters (ParameterHandler &prm)
  {

    prm.enter_subsection("Solver");
    solver_control.parse_parameters(prm);
    prm.leave_subsection();

    ParameterAcceptor::parse_parameters(prm);

    dpcout.set_condition(extra_debug_info && this_mpi_process == 0);
  }

  template <int dim>
  void BEMProblem<dim>::read_input_mesh_file(unsigned int frame, Triangulation<dim-1,dim> &triangulation)
  {
    std::ifstream in;
    dpcout<<input_grid_path+input_grid_base_name + Utilities::int_to_string(frame)+"." + input_grid_format<<std::endl;
    in.open (input_grid_path+input_grid_base_name + Utilities::int_to_string(frame)+"." + input_grid_format);
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation);
    if (input_grid_format=="vtk")
      gi.read_vtk (in);
    else if (input_grid_format=="msh")
      gi.read_msh (in);
    else if (input_grid_format=="inp")
      gi.read_ucd (in, true);
    else
      Assert (false, ExcNotImplemented());

    typename Triangulation<dim-1,dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (cell=triangulation.begin_active(); cell != endc; ++cell)
      {
        cell->set_material_id(0);
      }
    // triangulation.set_all
  }

  template <>
  void BEMProblem<2>::apply_flagellum_iges(Triangulation<1,2> &triangulation, std::string cad_file_name)
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template <>
  void BEMProblem<3>::apply_flagellum_iges(Triangulation<2,3> &triangulation, std::string cad_file_name)
  {
    const unsigned int dim = 3;
    double tolerance = 5e-3;
    double scale_factor = 1e-3;
#ifdef BEM_STOKES_WITH_OCE
    TopoDS_Shape flagellum_surface = OpenCASCADE::read_IGES(cad_file_name, scale_factor);
    // const double tolerance = OpenCASCADE::get_shape_tolerance(flagellum_surface) * 5;
    Point<3> CP = OpenCASCADE::closest_point(flagellum_surface, Point<3>(0.0,0.0,0.0));
    pcout<<"Origin projection on Flagellum is "<<CP<<endl;
    // std::vector<TopoDS_Compound>  compounds;
    // std::vector<TopoDS_CompSolid> compsolids;
    // std::vector<TopoDS_Solid>     solids;
    // std::vector<TopoDS_Shell>     shells;
    // std::vector<TopoDS_Wire>      wires;
    // OpenCASCADE::extract_compound_shapes(flagellum_surface,
    //                                      compounds,
    //                                      compsolids,
    //                                      solids,
    //                                      shells,
    //                                      wires);
    //
    typename Triangulation<dim-1,dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();
    for (cell=triangulation.begin_active(); cell != endc; ++cell)
      {
        if (cell->center()[0]>0)
          {
            cell->set_all_manifold_ids(2);
          }
        else
          {
            cell->set_all_manifold_ids(1);
          }
      }

    flagellum_manifold = std::shared_ptr<Manifold<dim-1,dim> >
                         (dynamic_cast<Manifold<dim-1,dim> * > (new OpenCASCADE::NormalToMeshProjectionBoundary<2,3> (flagellum_surface, tolerance)));
    triangulation.set_manifold(2,*flagellum_manifold);
#else
    AssertThrow(false, ExcMessage("You need to define BEM_STOKES_WITH_OCE with cmake to use iges files"));
#endif



  }


// A boundary element method triangulation is basically the same as a
// (dim-1) dimensional triangulation, with the difference that the vertices
// belong to a (dim) dimensional space.
//
// Some of the mesh formats supported in deal.II use by default three
// dimensional points to describe meshes. These are the formats which are
// compatible with the boundary element method capabilities of deal.II. In
// particular we can use either UCD or GMSH formats. In both cases, we have
// to be particularly careful with the orientation of the mesh, because,
// unlike in the standard finite element case, no reordering or
// compatibility check is performed here.  All meshes are considered as
// oriented, because they are embedded in a higher dimensional space. (See
// the documentation of the GridIn and of the Triangulation for further
// details on orientation of cells in a triangulation.) In our case, the
// normals to the mesh are external to both the circle in 2d or the sphere
// in 3d.
//
// The other detail that is required for appropriate refinement of the
// boundary element mesh, is an accurate description of the manifold that
// the mesh is approximating. We already saw this several times for the
// boundary of standard finite element meshes (for example in step-5 and
// step-6), and here the principle and usage is the same, except that the
// HyperBallBoundary class takes an additional template parameter that
// specifies the embedding space dimension. The function object still has to
// be static to live at least as long as the triangulation object to which
// it is attached.

  template <int dim>
  void BEMProblem<dim>::read_domain()
  {
    if (use_flagellum_handler == false)
      {
        switch (dim)
          {
          case 2:
          {
            dpcout << "We take " << input_grid_path+input_grid_base_name + "0."+input_grid_format+", as reference grid" << std::endl;
            read_input_mesh_file(0, tria);

            break;
          }
          case 3:
          {
            dpcout << "We take " << input_grid_path+input_grid_base_name + "0."+input_grid_format+", as reference grid" << std::endl;
            read_input_mesh_file(0, tria);
            std::cout<<tria.n_active_cells()<<std::endl;
            break;
          }
          default:
            Assert (false, ExcNotImplemented());
          }
        tria_unrefined.copy_triangulation(tria);
        Point<dim> pippo_1, pippo_2, pippo;
        pippo_2[0]=spheres_distance;

        manifold = std::shared_ptr<SphericalManifold<dim-1, dim> > (new SphericalManifold<dim-1, dim> (pippo));
        manifold_1 = std::shared_ptr<SphericalManifold<dim-1, dim> > (new SphericalManifold<dim-1, dim> (pippo_1));
        manifold_2 = std::shared_ptr<SphericalManifold<dim-1, dim> > (new SphericalManifold<dim-1, dim> (pippo_2));
        if (apply_iges)
          {
            std::string reference_flagellum_filename;
            reference_flagellum_filename = input_iges_file_1 + Utilities::int_to_string(0) + ".iges";
            apply_flagellum_iges(tria, reference_flagellum_filename);

            // typename Triangulation<dim-1,dim>::active_cell_iterator
            // cell = tria.begin_active(),
            // endc = tria.end();
            // unsigned int k=0;
            // for(cell=tria.begin_active(); cell != endc; ++cell)
            // {
            //     cell->set_refine_flag();
            //     k=k+1;
            //     if(k>10)
            //       break;
            // }
            std::ofstream history("mesh.history");
            tria.save_refine_flags(history);
            tria.prepare_coarsening_and_refinement();
            tria.execute_coarsening_and_refinement ();

          }
        if (remove_tria_anisotropies)
          {
            GridTools::remove_anisotropy(tria);
          }
        // tria.refine_global();

        // tria.set_manifold(2);
        if (grid_type == "ImposedForce" || grid_type == "Convergence" || grid_type == "ImposedVelocity")
          {
            if (build_sphere_in_deal == true)
              {
                tria.clear();
                Triangulation<dim-1, dim> tria_1, tria_2;
                pippo[0]=0.;
                bool two_spheres=false;
                if (two_spheres)
                  {
                    GridGenerator::hyper_sphere (tria_1,pippo_1,1. );
                    tria_1.set_all_manifold_ids(1);
                    // tria_1.set_manifold(1, *manifold_1);
                    GridGenerator::hyper_sphere (tria_2,pippo_2,1. );
                    tria_2.set_all_manifold_ids(2);
                    // tria_2.set_manifold(2, *manifold_2);
                    // tria.set_manifold(0);//, *manifold);
                    GridGenerator::merge_triangulations(tria_1,tria_2,tria);
                    typename Triangulation<dim-1,dim>::active_cell_iterator
                    cell = tria.begin_active(),
                    endc = tria.end();
                    for (cell=tria.begin_active(); cell != endc; ++cell)
                      {
                        if (cell->center()[0]>1.)
                          cell->set_all_manifold_ids(2);
                        else
                          cell->set_all_manifold_ids(1);
                      }

                    tria.set_manifold(1, *manifold_1);
                    tria.set_manifold(2, *manifold_2);
                  }
                else
                  {
                    GridGenerator::hyper_sphere (tria,pippo,1. );
                    tria.set_all_manifold_ids(0);
                    tria.set_manifold(0, *manifold);


                  }
                pcout<<"built and set the Spherical Manifold"<<std::endl;
                tria.refine_global(internal_sphere_refinements);
                // tria.flip_all_direction_flags();
                tria.flip_all_direction_flags();
                std::string filename = "tria.inp";
                std::ofstream wall_ofs;
                wall_ofs.open(filename, std::ofstream::out);
                GridOut go;
                go.write_ucd(tria,wall_ofs);
              }
            else
              {
                tria.set_all_manifold_ids(0);
                tria.set_manifold(0, *manifold);
                // tria.refine_global(1)mom;
                std::string filename = "tria.inp";
                std::ofstream wall_ofs;
                wall_ofs.open(filename, std::ofstream::out);
                GridOut go;
                go.write_ucd(tria,wall_ofs);


                // pcout<<"set the Spherical Manifold"<<std::endl;
              }
            // just to write the refined grid once.
            // tria.refine_global(1);
            // std::string filename = "sphere_refined.inp";
            //
            // if(this_mpi_process == 0)
            // {
            //   std::ofstream wall_ofs;
            //   wall_ofs.open(filename, std::ofstream::out);
            //   GridOut go;
            //   go.write_ucd(tria,wall_ofs);
            // }
            // MPI_Barrier(mpi_communicator);
            // tria.clear();
            // std::ifstream in;
            // in.open (filename);
            // GridIn<dim-1, dim> gi;
            // gi.attach_triangulation (tria);
            // gi.read_ucd (in, true);
            //
          }
      }
    else
      {
        flagellum_handler.create_initial_flagellum_triangulation(tria);
      }

    pcout<<"We have a tria of "<<tria.n_active_cells()<<" cells."<<std::endl;


    pcout << "Computing the euler vector for the reference grid" << std::endl;
    // pcout<<wall_types.size()<<" "<<wall_positions.size()<<" "<<wall_spans.size()<<std::endl;
    // pcout<<wall_types[0]<<" "<<wall_positions[0]<<" "<<wall_spans[0][0]<<" "<<wall_spans[0][1]<<" "<<wall_spans[0][2]<<" "<<std::endl;
    // std::cout<<create_box_bool<<std::endl;
    if (!create_box_bool)
      {
        pcout<<"Creating the box"<<std::endl;
        for (unsigned int i=0; i<wall_types.size(); ++i)
          if (wall_bool[i])
            create_wall(tria, wall_types[i], wall_positions[i], wall_spans[i], i, flip_normal_wall_bool[i]);
      }
    else
      {
        for (unsigned int i=0; i<first_index_box; ++i)
          if (wall_bool[i])
            {
              pcout<<"Creating wall "<<i<<std::endl;
              create_wall(tria, wall_types[i], wall_positions[i], wall_spans[i], i, flip_normal_wall_bool[i]);
            }
        for (unsigned int i=first_index_box+6; i<wall_types.size(); ++i)
          if (wall_bool[i])
            create_wall(tria, wall_types[i], wall_positions[i], wall_spans[i], i, flip_normal_wall_bool[i]);
        create_box(tria, first_index_box);
      }
    if (cylinder_create_bool)
      {
        pcout<<"Creating the Cylinder"<<std::endl;
        Triangulation<dim-1,dim> triangulation_cyl;
        create_cylindrical_wall(triangulation_cyl, cylinder_wall_type, cylinder_direction, cylinder_point_on_axis,
                                cylinder_radius,cylinder_heigth, cylinder_manifold_bool);
        // triangulation_cyl.set_manifold(99);
        add_cylinder_to_tria(tria,cylinder_manifold_bool);
      }
    else if (cylinder_import_bool)
      {
        pcout<<"Importing the Cylinder"<<std::endl;
        std::string filename = cylinder_mesh_filename+"."+input_grid_format;//input_grid_path+
        Triangulation<dim-1,dim> triangulation_cyl;
        import_cylinder(triangulation_cyl, filename, cylinder_wall_type, cylinder_direction, cylinder_point_on_axis,
                        cylinder_manifold_bool, cylinder_flip_normal_bool);
        // triangulation_cyl.set_manifold(99);
        add_cylinder_to_tria(tria, cylinder_manifold_bool);
      }
    refine_walls(tria,refine_distance_from_center,wall_threshold,refinement_center,gradual_wall_refinement);
    // pcout<<refine_distance_from_center<<" "<<wall_threshold<<" "<<refinement_center<<std::endl;
    remove_hanging_nodes_between_different_material_id(tria);

    std::string reference_filename = "reference_tria.bin";
    std::ofstream out;
    out.open (reference_filename);
    boost::archive::binary_oarchive oa(out);
    tria.save(oa, 0);
    GridTools::partition_triangulation(n_mpi_processes, tria);
    if (this_mpi_process == 0)
      {
        std::ostringstream filename;
        filename << "mesh.vtu";
        std::ofstream output (filename.str().c_str());
        FE_Q<dim-1, dim> fe_dummy(1);
        DoFHandler<dim-1,dim> dof_handler(tria);
        dof_handler.distribute_dofs (fe_dummy);
        DataOut<dim-1,DoFHandler<dim-1, dim>> data_out;
        data_out.attach_dof_handler (dof_handler);
        std::vector<unsigned int> partition_int (tria.n_active_cells());
        GridTools::get_subdomain_association (tria, partition_int);
        const Vector<double> partitioning(partition_int.begin(),
                                          partition_int.end());

        data_out.add_data_vector (partitioning, "partitioning",DataOut<dim-1, DoFHandler<dim-1, dim> >::type_cell_data);
        data_out.build_patches ();
        data_out.write_vtu (output);
      }
    std::string filename = "tria.vtu";
    std::ofstream wall_ofs;
    wall_ofs.open(filename, std::ofstream::out);
    GridOut go;
    go.write_vtu(tria,wall_ofs);
  }

  template<int dim>
  void BEMProblem<dim>::apply_rotation_along_axis(Point<dim> &out, const Point<dim> &in, const Point<dim> &axis, const double angle)
  {
    FullMatrix<double> R(3);

    R.set(0,0,cos(angle)+axis[0]*axis[0]*(1-cos(angle)));
    R.set(1,1,cos(angle)+axis[1]*axis[1]*(1-cos(angle)));
    R.set(2,2,cos(angle)+axis[2]*axis[2]*(1-cos(angle)));

    R.set(0,1,axis[0]*axis[1]*(1-cos(angle))-axis[2]*sin(angle));
    R.set(1,0,axis[0]*axis[1]*(1-cos(angle))+axis[2]*sin(angle));

    R.set(0,2,axis[0]*axis[2]*(1-cos(angle))+axis[1]*sin(angle));
    R.set(2,0,axis[0]*axis[2]*(1-cos(angle))-axis[1]*sin(angle));

    R.set(1,2,axis[1]*axis[2]*(1-cos(angle))-axis[0]*sin(angle));
    R.set(2,1,axis[1]*axis[2]*(1-cos(angle))+axis[0]*sin(angle));

    Vector<double> dummy_1(3);
    Vector<double> dummy_2(3);
    for (unsigned int i=0; i<3; ++i)
      {
        dummy_1[i]=in[i];
      }
    R.vmult(dummy_2, dummy_1);
    for (unsigned int i=0; i<3; ++i)
      {
        out[i]=dummy_2[i];
      }
    // R.print_formatted(std::cout);
  }


  template<>
  void BEMProblem<2>::create_cylindrical_wall(Triangulation<1, 2> &,
                                              const std::string &, const Point<2> &, const Point<2> &,
                                              const double , const double , const bool )
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template<int dim>
  void BEMProblem<dim>::create_cylindrical_wall(Triangulation<dim-1, dim> &triangulation_wall,
                                                const std::string &wall_type, const Point<dim> &direction, const Point<dim> &point_on_axis,
                                                const double radius, const double heigth, const bool apply_manifold)
  {

    Point<dim> rand_point,orthogonal,dummy,unit_direction;
    for (unsigned int i=0; i<dim; ++i)
      rand_point[i] = (double)rand();

    double helper = rand_point * direction;
    dummy = direction;
    unit_direction = direction;
    unit_direction/=unit_direction.norm();
    dummy *= helper;
    orthogonal = rand_point;
    orthogonal -= dummy;
    orthogonal*=radius/orthogonal.norm();

    std::vector<Point<dim> > cylinder_points(8);

    apply_rotation_along_axis(dummy, orthogonal, unit_direction, 0.);
    cylinder_points[0][0]=point_on_axis[0]-direction[0]*heigth/2+dummy[0];
    cylinder_points[0][1]=point_on_axis[1]-direction[1]*heigth/2+dummy[1];
    cylinder_points[0][2]=point_on_axis[2]-direction[2]*heigth/2+dummy[2];
    cylinder_points[4][0]=point_on_axis[0]+direction[0]*heigth/2+dummy[0];
    cylinder_points[4][1]=point_on_axis[1]+direction[1]*heigth/2+dummy[1];
    cylinder_points[4][2]=point_on_axis[2]+direction[2]*heigth/2+dummy[2];

    apply_rotation_along_axis(dummy, orthogonal, unit_direction, numbers::PI/2);
    cylinder_points[1][0]=point_on_axis[0]-direction[0]*heigth/2+dummy[0];
    cylinder_points[1][1]=point_on_axis[1]-direction[1]*heigth/2+dummy[1];
    cylinder_points[1][2]=point_on_axis[2]-direction[2]*heigth/2+dummy[2];
    cylinder_points[5][0]=point_on_axis[0]+direction[0]*heigth/2+dummy[0];
    cylinder_points[5][1]=point_on_axis[1]+direction[1]*heigth/2+dummy[1];
    cylinder_points[5][2]=point_on_axis[2]+direction[2]*heigth/2+dummy[2];

    apply_rotation_along_axis(dummy, orthogonal, unit_direction, numbers::PI);
    cylinder_points[2][0]=point_on_axis[0]-direction[0]*heigth/2+dummy[0];
    cylinder_points[2][1]=point_on_axis[1]-direction[1]*heigth/2+dummy[1];
    cylinder_points[2][2]=point_on_axis[2]-direction[2]*heigth/2+dummy[2];
    cylinder_points[6][0]=point_on_axis[0]+direction[0]*heigth/2+dummy[0];
    cylinder_points[6][1]=point_on_axis[1]+direction[1]*heigth/2+dummy[1];
    cylinder_points[6][2]=point_on_axis[2]+direction[2]*heigth/2+dummy[2];

    apply_rotation_along_axis(dummy, orthogonal, unit_direction, numbers::PI*3./2.);
    cylinder_points[3][0]=point_on_axis[0]-direction[0]*heigth/2+dummy[0];
    cylinder_points[3][1]=point_on_axis[1]-direction[1]*heigth/2+dummy[1];
    cylinder_points[3][2]=point_on_axis[2]-direction[2]*heigth/2+dummy[2];
    cylinder_points[7][0]=point_on_axis[0]+direction[0]*heigth/2+dummy[0];
    cylinder_points[7][1]=point_on_axis[1]+direction[1]*heigth/2+dummy[1];
    cylinder_points[7][2]=point_on_axis[2]+direction[2]*heigth/2+dummy[2];

    for (unsigned int i=0; i<cylinder_points.size(); ++i)
      pcout<<i<<" : "<<cylinder_points[i]<<std::endl;
    std::vector<CellData<dim-1> > cells(6);
    if (cylinder_flip_normal_bool)
      {
        cells[0].vertices[0]=0;
        cells[0].vertices[1]=3;
        cells[0].vertices[2]=2;
        cells[0].vertices[3]=1;
        cells[1].vertices[0]=4;
        cells[1].vertices[1]=5;
        cells[1].vertices[2]=6;
        cells[1].vertices[3]=7;
        cells[2].vertices[0]=1;
        cells[2].vertices[1]=2;
        cells[2].vertices[2]=6;
        cells[2].vertices[3]=5;//lateral
        cells[3].vertices[0]=6;
        cells[3].vertices[1]=2;
        cells[3].vertices[2]=3;
        cells[3].vertices[3]=7;//lateral
        cells[4].vertices[0]=7;
        cells[4].vertices[1]=3;
        cells[4].vertices[2]=0;
        cells[4].vertices[3]=4;//lateral
        cells[5].vertices[0]=4;
        cells[5].vertices[1]=0;
        cells[5].vertices[2]=1;
        cells[5].vertices[3]=5;//lateral

      }
    else
      {
        cells[0].vertices[0]=0;
        cells[0].vertices[1]=1;
        cells[0].vertices[2]=2;
        cells[0].vertices[3]=3;
        cells[1].vertices[0]=4;
        cells[1].vertices[1]=7;
        cells[1].vertices[2]=6;
        cells[1].vertices[3]=5;
        cells[2].vertices[0]=1;
        cells[2].vertices[1]=5;
        cells[2].vertices[2]=6;
        cells[2].vertices[3]=2;//lateral
        cells[3].vertices[0]=6;
        cells[3].vertices[1]=7;
        cells[3].vertices[2]=3;
        cells[3].vertices[3]=2;//lateral
        cells[4].vertices[0]=7;
        cells[4].vertices[1]=4;
        cells[4].vertices[2]=0;
        cells[4].vertices[3]=3;//lateral
        cells[5].vertices[0]=4;
        cells[5].vertices[1]=5;
        cells[5].vertices[2]=1;
        cells[5].vertices[3]=0;//lateral
      }
    cells[0].material_id = 4.;
    if (wall_type == "no_slip")
      {
        cells[2].material_id = 2;
        cells[3].material_id = 2;
        cells[4].material_id = 2;
        cells[5].material_id = 2;
      }
    else if (wall_type == "free_surface")
      {
        cells[2].material_id = 3;
        cells[3].material_id = 3;
        cells[4].material_id = 3;
        cells[5].material_id = 3;

      }
    if (apply_manifold)
      {
        cells[2].manifold_id = 99;
        cells[3].manifold_id = 99;
        cells[4].manifold_id = 99;
        cells[5].manifold_id = 99;

      }

    cells[1].material_id = 4.;

    SubCellData subcelldata;

    pcout<<"creating cylinder from scratch"<<std::endl;
    const std::vector<Point<dim> > dummy_vertices(cylinder_points);
    const std::vector<CellData<dim-1> > dummy_cells(cells);
    triangulation_wall.clear();
    GridTools::delete_unused_vertices (cylinder_points, cells, subcelldata);
    GridReordering<dim-1,dim>::reorder_cells (cells);
    triangulation_wall.create_triangulation_compatibility(cylinder_points, cells, subcelldata );
    pcout<<"created cylinder from scratch"<<std::endl;

    if (apply_manifold)
      {
        typename Triangulation<dim-1,dim>::active_cell_iterator
        cell = triangulation_wall.begin_active(),
        endc = triangulation_wall.end();
        for (cell=triangulation_wall.begin_active(); cell != endc; ++cell)
          {
            if (cell->material_id()!=4)
              {
                cell->set_all_manifold_ids(99);
              }
          }
        cylinder_manifold = std::shared_ptr<CylindricalManifold<dim-1, dim> > (new CylindricalManifold<dim-1, dim> (direction,point_on_axis));
        triangulation_wall.set_manifold(99, *cylinder_manifold);
      }
    GridTools::remove_anisotropy(triangulation_wall);
    triangulation_wall.refine_global();
    if (this_mpi_process == 0)
      {
        std::string filename = "cylinder.inp";
        std::ofstream wall_ofs;
        wall_ofs.open(filename, std::ofstream::out);
        GridOut go;
        go.write_ucd(triangulation_wall,wall_ofs);
      }
    MPI_Barrier(mpi_communicator);
  }

  template<>
  void BEMProblem<2>::import_cylinder(Triangulation<2-1, 2> &triangulation_wall, const std::string &filename,
                                      const std::string &wall_type, const Point<2> &direction, const Point<2> &point_on_axis,
                                      const bool apply_manifold, const bool flip_all)
  {
    Assert (false, ExcNotImplemented());
  }


  template<int dim>
  void BEMProblem<dim>::import_cylinder(Triangulation<dim-1, dim> &triangulation_wall, const std::string &filename,
                                        const std::string &wall_type, const Point<dim> &direction, const Point<dim> &point_on_axis,
                                        const bool apply_manifold, const bool flip_all)
  {
    pcout<<"reading the cylinder from "<<filename<<std::endl;
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation_wall);
    std::ifstream in;
    in.open (filename);
    if (input_grid_format=="vtk")
      gi.read_vtk (in);
    else if (input_grid_format=="msh")
      gi.read_msh (in);
    else if (input_grid_format=="inp")
      gi.read_ucd (in, true);
    else
      Assert (false, ExcNotImplemented());
    if (flip_all)
      {
        triangulation_wall.flip_all_direction_flags();
        pcout<<"flipping normals"<<std::endl;
      }
    GridTools::remove_anisotropy(triangulation_wall);
    pcout<<"read the cylinder from "<<filename<<std::endl;
    Point<dim> direction_versor(direction);
    direction_versor /= direction_versor.norm();


    FE_Q<dim-1,dim> fe_dummy(1);
    QGauss<dim-1> quadrature_dummy(1);
    FEValues<dim-1,dim> fev_dummy(StaticMappingQ1<dim-1, dim>::mapping, fe_dummy, quadrature_dummy,
                                  update_cell_normal_vectors);

    typename Triangulation<dim-1,dim>::active_cell_iterator
    cell = triangulation_wall.begin_active(),
    endc = triangulation_wall.end();
    pcout<<"setting material_id and manifold_id on imported cylinder"<<std::endl;

    for (cell=triangulation_wall.begin_active(); cell != endc; ++cell)
      {
        fev_dummy.reinit(cell);
        const std::vector<Tensor<1, dim> > &normals  = fev_dummy.get_normal_vectors();
        double helper = std::abs(std::abs(normals[0] * direction_versor) - 1);
        // pcout<<normals.size()<<" : "<<normals[0]<<" : "<<direction_versor<<std::endl;
        if (helper > 1e-1)
          {
            if (wall_type == "no_slip")
              {
                cell->set_material_id(2);
              }
            else if (wall_type == "free_surface")
              {
                cell->set_material_id(3);
              }
            if (apply_manifold)
              cell->set_all_manifold_ids(99);
            // pcout<<"found lateral surface : helper = "<<helper<<" : "<<std::abs(normals[0] * direction_versor)<<std::endl;
          }
        else
          {
            if (wall_type == "no_slip")
              cell->set_material_id(4);
            else if (wall_type == "dirichlet")
              cell->set_material_id(5);
            // pcout<<"found top - bottom surface : helper = "<<helper<<" : "<<std::abs(normals[0] * direction_versor)<<std::endl;
          }
      }
    pcout<<"set material and manifold ids"<<std::endl;
    cylinder_manifold = std::shared_ptr<CylindricalManifold<dim-1, dim> > (new CylindricalManifold<dim-1, dim> (direction,point_on_axis));
    triangulation_wall.set_manifold(99, *cylinder_manifold);
    if (this_mpi_process == 0)
      {
        std::string filename_new = "cylinder.inp";
        std::ofstream wall_ofs;
        wall_ofs.open(filename_new, std::ofstream::out);
        GridOut go;
        go.write_ucd(triangulation_wall,wall_ofs);
      }
    MPI_Barrier(mpi_communicator);

  }


  template<int dim>
  void BEMProblem<dim>::add_cylinder_to_tria(Triangulation<dim-1, dim> &triangulation, bool apply_manifold, std::string filename)
  {
    Triangulation<dim-1, dim> triangulation_wall;
    Triangulation<dim-1, dim> triangulation_old;
    triangulation_old.copy_triangulation(triangulation);
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation_wall);
    std::ifstream in;
    in.open (filename);
    gi.read_ucd(in, true);

    GridGenerator::merge_triangulations(triangulation_old, triangulation_wall, triangulation);

    if (apply_manifold)
      {
        typename Triangulation<dim-1,dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (cell=triangulation.begin_active(); cell != endc; ++cell)
          {
            if (cell->material_id()==2 || cell->material_id()==3)
              {
                cell->set_all_manifold_ids(99);
              }
          }
        if (cylinder_manifold)
          triangulation.set_manifold(99, *cylinder_manifold);
        else
          triangulation.set_manifold(99);
      }
  }

  template<>
  void BEMProblem<2>::create_coarse_wall(Triangulation<1, 2> &, std::string &, Point<2> &, std::vector<double> &, bool) const
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template<int dim>
  void BEMProblem<dim>::create_coarse_wall(Triangulation<dim-1, dim> &triangulation_wall, std::string &wall_type, Point<dim> &position, std::vector<double> &span, bool flip_normal) const
  {
    Assert(span.size() == dim, ExcMessage("Incopatible span, size. Expected == dim"));
    pcout<<"Creating a coarse "<<wall_type<<std::endl;
    // Triangulation<dim-1, dim> triangulation1(triangulation);

    std::vector<Point<dim> > vertices(4);
    // Point<dim> P1(position), P2(position), P3(position), P4(position);
    unsigned int foo_dim=numbers::invalid_unsigned_int, k=0;
    std::vector<unsigned int> true_dim(dim-1);

    vertices[0] = position;
    vertices[1] = position;
    vertices[2] = position;
    vertices[3] = position;

    for (unsigned int i=0; i<dim; ++i)
      {
        if (span[i] != -1. && span[i] !=0)
          {
            vertices[0][i] += span[i];
            vertices[2][i] -= span[i];
          }
        if (span[i]==0)
          {
            foo_dim=i;
          }
        else
          {
            true_dim[k]=i;
            k+=1;
          }
        // else
        // {
        //   P1[i] += max_span * infinite_factor;
        //   P2[i] -= max_span * infinite_factor;
        // }

      }
    // pcout<<foo_dim<<std::endl;
    Assert(foo_dim != numbers::invalid_unsigned_int, ExcMessage("A wall needs a zero dimension"))

    vertices[1][true_dim[0]]+=span[true_dim[0]];
    vertices[1][true_dim[1]]-=span[true_dim[1]];
    vertices[3][true_dim[0]]-=span[true_dim[0]];
    vertices[3][true_dim[1]]+=span[true_dim[1]];

    // pcout<<P1<<" "<<P2<<" "<<P3<<" "<<P4<<" "<<std::endl;
    std::vector<unsigned int> repetitions(dim-1);
    // for(unsigned int i=0; i<dim-1; ++i)
    //   repetitions[i]=2;
    // repetitions[1]=2;

    std::vector<CellData<dim-1> > cells(1);
    SubCellData subcelldata;

    if (foo_dim == 1)
      {
        // pos_y>0=>ok
        if (position[foo_dim]>0 && !(flip_normal))
          {
            cells[0].vertices[0]=0;
            cells[0].vertices[1]=3;
            cells[0].vertices[2]=1;
            cells[0].vertices[3]=2;
          }
        else
          {
            cells[0].vertices[0]=0;
            cells[0].vertices[1]=1;
            cells[0].vertices[2]=3;
            cells[0].vertices[3]=2;
          }
      }
    else
      {
        if (position[foo_dim]>0 && !(flip_normal))
          {
            cells[0].vertices[0]=0;
            cells[0].vertices[1]=1;
            cells[0].vertices[2]=3;
            cells[0].vertices[3]=2;
          }
        else
          {
            cells[0].vertices[0]=0;
            cells[0].vertices[1]=3;
            cells[0].vertices[2]=1;
            cells[0].vertices[3]=2;
          }

      }

    const std::vector<Point<dim> > dummy_vertices(vertices);
    const std::vector<CellData<dim-1> > dummy_cells(cells);
    triangulation_wall.clear();
    triangulation_wall.create_triangulation(dummy_vertices, dummy_cells, subcelldata);
    if (wall_type == "no_slip")
      {
        triangulation_wall.begin_active()->set_material_id(2);
      }
    else if (wall_type == "free_surface")
      {
        triangulation_wall.begin_active()->set_material_id(3);
      }
    else if (wall_type == "do_nothing")
      {
        triangulation_wall.begin_active()->set_material_id(4);
      }
    else if (wall_type == "dirichlet")
      {
        triangulation_wall.begin_active()->set_material_id(5);
      }
    else if (wall_type == "neumann")
      {
        triangulation_wall.begin_active()->set_material_id(6);
      }
    else
      Assert(false, ExcNotImplemented());

  }

  template<int dim>
  void BEMProblem<dim>::create_wall(Triangulation<dim-1, dim> &triangulation, std::string wall_type, Point<dim> position, std::vector<double> span, unsigned int wall_number, bool flip_normal)
  {
    pcout<<"creating wall with id "<<wall_number<<std::endl;
    Triangulation<dim-1, dim> triangulation_wall;
    create_coarse_wall(triangulation_wall, wall_type, position, span, flip_normal);
    Point<dim> center(triangulation_wall.begin_active()->center());
    // GridGenerator::subdivided_hyper_rectangle(triangulation_wall,repetitions, P1, P2);
    triangulation_wall.refine_global();
    if (this_mpi_process == 0)
      {
        std::string filename = "wall_"+Utilities::int_to_string(wall_number)+".inp";
        std::ofstream wall_ofs;
        wall_ofs.open(filename, std::ofstream::out);
        GridOut go;
        go.write_ucd(triangulation_wall,wall_ofs);
      }
    MPI_Barrier(mpi_communicator);
    // go.write_eps(triangulation_wall,wall_ofs);
    add_wall_to_tria(triangulation,wall_number);

    // triangulation_wall.clear();
    // std::ifstream in;
    // in.open (filename);
    // GridIn<dim-1, dim> gi;
    // gi.attach_triangulation (triangulation_wall);
    // gi.read_ucd (in, true);
    // pcout<<"MERGING"<<std::endl;
    // Triangulation<dim-1, dim> foo_tria;
    // foo_tria.copy_triangulation(triangulation);
    // GridGenerator::merge_triangulations(foo_tria, triangulation_wall, triangulation);
    // pcout<<"MERGED"<<std::endl;

  }

  template<int dim>
  void BEMProblem<dim>::create_box(Triangulation<dim-1, dim> &triangulation, unsigned int starting_index)
  {
    Triangulation<dim-1, dim> triangulation_wall;
    Triangulation<dim-1, dim> triangulation_box;
    Triangulation<dim-1, dim> triangulation_helper;

    for (unsigned int i=starting_index; i<starting_index+6; ++i)
      {
        AssertThrow(wall_bool[i], ExcMessage("I need the wall to create a box"));
        triangulation_wall.clear();
        create_coarse_wall(triangulation_wall, wall_types[i], wall_positions[i], wall_spans[i], flip_normal_wall_bool[i]);
        if (i>starting_index)
          {
            triangulation_helper.clear();
            triangulation_helper.copy_triangulation(triangulation_box);
            triangulation_box.clear();
            GridGenerator::merge_triangulations(triangulation_wall, triangulation_helper, triangulation_box);

          }
        else
          {
            triangulation_box.clear();
            triangulation_box.copy_triangulation(triangulation_wall);
          }
        pcout<<"Added wall "<<i<<" to the box "<<triangulation_box.n_active_cells()<<std::endl;


      }
    triangulation_box.refine_global(1);
    if (this_mpi_process == 0)
      {
        std::string filename = "box.inp";
        std::ofstream wall_ofs;
        wall_ofs.open(filename, std::ofstream::out);
        GridOut go;
        go.write_ucd(triangulation_box,wall_ofs);
      }
    // dpcout<<"MAREMMA BUFALA"<<std::endl;

    MPI_Barrier(mpi_communicator);
    // go.write_eps(triangulation_wall,wall_ofs);
    // dpcout<<"MAREMMA BUFALA"<<std::endl;
    add_box_to_tria(triangulation);



  }

  template<int dim>
  void BEMProblem<dim>::add_wall_to_tria(Triangulation<dim-1, dim> &triangulation, unsigned int wall_number)
  {
    Triangulation<dim-1, dim> triangulation_wall;
    Triangulation<dim-1, dim> triangulation_old;
    if (triangulation.n_active_cells()>=1)
      triangulation_old.copy_triangulation(triangulation);
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation_wall);
    std::ifstream in;
    std::string filename = "wall_"+Utilities::int_to_string(wall_number)+".inp";
    in.open (filename);
    gi.read_ucd(in, true);
    triangulation_wall.set_all_manifold_ids(0);
    if (triangulation.n_active_cells()>=1)
      GridGenerator::merge_triangulations(triangulation_old, triangulation_wall, triangulation);
    else
      triangulation.copy_triangulation(triangulation_wall);

  }

  template<int dim>
  void BEMProblem<dim>::add_box_to_tria(Triangulation<dim-1, dim> &triangulation, std::string filename)
  {
    Triangulation<dim-1, dim> triangulation_box;
    Triangulation<dim-1, dim> triangulation_old;
    GridIn<dim-1, dim> gi;
    gi.attach_triangulation (triangulation_box);
    std::ifstream in;
    in.open (filename);
    gi.read_ucd(in, true);
    // triangulation.clear();
    if (triangulation.n_active_cells()!=0)
      {
        triangulation_old.copy_triangulation(triangulation);
        GridGenerator::merge_triangulations(triangulation_old, triangulation_box, triangulation);
      }
    else
      {
        triangulation.copy_triangulation(triangulation_box);
      }
    // dpcout<<"PIPPO"<<std::endl;
  }


  template<>
  void BEMProblem<2>::create_wall(Triangulation<1, 2> &, std::string , Point<2> , std::vector<double> , unsigned int , bool )
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }


  template <int dim>
  void BEMProblem<dim>::refine_walls(Triangulation<dim-1, dim> &triangulation, const double max_distance, const double threshold, const Point<dim> &center, bool gradual_refinement)
  {
    bool refine = true;
    while (refine)
      {
        refine = false;
        typename Triangulation<dim-1,dim>::active_cell_iterator
        cell = triangulation.begin_active(),
        endc = triangulation.end();
        for (cell=triangulation.begin_active(); cell != endc; ++cell)
          {
            // We refine ONLY if we are on the walls (free surface o no slip).
            if (cell->material_id()==2 || cell->material_id()==3 )
              {
                for (unsigned int v=0; v < GeometryInfo<dim-1>::vertices_per_cell; ++v)
                  {

                    double distance_from_center = center.distance(cell->vertex(v));
                    double my_threshold = threshold;
                    if (gradual_refinement)
                      {
                        my_threshold *= std::min(std::max(std::pow((distance_from_center)/max_distance,1.),0.1),1.);//,1.);
                        // distance_from_center = max_distance-0.1;
                      }
                    if (distance_from_center < max_distance && cell->diameter() > my_threshold)
                      {
                        cell->set_refine_flag();
                        refine = true;
                      }
                    // else
                    // pcout<<distance_from_center<<" "<<max_distance<<" : "<<cell->diameter()<<" "<<threshold<<std::endl;
                    // break;
                  }
              }
          }
        triangulation.prepare_coarsening_and_refinement();
        triangulation.execute_coarsening_and_refinement();
      }
    typename Triangulation<dim-1,dim>::active_cell_iterator
    cell = triangulation.begin_active(),
    endc = triangulation.end();




  }

  template<int dim>
  void BEMProblem<dim>::remove_hanging_nodes_between_different_material_id( Triangulation<dim-1,dim> &tria_in,
      const bool isotropic,
      const unsigned int max_iterations)
  {
    unsigned int iter = 0;
    bool continue_refinement = true;

    typename Triangulation<dim-1, dim>::active_cell_iterator
    cell = tria_in.begin_active(),
    endc = tria_in.end();

    while ( continue_refinement && (iter < max_iterations) )
      {
        if (max_iterations != numbers::invalid_unsigned_int) iter++;
        continue_refinement = false;

        for (cell=tria_in.begin_active(); cell!= endc; ++cell)
          for (unsigned int j = 0; j < GeometryInfo<dim-1>::faces_per_cell; j++)
            if (cell->at_boundary(j)==false && cell->neighbor(j)->has_children() && (cell->neighbor(j)->material_id() != cell->material_id()))
              {
                if (isotropic)
                  {
                    cell->set_refine_flag();
                    continue_refinement = true;
                  }
                else
                  continue_refinement |= cell->flag_for_face_refinement(j);
              }

        tria_in.execute_coarsening_and_refinement();
      }
  }

// @sect4{BEMProblem::reinit}

// This function globally refines the mesh, distributes degrees of freedom,
// and resizes matrices and vectors.

  template <int dim>
  void BEMProblem<dim>::reinit()
  {
    if (dim == 2)
      num_rigid = dim + 1;
    else if (dim == 3)
      num_rigid = dim + dim;

    //std::cout<<"initialize"<<std::endl;

    //tria.refine_global(1);̵
    if (dim == 3)
      {
        rotation_matrix[0][0] = 1. + 2 * -(initial_quaternion[3]*initial_quaternion[3] + initial_quaternion[2]*initial_quaternion[2]);
        rotation_matrix[0][1] = - 2 * initial_quaternion[0] * initial_quaternion[3] + 2 * (initial_quaternion[1] * initial_quaternion[2]);
        rotation_matrix[0][2] = + 2 * initial_quaternion[0] * initial_quaternion[2] + 2 * (initial_quaternion[1] * initial_quaternion[3]);
        rotation_matrix[1][0] = + 2 * initial_quaternion[0] * initial_quaternion[3] + 2 * (initial_quaternion[1] * initial_quaternion[2]);
        rotation_matrix[1][1] = 1. + 2 * -(initial_quaternion[3]*initial_quaternion[3] + initial_quaternion[1]*initial_quaternion[1]);
        rotation_matrix[1][2] = - 2 * initial_quaternion[0] * initial_quaternion[1] + 2 * (initial_quaternion[3] * initial_quaternion[2]);
        rotation_matrix[2][0] = - 2 * initial_quaternion[0] * initial_quaternion[2] + 2 * (initial_quaternion[1] * initial_quaternion[3]);
        rotation_matrix[2][1] = + 2 * initial_quaternion[0] * initial_quaternion[1] + 2 * (initial_quaternion[3] * initial_quaternion[2]);
        rotation_matrix[2][2] = 1. + 2 * -(initial_quaternion[1]*initial_quaternion[1] + initial_quaternion[2]*initial_quaternion[2]);
      }
    else
      for (unsigned int i=0; i<dim; ++i)
        rotation_matrix(i,i) = 1.;
    dh_stokes.distribute_dofs(*fe_stokes);
    pcout << "There are " << dh_stokes.n_dofs() << " degrees of freedom"<< std::endl;
    map_dh.distribute_dofs(*fe_map);
    // pcout<<dh_stokes.n_dofs()<<" "<<map_dh.n_dofs()<<std::endl;
    pcout<<"re-ordering vector"<<std::endl;

    DoFRenumbering::component_wise (dh_stokes);
    DoFRenumbering::component_wise (map_dh);

    // DoFRenumbering::subdomain_wise (dh_stokes);
    // DoFRenumbering::subdomain_wise (map_dh);

    std::vector<types::subdomain_id> stokes_dofs_domain_association(dh_stokes.n_dofs());

    DoFTools::get_subdomain_association   (dh_stokes,stokes_dofs_domain_association);

    std::vector<types::subdomain_id> map_dofs_domain_association(map_dh.n_dofs());

    DoFTools::get_subdomain_association   (map_dh,map_dofs_domain_association);

    this_cpu_set.clear();
    map_cpu_set.clear();
    monolithic_cpu_set.clear();
    this_cpu_set.set_size(dh_stokes.n_dofs());
    map_cpu_set.set_size(map_dh.n_dofs());
    if (solve_with_torque)
      monolithic_cpu_set.set_size(dh_stokes.n_dofs()+num_rigid+1);
    else
      monolithic_cpu_set.set_size(dh_stokes.n_dofs()+num_rigid);
    for (types::global_dof_index i = 0; i<dh_stokes.n_dofs()/dim; ++i )
      {
        // pcout<<i<<" "<<stokes_dofs_domain_association[i]<<std::endl;
        if (stokes_dofs_domain_association[i] == this_mpi_process)
          {
            for (unsigned int idim = 0; idim<dim; ++idim)
              {
                this_cpu_set.add_index(i+idim*dh_stokes.n_dofs()/dim);
                monolithic_cpu_set.add_index(i+idim*dh_stokes.n_dofs()/dim);
              }
          }
      }
    if (this_mpi_process == 0)
      {
        for (types::global_dof_index i = dh_stokes.n_dofs(); i<dh_stokes.n_dofs() + num_rigid; ++i)
          monolithic_cpu_set.add_index(i);
        if (solve_with_torque)
          monolithic_cpu_set.add_index(dh_stokes.n_dofs() + num_rigid);
      }
    for (types::global_dof_index i = 0; i<map_dh.n_dofs()/dim; ++i )
      {
        if (map_dofs_domain_association[i] == this_mpi_process)
          {
            for (unsigned int idim = 0; idim<dim; ++idim)
              {
                map_cpu_set.add_index(i+idim*map_dh.n_dofs()/dim);
              }
          }
      }
    monolithic_cpu_set.compress();
    this_cpu_set.compress();
    map_cpu_set.compress();

    create_wall_body_index_sets();

    const unsigned int n_dofs_stokes =  dh_stokes.n_dofs();

    // PROBLEM WITH CONSTRAINTS
    constraints.reinit ();
    DoFTools::make_hanging_node_constraints (dh_stokes,constraints);
    constraints.close ();
    // constraints_null.reinit ();
    // constraints_null.close ();

    // IndexSet full_index_set, full_monolithic_index_set;
    full_sparsity_pattern.reinit(this_cpu_set,mpi_communicator,this_cpu_set.size());//(this_cpu_set, complete_index_set(dh_stokes.n_dofs()), mpi_communicator, dh_stokes.n_dofs());
    monolithic_full_sparsity_pattern.reinit(monolithic_cpu_set,mpi_communicator,monolithic_cpu_set.size()) ;//(monolithic_cpu_set, complete_index_set(dh_stokes.n_dofs()+num_rigid), mpi_communicator, dh_stokes.n_dofs()+num_rigid);
    pcout<<"re-initializing sparsity patterns and matrices"<<std::endl;

    for (auto i : this_cpu_set)
      {
        if (constraints.is_constrained(i)==true)
          {
            std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);
            for (unsigned int ic=0; ic<constr_in.size(); ic++)
              {
                full_sparsity_pattern.add(i,i);
                monolithic_full_sparsity_pattern.add(i,i);
                full_sparsity_pattern.add(i,constr_in[ic].first);
                monolithic_full_sparsity_pattern.add(i,constr_in[ic].first);
              }

          }
        else
          for (types::global_dof_index j=0; j<dh_stokes.n_dofs(); ++j)
            {
              full_sparsity_pattern.add(i,j);
              monolithic_full_sparsity_pattern.add(i,j);
            }
        for (types::global_dof_index j=dh_stokes.n_dofs(); j<dh_stokes.n_dofs()+num_rigid; ++j)
          monolithic_full_sparsity_pattern.add(i,j);


      }
    if (this_mpi_process==0)
      for (types::global_dof_index i=dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
        for (types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid; ++j)
          monolithic_full_sparsity_pattern.add(i,j);

    if (solve_with_torque)
      {
        for (auto i : monolithic_cpu_set)
          {
            monolithic_full_sparsity_pattern.add(i,dh_stokes.n_dofs()+num_rigid);
          }
        if (this_mpi_process == 0)
          {
            for (types::global_dof_index i=dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
              monolithic_full_sparsity_pattern.add(i,dh_stokes.n_dofs()+num_rigid);
            for (types::global_dof_index j=0; j<dh_stokes.n_dofs()+num_rigid+1; ++j)
              monolithic_full_sparsity_pattern.add(dh_stokes.n_dofs()+num_rigid,j);


          }
      }
    full_sparsity_pattern.compress();
    monolithic_full_sparsity_pattern.compress();

    V_matrix.clear();
    K_matrix.clear();
    // D_matrix.clear();
    V_matrix.reinit(full_sparsity_pattern);
    K_matrix.reinit(full_sparsity_pattern);
    // D_matrix.reinit(full_sparsity_pattern);

    monolithic_system_matrix.clear();
    monolithic_system_matrix.reinit(monolithic_full_sparsity_pattern);
    monolithic_system_preconditioner_matrix.clear();
    if (bandwith_preconditioner)
      {
        monolithic_preconditioner_sparsity_pattern = new TrilinosWrappers::SparsityPattern (monolithic_cpu_set, mpi_communicator, bandwith);
        for (auto i : monolithic_cpu_set)
          {
            if (constraints.is_constrained(i)==true)
              {
                std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);
                for (unsigned int ic=0; ic<constr_in.size(); ic++)
                  {
                    monolithic_preconditioner_sparsity_pattern->add(i,constr_in[ic].first);
                  }

              }
            else
              {
                types::global_dof_index my_start, my_end;
                if (i > bandwith)
                  my_start = i-bandwith;
                else
                  my_start = 0;
                if (i+bandwith > dh_stokes.n_dofs()+num_rigid)
                  my_end = dh_stokes.n_dofs()+num_rigid;
                else
                  my_end = i+bandwith;
                for (types::global_dof_index j=my_start; j<my_end; ++j)
                  {
                    monolithic_preconditioner_sparsity_pattern->add(i,j);
                  }

              }


          }


        monolithic_preconditioner_sparsity_pattern->compress();
      }
    else
      {
        monolithic_preconditioner_sparsity_pattern = &monolithic_full_sparsity_pattern;

      }

    monolithic_system_preconditioner_matrix.reinit(*monolithic_preconditioner_sparsity_pattern);

    monolithic_solution.reinit(monolithic_cpu_set, mpi_communicator);
    monolithic_rhs.reinit(monolithic_cpu_set, mpi_communicator);

    std::vector<unsigned int> my_sizes(n_dofs_stokes, n_dofs_stokes);
    NdiadicN.reinit(full_sparsity_pattern);
    //matrix.resize(n_row, std::vector<double> (n_col));
    N_rigid.clear();
    DN_N_rigid.clear();
    N_rigid_dual.clear();
    N_rigid_dual_complete.clear();
    N_rigid_map.clear();
    dpcout<<"CHIEDERE PER REINIT RIDONDANTE PER V_test_with_Green"<<std::endl;
    N_rigid.resize         (num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_complete.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_flagellum_translation.reinit(this_cpu_set,mpi_communicator);
    N_flagellum_torque.reinit(this_cpu_set,mpi_communicator);
    N_flagellum_torque_dual.reinit(this_cpu_set,mpi_communicator);
    for (unsigned int i=0; i<num_rigid; ++i)
      {
        N_rigid_complete[i].reinit(this_cpu_set, mpi_communicator);
      }
    DN_N_rigid.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_dual.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_dual_complete.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_map.resize(num_rigid, TrilinosWrappers::MPI::Vector (map_cpu_set,mpi_communicator));
    N_rigid_map_complete.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    // pcout<<"???"<<std::endl;
    // this_cpu_set.print(std::cout);
    // N_rigid[0].locally_owned_elements().print(std::cout);
    // N_rigid_complete[0].locally_owned_elements().print(std::cout);
    V_x_normals.reinit(this_cpu_set,mpi_communicator);
    V_x_normals_body.reinit(this_cpu_set,mpi_communicator);
    vnormals.reinit(this_cpu_set,mpi_communicator);

    rigid_velocities.reinit(num_rigid);
    baricenter_rigid_velocities.reinit(num_rigid);
    rigid_total_forces.reinit(num_rigid);
    rigid_total_forces_complete.reinit(num_rigid);
    rigid_puntual_velocities.reinit(this_cpu_set,mpi_communicator);
    rigid_puntual_translation_velocities.reinit(this_cpu_set,mpi_communicator);
    rigid_puntual_displacements.reinit(this_cpu_set,mpi_communicator);
    next_rigid_puntual_displacements.reinit(this_cpu_set,mpi_communicator);
    stokes_forces.reinit(this_cpu_set,mpi_communicator);
    final_test.reinit(this_cpu_set,mpi_communicator);
    stokes_rhs.reinit(this_cpu_set,mpi_communicator);
    global_components.reinit(this_cpu_set,mpi_communicator);
    shape_velocities.reinit(this_cpu_set,mpi_communicator);
    total_velocities.reinit(this_cpu_set,mpi_communicator);

    euler_vec.reinit(map_dh.n_dofs());
    reference_euler_vec.reinit(map_dh.n_dofs());
    rigid_displacements_for_sim.reinit(map_dh.n_dofs());
    next_euler_vec.reinit(map_dh.n_dofs());
    total_euler_vec.reinit(map_dh.n_dofs());

    normal_vector.reinit(this_cpu_set,mpi_communicator);
    wall_velocities.reinit(this_cpu_set,mpi_communicator);
    M_normal_vector.reinit(this_cpu_set,mpi_communicator);
    normal_vector_pure.reinit(this_cpu_set,mpi_communicator);
    M_normal_vector_pure.reinit(this_cpu_set,mpi_communicator);

    // first_evec.reinit(dh_stokes.n_dofs());
    // M_first_evec.reinit(dh_stokes.n_dofs());

    // reference_support_points.resize(map_dh.n_dofs());
    // DoFTools::map_dofs_to_support_points<dim-1, dim>(StaticMappingQ1<dim-1, dim>::mapping,
    //         map_dh, reference_support_points);
    //csp.reinit(dh_stokes.n_dofs(), dh_stokes.n_dofs());

    sparsity_pattern.clear();
    sparsity_pattern.reinit(this_cpu_set, this_cpu_set, mpi_communicator);
    DoFTools::make_sparsity_pattern (dh_stokes, sparsity_pattern, constraints, true, this_mpi_process);
    sparsity_pattern.compress();
    //csp.compress ();
    Mass_Matrix.clear();
    Mass_Matrix.reinit(sparsity_pattern);
    // time_step = 0.1;
    // std::cout<<this_cpu_set.size()<<std::endl;

    dpcout<<fe_map->n_components()<<std::endl;
    if (mappingeul == NULL)
      mappingeul = SP(new MappingFEField<dim-1,dim>(map_dh, euler_vec));
    if (use_flagellum_handler)
      flagellum_handler.set_geometry_cache(map_dh,&map_flagellum_cpu_set,&(*mappingeul));
    //compute_global_components();
    //euler_vec.reinit(map_dh.n_dofs());
    gmres_additional_data.max_n_tmp_vectors=gmres_restart;
  }

  template <int dim>
  void BEMProblem<dim>::compute_constraints_for_single_layer()
  {
    ConstraintMatrix constraints_single_layer;
    // compute_normal_vector();
    for (types::global_dof_index i = (dim-1)*dh_stokes.n_dofs()/dim; i<dh_stokes.n_dofs(); ++i)
      {
        if (!(constraints.is_constrained(i)) && body_cpu_set.is_element(i))
          {
            constraints_single_layer.add_line(i);
            for (unsigned int j=1; j<dim; ++j)
              {
                constraints_single_layer.add_entry(i, i-j*dh_stokes.n_dofs()/dim, -normal_vector_pure(i-j*dh_stokes.n_dofs()/dim)/normal_vector_pure(i));
              }
            i_single_layer_constraint = i;
            pcout<<"!!!!!!! montato constraint"<<std::endl;

            break;
          }
      }
    constraints_single_layer.close();
    constraints.merge(constraints_single_layer);
  }


// This function creates the index sets that identifies the dof of the body or the walls,
// we need to distinguish between no slip and free surface walls. We will use them to compute the
// shape velocities ONLY on the body and not on the walls. Moreover we will use them to properly
// assemble the monolithic system matrix.
  template <int dim>
  void BEMProblem<dim>::create_wall_body_index_sets()
  {
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();

    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    body_cpu_set.clear();
    body_cpu_set.set_size(dh_stokes.n_dofs());
    flagellum_cpu_set.clear();
    flagellum_cpu_set.set_size(dh_stokes.n_dofs());
    head_cpu_set.clear();
    head_cpu_set.set_size(dh_stokes.n_dofs());
    wall_no_slip_cpu_set.clear();
    wall_no_slip_cpu_set.set_size(dh_stokes.n_dofs());
    wall_free_surface_cpu_set.clear();
    wall_free_surface_cpu_set.set_size(dh_stokes.n_dofs());
    wall_do_nothing_cpu_set.clear();
    wall_do_nothing_cpu_set.set_size(dh_stokes.n_dofs());
    wall_dirichlet_cpu_set.clear();
    wall_dirichlet_cpu_set.set_size(dh_stokes.n_dofs());
    wall_neumann_cpu_set.clear();
    wall_neumann_cpu_set.set_size(dh_stokes.n_dofs());
    std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( StaticMappingQ1<dim-1, dim>::mapping, dh_stokes, support_points);//StaticMappingQ1<dim-1, dim>::mapping
    // for(unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
    // {
    //   pcout<<support_points[i].square()<<std::endl;
    // }
    for (cell = dh_stokes.begin_active(); cell<endc; ++cell)
      {
        // if(cell->subdomain_id() == this_mpi_process)
        {
          cell->get_dof_indices(local_dof_indices);
          if (cell->material_id() == 2) //wall_no_slip_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  wall_no_slip_cpu_set.add_index(local_dof_indices[j]);
                }
            }
          else if (cell->material_id() == 3) //wall_free_surface_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  wall_free_surface_cpu_set.add_index(local_dof_indices[j]);
                }

            }
          else if (cell->material_id() == 4) //wall_do_nothing_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  wall_do_nothing_cpu_set.add_index(local_dof_indices[j]);
                }

            }
          else if (cell->material_id() == 5) //wall_dirichlet_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  wall_dirichlet_cpu_set.add_index(local_dof_indices[j]);
                }

            }
          else if (cell->material_id() == 6) //wall_neumann_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  wall_neumann_cpu_set.add_index(local_dof_indices[j]);
                }

            }
          else if (cell->material_id() == 0) // body_cpu_set
            {
              for (unsigned int j=0; j<local_dof_indices.size(); ++j)
                {
                  body_cpu_set.add_index(local_dof_indices[j]);
                  if (support_points[local_dof_indices[j]][0]>-0.1)
                    flagellum_cpu_set.add_index(local_dof_indices[j]);
                  else
                    head_cpu_set.add_index(local_dof_indices[j]);
                }

            }
        }
      }
    // body_cpu_set.print(std::cout);
    // head_cpu_set.print(std::cout);
    body_cpu_set.compress();
    flagellum_cpu_set.compress();
    head_cpu_set.compress();
    wall_free_surface_cpu_set.compress();
    wall_no_slip_cpu_set.compress();
    wall_do_nothing_cpu_set.compress();

    // wall_no_slip_cpu_set.print(std::cout);
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell_map = map_dh.begin_active(),
    endc_map = map_dh.end();

    std::vector<types::global_dof_index> map_local_dof_indices(fe_map->dofs_per_cell);

    map_body_cpu_set.clear();
    map_body_cpu_set.set_size(map_dh.n_dofs());
    map_flagellum_cpu_set.clear();
    map_flagellum_cpu_set.set_size(map_dh.n_dofs());
    map_head_cpu_set.clear();
    map_head_cpu_set.set_size(map_dh.n_dofs());
    map_wall_no_slip_cpu_set.clear();
    map_wall_no_slip_cpu_set.set_size(map_dh.n_dofs());
    map_wall_free_surface_cpu_set.clear();
    map_wall_free_surface_cpu_set.set_size(map_dh.n_dofs());
    map_wall_do_nothing_cpu_set.clear();
    map_wall_do_nothing_cpu_set.set_size(map_dh.n_dofs());
    map_wall_dirichlet_cpu_set.clear();
    map_wall_dirichlet_cpu_set.set_size(map_dh.n_dofs());
    map_wall_neumann_cpu_set.clear();
    map_wall_neumann_cpu_set.set_size(map_dh.n_dofs());
    std::vector<Point<dim> > map_support_points(map_dh.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( StaticMappingQ1<dim-1, dim>::mapping, map_dh, map_support_points);
    for (cell_map = map_dh.begin_active(); cell_map<endc_map; ++cell_map)
      {
        // if(cell_map->subdomain_id() == this_mpi_process)
        {
          cell_map->get_dof_indices(map_local_dof_indices);
          if (cell_map->material_id() == 2) //wall_no_slip_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_wall_no_slip_cpu_set.add_index(map_local_dof_indices[j]);
                }
            }
          else if (cell_map->material_id() == 3) //wall_free_surface_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_wall_free_surface_cpu_set.add_index(map_local_dof_indices[j]);
                }

            }
          else if (cell_map->material_id() == 4) //wall_do_nothing_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_wall_do_nothing_cpu_set.add_index(map_local_dof_indices[j]);
                }

            }
          else if (cell_map->material_id() == 5) //wall_dirichlet_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_wall_dirichlet_cpu_set.add_index(map_local_dof_indices[j]);
                }

            }
          else if (cell_map->material_id() == 6) //wall_neumann_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_wall_neumann_cpu_set.add_index(map_local_dof_indices[j]);
                }

            }
          else// body_cpu_set
            {
              for (unsigned int j=0; j<map_local_dof_indices.size(); ++j)
                {
                  map_body_cpu_set.add_index(map_local_dof_indices[j]);
                  if (map_support_points[map_local_dof_indices[j]][0]>0)
                    map_flagellum_cpu_set.add_index(map_local_dof_indices[j]);
                  else
                    map_head_cpu_set.add_index(map_local_dof_indices[j]);

                }

            }
        }
      }
    map_body_cpu_set.compress();
    map_flagellum_cpu_set.compress();
    map_head_cpu_set.compress();
    map_wall_free_surface_cpu_set.compress();
    map_wall_no_slip_cpu_set.compress();
    map_wall_do_nothing_cpu_set.compress();

  }

  template <int dim>
  void BEMProblem<dim>::compute_rotational_shape_velocities(TrilinosWrappers::MPI::Vector &rotational_shape_vel, const TrilinosWrappers::MPI::Vector &rotation_mode)
  {
    pcout<<"Imposing rotational velocity"<<std::endl;
    double omega=-2.*numbers::PI/time_step/n_frames;

    // std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
    // DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);

    for (auto i : this_cpu_set)
      {
        if (flagellum_cpu_set.is_element(i))
          {
            rotational_shape_vel[i]=rotation_mode[i]*omega;
          }
      }


  }
  template <int dim>
  void BEMProblem<dim>::compute_traslational_shape_velocities(TrilinosWrappers::MPI::Vector &traslation_shape_vel, const TrilinosWrappers::MPI::Vector &traslation_mode)
  {
    pcout<<"Imposing translational velocity"<<std::endl;
    double vel=1.;//-2.*numbers::PI/time_step/n_frames;

    // std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
    // DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);

    for (auto i : this_cpu_set)
      {
        if (flagellum_cpu_set.is_element(i))
          {
            traslation_shape_vel[i]=traslation_mode[i]*vel;
          }
      }

    // traslation_shape_vel.print(std::cout);
  }

// If we consider two different FESystems for mapping and unkwons we need to project the vectors between them
  template <int dim>
  void BEMProblem<dim>::project_shape_velocities()
  {
    pcout<<"Projecting shape velocities for non isoparametric BEMs"<<std::endl;
    if (fe_stokes->get_name() == fe_map->get_name())
      for (auto i : this_cpu_set)
        shape_velocities[i] = next_euler_vec[i]/time_step - euler_vec[i]/time_step;
    else
      {
        FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                        update_values |
                                        update_cell_normal_vectors |
                                        update_quadrature_points |
                                        update_JxW_values);
        FEValues<dim-1,dim> fe_map_v(*mappingeul, *fe_map, quadrature,
                                     update_values |
                                     update_cell_normal_vectors |
                                     update_quadrature_points |
                                     update_JxW_values);
        std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);
        std::vector<types::global_dof_index> local_dof_indices_map(fe_map->dofs_per_cell);

        typename DoFHandler<dim-1,dim>::active_cell_iterator
        cell = dh_stokes.begin_active(),
        endc = dh_stokes.end();

        typename DoFHandler<dim-1,dim>::active_cell_iterator
        cell_map = map_dh.begin_active(),
        endc_map = map_dh.end();

        unsigned int comp_i;
        TrilinosWrappers::MPI::Vector rhs_shape(this_cpu_set, mpi_communicator);
        Vector<double> helper(next_euler_vec);
        helper.sadd(1.,-1.,euler_vec);

        for (cell = dh_stokes.begin_active(), cell_map=map_dh.begin_active(); cell != endc; ++cell, ++cell_map)
          {
            if (cell->subdomain_id() == this_mpi_process)
              {
                fe_stokes_v.reinit(cell);
                fe_map_v.reinit(cell_map);

                const std::vector<Point<dim> > q_points = fe_stokes_v.get_quadrature_points();
                double n_q_points = q_points.size();

                cell->get_dof_indices(local_dof_indices);
                cell_map->get_dof_indices(local_dof_indices_map);

                Vector<double> local_rhs(fe_stokes->dofs_per_cell);
                for (unsigned int q=0; q<n_q_points; ++q)
                  {
                    std::vector<Vector<double> > shape_map_values(n_q_points, Vector<double> (dim));
                    fe_map_v.get_function_values(helper, shape_map_values);

                    for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
                      {
                        comp_i = fe_stokes->system_to_component_index(i).first;
                        local_rhs[i] += fe_stokes_v.shape_value(i,q) *
                                        shape_map_values[q][comp_i] *
                                        fe_stokes_v.JxW(q);

                        // for(unsigned int j = 0; j<fe_map->dofs_per_cell; ++j)
                        // {
                        //   comp_j = fe_map->system_to_component_index(j).first;
                        //   if(comp_j == comp_i)
                        //     local_rhs[i] += fe_stokes_v.shape_value(i,q) *
                        //                                fe_map_v.shape_value(j,q) *
                        //                                (next_euler_vec[local_dof_indices_map[j]]-euler_vec[local_dof_indices_map[j]]) *
                        //                                fe_stokes_v.JxW(q);
                        // }

                      }

                  }

                local_rhs /= time_step;
                constraints.distribute_local_to_global
                (local_rhs,
                 local_dof_indices,
                 rhs_shape);

              }
          }
        rhs_shape.compress(VectorOperation::add);
        // Mass_Matrix.print(std::cout);
        TrilinosWrappers::SolverCG solver(solver_control);
        TrilinosWrappers::PreconditionJacobi preconditioner_mass;
        preconditioner_mass.initialize(Mass_Matrix, 1.3);
        // mass_prec.initialize(Mass_Matrix);
        solver.solve (Mass_Matrix, shape_velocities, rhs_shape, preconditioner_mass);

        // TrilinosWrappers::MPI::Vector foo(this_cpu_set, mpi_communicator);
        // Mass_Matrix.vmult(foo, shape_velocities);
        // std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
        //   DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);//StaticMappingQ1<dim-1, dim>::mapping
        //
        // for(unsigned int i =0; i<dh_stokes.n_dofs()/dim; ++i)
        //   if(support_points[i][0]<-0.9)
        //     pcout<<support_points[i]<<" "<<shape_velocities[i]<<" "<<shape_velocities[i+dh_stokes.n_dofs()/dim]<<" "<<shape_velocities[i+dh_stokes.n_dofs()/dim*2]<<std::endl;
        //
        // std::vector<Point<dim> > support_points_map(map_dh.n_dofs());
        //   DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, map_dh, support_points_map);//StaticMappingQ1<dim-1, dim>::mapping
        // pcout<<std::endl;
        // for(unsigned int i =0; i<dh_stokes.n_dofs()/dim; ++i)
        //   if(support_points[i][0]<-0.9)
        //     pcout<<support_points_map[i]<<" "<<next_euler_vec[i]-euler_vec[i]<<" "<<next_euler_vec[i+map_dh.n_dofs()/dim]-euler_vec[i+map_dh.n_dofs()/dim]<<" "<<next_euler_vec[i+map_dh.n_dofs()/dim*2]-euler_vec[i+map_dh.n_dofs()/dim*2]<<std::endl;


      }
  }

// From filename we read the grid on the next frame and we compute both the euler_vector to map the tria on the updated position and
// the velocity.
  template <int dim>
  void BEMProblem<dim>::compute_euler_vector(Vector<double> &euler, unsigned int frame, bool consider_displacements)
  {
    euler = 0;
    double theta = ((double) frame)/n_frames;
    if (use_flagellum_handler)
      flagellum_handler.compute_euler_at_theta(euler, reference_euler_vec, theta);
    else if (build_sphere_in_deal == true)
      {
        VectorTools::get_position_vector(map_dh,euler);
        for (unsigned int i=0; i<euler.size()/dim; ++i)
          {
            if (body_cpu_set.is_element(i))
              {
                Tensor<1,dim> p;
                for (unsigned int d=0; d<dim; ++d)
                  p[d] = euler(i+d*euler.size()/dim);
                if (p[0]>1.)
                  {
                    p[0]-=spheres_distance;
                    p /= p.norm();
                    p[0]+=spheres_distance;
                  }
                else
                  p /= p.norm();
                for (unsigned int d=0; d<dim; ++d)
                  euler(i+d*euler.size()/dim) = p[d];
              }
          }
        pcout << "Fixed euler vector with exact norm of support points" << std::endl;
      }
    else
      {
        Triangulation<dim-1, dim> frame_tria;
        DoFHandler<dim-1, dim> frame_map_dh(frame_tria);

        std::string filename = input_grid_path+input_grid_base_name+Utilities::int_to_string(frame)+"."+input_grid_format;
        dpcout << "Analyzing file " << filename << std::endl;
        pcout << "Analyzing frame = "<< frame << " over " << n_frames << std::endl;
        std::ifstream in;
        in.open (filename);
        read_input_mesh_file(frame, frame_tria);
        // }

        if (apply_iges_to_frame)
          {
            std::string flagellum_frame_filename;
            flagellum_frame_filename=input_iges_file_1+Utilities::int_to_string(frame)+".iges";
            apply_flagellum_iges(frame_tria, flagellum_frame_filename);
            std::ifstream history("mesh.history");
            frame_tria.load_refine_flags(history);
            frame_tria.execute_coarsening_and_refinement();
          }
        if (remove_tria_anisotropies)
          {
            GridTools::remove_anisotropy(frame_tria);
          }
        // for(i=0; i<num_walls)
        //   insert_wall(frame_tria,..)
        // refine_walls()
        // if(grid_type == "Real")
        {
          if (!create_box_bool)
            for (unsigned int wall_number=0; wall_number<wall_bool.size(); ++wall_number)
              {
                if (wall_bool[wall_number]==true)
                  {
                    add_wall_to_tria(frame_tria,wall_number);
                  }
              }
          else
            {
              for (unsigned int wall_number=0; wall_number<first_index_box; ++wall_number)
                {
                  if (wall_bool[wall_number]==true)
                    {
                      add_wall_to_tria(frame_tria,wall_number);
                    }
                }
              for (unsigned int wall_number=first_index_box+6; wall_number<wall_bool.size(); ++wall_number)
                {
                  if (wall_bool[wall_number]==true)
                    {
                      add_wall_to_tria(frame_tria,wall_number);
                    }
                }
              add_box_to_tria(frame_tria);

            }
          if (cylinder_create_bool || cylinder_import_bool)
            add_cylinder_to_tria(frame_tria, cylinder_manifold_bool);
          refine_walls(frame_tria, refine_distance_from_center, wall_threshold, refinement_center, gradual_wall_refinement);
          remove_hanging_nodes_between_different_material_id(frame_tria);

          std::string euler_filename = "euler_tria_"+ Utilities::int_to_string(frame)+".bin";
          std::ofstream out;
          out.open (euler_filename);
          boost::archive::binary_oarchive oa(out);
          frame_tria.save(oa, 0);
          // std::ofstream euler_ofs;
          // euler_ofs.open(euler_filename, std::ofstream::out);
          // GridOut euler_go;
          // euler_go.write_ucd(frame_tria,euler_ofs);
        }
        frame_map_dh.distribute_dofs(*fe_map);
        DoFRenumbering::component_wise (frame_map_dh);
        // DoFRenumbering::subdomain_wise (frame_map_dh);
        ConstraintMatrix constr_mapp;
        constr_mapp.reinit();
        DoFTools::make_hanging_node_constraints(frame_map_dh,constr_mapp);
        constr_mapp.close();
        //MappingQEulerian<dim-1,Vector<double>, dim>  frame_mappingeul(mapping.get_degree(),euler,frame_map_dh);
        //DoFTools::map_dofs_to_support_points<dim-1, dim>( StaticMappingQ1<dim-1, dim>::mapping, frame_map_dh, frame_support_points);
        // Vector<double> euler(frame_map_dh.n_dofs());
        AssertDimension(euler.size(), frame_map_dh.n_dofs());
        AssertDimension(euler.size(), map_dh.n_dofs());
        // AssertDimension(frame_map_dh.n_dofs(), map_dh.n_dofs());
        VectorTools::get_position_vector(frame_map_dh,euler);
        constr_mapp.distribute(euler);
        // rotation_matrix.print(std::cout);
        // map_body_cpu_set.print(std::cout);
        // if(consider_rotations)
      }

    if (spherical_head)
      {
        Point<dim> el_center;
        el_center[0] = -1.6;
        for (types::global_dof_index i = 0; i<euler.size()/dim; ++i)
          if (euler[i]<=0. && map_body_cpu_set.is_element(i))
            {
              // pcout<<"pippo "<<std::endl;

              Point<dim> pp;
              for (unsigned int d=0; d<dim; ++d)
                pp[d] = euler(i+d*euler.size()/dim)-el_center[d];
              pp /= pp.norm();
              euler(i+0*euler.size()/dim) = 1.6*pp[0]+el_center[0];
              euler(i+1*euler.size()/dim) =  .8*pp[1]+el_center[1];
              euler(i+2*euler.size()/dim) =  .8*pp[2]+el_center[2];

            }

      }

    for (types::global_dof_index i = 0; i<euler.size()/dim; ++i)
      {
        // pcout<<"ACCIDENTI A TE ! "<<std::endl;
        if (map_body_cpu_set.is_element(i))
          {
            Vector<double> pos(dim), new_pos(dim);
            for (unsigned int idim=0; idim<dim; ++idim)
              pos[idim] = euler[i+idim*euler.size()/dim];
            rotation_matrix.vmult(new_pos, pos);
            for (unsigned int idim=0; idim<dim; ++idim)
              euler[i+idim*euler.size()/dim] = new_pos[idim];
            if (consider_displacements)
              {
                euler.sadd(1.,1.,rigid_displacements_for_sim);

              }

          }
        // if(i==10)
        // {
        //   std::cout<<"PIPPO"<<i*dim<<" "<<euler[30]<<" "<<euler[31]<<" "<<euler[32]<<std::endl;;
        //   pos.print(std::cout);
        //   rotation_matrix.print(std::cout);
        //   new_pos.print(std::cout);
        // }
      }

    //  DoFHandler<dim-1, dim> map_dh_unrefined(tria_unrefined);
    //  map_dh_unrefined.distribute_dofs(*fe_map);
    //  DoFRenumbering::component_wise (map_dh_unrefined);
    //  euler = euler;
    //  VectorTools::interpolate_to_different_mesh(map_dh_unrefined, euler, map_dh, euler);
    //  for(auto i : euler.locally_owned_elements())
    //   pcout<<euler[i]<<" : "<<euler[i]<<std::endl;
    pcout<<euler.l2_norm()<<" : "<<euler.l2_norm()<<std::endl;

  }




  // In this function we compute the 3(dim == 2)-6(dim == 3) rigid modes we will need.
  // We compute also the actual position of the center of mass.
  template <int dim>
  void BEMProblem<dim>::compute_center_of_mass_and_rigid_modes(unsigned int frame)
  {
    // MappingQ<dim-1,dim> mapping_torque(1);
    pcout << std::endl << "Computing the center of mass of our system for the frame " << frame << std::endl;
    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_quadrature_points |
                                    update_JxW_values);

    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    FullMatrix<double> local_mass_matrix(fe_stokes->dofs_per_cell,fe_stokes->dofs_per_cell);

    std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
    std::vector<Point<dim> > support_points_tria(dh_stokes.n_dofs());
    std::vector<Point<dim> > support_points_map(map_dh.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, map_dh, support_points_map);
    DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);//StaticMappingQ1<dim-1, dim>::mapping
    // DoFTools::map_dofs_to_support_points<dim-1, dim>( mapping_torque, dh_stokes, support_points_tria);//StaticMappingQ1<dim-1, dim>::mapping
    double mass;
    double mass_body;
    mass=0.;
    mass_body=0.;
    for (unsigned int i =0; i< dim; ++i)
      {
        center_of_mass(i)=0.;
        center_of_mass_body(i)=0.;
      }
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();

    unsigned int comp_i, comp_j;

    for (cell = dh_stokes.begin_active(); cell != endc; ++cell)
      {
        if (cell->subdomain_id() == this_mpi_process)
          {
            fe_stokes_v.reinit(cell);

            const std::vector<Point<dim> > &q_points = fe_stokes_v.get_quadrature_points();
            double n_q_points = q_points.size();

            cell->get_dof_indices(local_dof_indices);
            local_mass_matrix = 0;

            for (unsigned int q=0; q<n_q_points; ++q)
              {
                // unsigned int idim = fe_stokes->system_to_component_index(i).first;
                center_of_mass +=  q_points[q]*fe_stokes_v.JxW(q);
                mass += fe_stokes_v.JxW(q);

                if (cell->material_id() == 0)
                  {
                    center_of_mass_body += q_points[q]*fe_stokes_v.JxW(q);
                    mass_body += fe_stokes_v.JxW(q);
                  }

                for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
                  {
                    comp_i = fe_stokes->system_to_component_index(i).first;

                    for (unsigned int j = 0; j<fe_stokes->dofs_per_cell; ++j)
                      {
                        comp_j = fe_stokes->system_to_component_index(j).first;
                        if (comp_j == comp_i)
                          local_mass_matrix(i,j) += fe_stokes_v.shape_value(i,q)*
                                                    fe_stokes_v.shape_value(j,q)*
                                                    fe_stokes_v.JxW(q);
                      }
                  }

              }
            constraints.distribute_local_to_global
            (local_mass_matrix,
             local_dof_indices,
             Mass_Matrix);
          }
      }
    Mass_Matrix.compress(VectorOperation::add);
    //Reduction of massand center of mass
    double mass_foo = 0.;
    Point<dim> center_of_mass_foo;
    pcout<<mass<<" ";
    MPI_Allreduce(&mass, &mass_foo, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&center_of_mass[0], &center_of_mass_foo[0], dim, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    mass = mass_foo;
    pcout<<mass<<std::endl;

    center_of_mass = center_of_mass_foo;

    dpcout<<"CONTROLLARE STORIA CENTRI DI MASSA!!!"<<std::endl;
    mass_foo=0.;
    for (unsigned int i =0; i<dim; ++i)
      center_of_mass_foo[i]=0.;
    MPI_Allreduce(&mass_body, &mass_foo, 1, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    MPI_Allreduce(&center_of_mass_body[0], &center_of_mass_foo[0], dim, MPI_DOUBLE, MPI_SUM, mpi_communicator);
    mass_body = mass_foo;
    pcout<<"The Mass (Surface) of the entire system is : "<<mass_body<<std::endl;
    if (grid_type == "ImposedForce" || grid_type == "Convergence" || grid_type == "ImposedVelocity")
      {
        double exact_mass = 4.*numbers::PI;
        pcout<<"The Mass (Surface) error of the entire system is : "<<std::fabs(mass_body-exact_mass)/exact_mass<<std::endl;
      }
    center_of_mass_body = center_of_mass_foo;

    for (unsigned int i =0; i< dim; ++i)
      center_of_mass_body(i)=center_of_mass_body(i) / mass_body;

    Point<dim> origin;
    point_force_pole=origin;

    if (force_pole=="Baricenter")
      point_force_pole=center_of_mass_body;
    else if (force_pole=="Point")
      point_force_pole=force_arbitrary_point;


    // We can compute the rigid modes with respect to the chosen force pole. We compute the modes for the two finite elements (stokes and mapping), relative to the swimmer (for computational purposes) and complete (for debugging for the K_matrix)
    pcout<<"Using "<<point_force_pole<<" as force pole."<<std::endl;
    if (dim == 2)
      {
        for (unsigned int i = 0; i < dh_stokes.n_dofs()/dim; ++i)
          {
            if (this_cpu_set.is_element(i))
              {
                if (body_cpu_set.is_element(i)) //.is_element(i)
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                        N_rigid_complete[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                      }
                    N_rigid[dim](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid[dim](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_complete[dim](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_complete[dim](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                  }
                else
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_complete[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                      }
                    N_rigid_complete[dim](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_complete[dim](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                  }
              }
          }
        for (unsigned int i = 0; i < map_dh.n_dofs()/dim; ++i)
          {
            if (map_cpu_set.is_element(i))
              {
                if (map_body_cpu_set.is_element(i)) ////map_flagellum_cpu_set
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_map[j](i+j*map_dh.n_dofs()/dim) = 1.;
                        N_rigid_map_complete[j](i+j*map_dh.n_dofs()/dim) = 1.;
                      }
                    N_rigid_map[dim](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map[dim](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_map_complete[dim](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map_complete[dim](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                  }
                else
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_map_complete[j](i+j*map_dh.n_dofs()/dim) = 1.;
                      }
                    N_rigid_map_complete[dim](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map_complete[dim](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                  }
              }
          }
      }
    else if (dim == 3)
      {
        for (unsigned int i = 0; i < dh_stokes.n_dofs()/dim; ++i)
          {
            if (this_cpu_set.is_element(i))
              {
                if (body_cpu_set.is_element(i))//flagellum_cpu_set
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                        N_rigid_complete[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                      }
                    N_rigid[dim + 0](i+0*dh_stokes.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid[dim + 0](i+1*dh_stokes.n_dofs()/dim) = - (support_points[i+1*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid[dim + 0](i+2*dh_stokes.n_dofs()/dim) = + (support_points[i+2*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));

                    N_rigid[dim + 1](i+0*dh_stokes.n_dofs()/dim) = + (support_points[i+0*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid[dim + 1](i+1*dh_stokes.n_dofs()/dim) = 0;
                    N_rigid[dim + 1](i+2*dh_stokes.n_dofs()/dim) = - (support_points[i+2*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid[dim + 2](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid[dim + 2](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid[dim + 2](i+2*dh_stokes.n_dofs()/dim) = 0;


                    N_rigid_complete[dim + 0](i+0*dh_stokes.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid_complete[dim + 0](i+1*dh_stokes.n_dofs()/dim) = - (support_points[i+1*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_complete[dim + 0](i+2*dh_stokes.n_dofs()/dim) = + (support_points[i+2*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));


                    N_rigid_complete[dim + 1](i+0*dh_stokes.n_dofs()/dim) = + (support_points[i+0*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_complete[dim + 1](i+1*dh_stokes.n_dofs()/dim) = 0;
                    N_rigid_complete[dim + 1](i+2*dh_stokes.n_dofs()/dim) = - (support_points[i+2*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid_complete[dim + 2](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_complete[dim + 2](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_complete[dim + 2](i+2*dh_stokes.n_dofs()/dim) = 0;

                    if (flagellum_cpu_set.is_element(i))
                      {
                        Vector<double> supp_point_1(dim), supp_point_2(dim);
                        for (unsigned int idim = 0; idim<dim; ++idim)
                          {
                            Assert(support_points[i+idim*dh_stokes.n_dofs()/dim] == support_points[i], ExcMessage("PIRLA SCHIFOSO"));
                            supp_point_1[idim] = support_points[i+idim*dh_stokes.n_dofs()/dim][idim];
                            rotation_matrix.Tvmult(supp_point_2, supp_point_1);
                          }

                        N_flagellum_torque(i+0*dh_stokes.n_dofs()/dim) = 0;//) - point_force_pole(2));
                        N_flagellum_torque(i+1*dh_stokes.n_dofs()/dim) = - (supp_point_2[2] - point_force_pole(2));
                        N_flagellum_torque(i+2*dh_stokes.n_dofs()/dim) = + (supp_point_2[1] - point_force_pole(1));
                        N_flagellum_translation(i)=1.;
                      }

                    // pcout<<support_points[i]<<std::endl;
                  }
                else
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_complete[j](i+j*dh_stokes.n_dofs()/dim) = 1.;
                      }
                    N_rigid_complete[dim + 0](i+0*dh_stokes.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid_complete[dim + 0](i+1*dh_stokes.n_dofs()/dim) = - (support_points[i+1*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_complete[dim + 0](i+2*dh_stokes.n_dofs()/dim) = + (support_points[i+2*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));

                    N_rigid_complete[dim + 1](i+0*dh_stokes.n_dofs()/dim) = + (support_points[i+0*dh_stokes.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_complete[dim + 1](i+1*dh_stokes.n_dofs()/dim) = 0;
                    N_rigid_complete[dim + 1](i+2*dh_stokes.n_dofs()/dim) = - (support_points[i+2*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid_complete[dim + 2](i+0*dh_stokes.n_dofs()/dim) = - (support_points[i+0*dh_stokes.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_complete[dim + 2](i+1*dh_stokes.n_dofs()/dim) = + (support_points[i+1*dh_stokes.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_complete[dim + 2](i+2*dh_stokes.n_dofs()/dim) = 0;
                  }
              }
          }
        for (unsigned int i = 0; i < map_dh.n_dofs()/dim; ++i)
          {
            if (map_cpu_set.is_element(i))
              {
                if (map_body_cpu_set.is_element(i)) ////map_flagellum_cpu_set
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_map[j](i+j*map_dh.n_dofs()/dim) = 1.;
                        N_rigid_map_complete[j](i+j*map_dh.n_dofs()/dim) = 1.;
                      }
                    N_rigid_map[dim + 0](i+0*map_dh.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid_map[dim + 0](i+1*map_dh.n_dofs()/dim) = - (support_points_map[i+1*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map[dim + 0](i+2*map_dh.n_dofs()/dim) = + (support_points_map[i+2*map_dh.n_dofs()/dim][1] - point_force_pole(1));

                    N_rigid_map[dim + 1](i+0*map_dh.n_dofs()/dim) = + (support_points_map[i+0*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map[dim + 1](i+1*map_dh.n_dofs()/dim) = 0;
                    N_rigid_map[dim + 1](i+2*map_dh.n_dofs()/dim) = - (support_points_map[i+2*map_dh.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid_map[dim + 2](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map[dim + 2](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_map[dim + 2](i+2*map_dh.n_dofs()/dim) = 0;



                    N_rigid_map_complete[dim + 0](i+0*map_dh.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid_map_complete[dim + 0](i+1*map_dh.n_dofs()/dim) = - (support_points_map[i+1*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map_complete[dim + 0](i+2*map_dh.n_dofs()/dim) = + (support_points_map[i+2*map_dh.n_dofs()/dim][1] - point_force_pole(1));

                    N_rigid_map_complete[dim + 1](i+0*map_dh.n_dofs()/dim) = + (support_points_map[i+0*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map_complete[dim + 1](i+1*map_dh.n_dofs()/dim) = 0;
                    N_rigid_map_complete[dim + 1](i+2*map_dh.n_dofs()/dim) = - (support_points_map[i+2*map_dh.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid_map_complete[dim + 2](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map_complete[dim + 2](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_map_complete[dim + 2](i+2*map_dh.n_dofs()/dim) = 0;

                  }
                else
                  {
                    for (unsigned int j = 0; j < dim; ++j)
                      {
                        N_rigid_map_complete[j](i+j*map_dh.n_dofs()/dim) = 1.;
                      }

                    N_rigid_map_complete[dim + 0](i+0*map_dh.n_dofs()/dim) = 0;//) - point_force_pole(2));
                    N_rigid_map_complete[dim + 0](i+1*map_dh.n_dofs()/dim) = - (support_points_map[i+1*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map_complete[dim + 0](i+2*map_dh.n_dofs()/dim) = + (support_points_map[i+2*map_dh.n_dofs()/dim][1] - point_force_pole(1));

                    N_rigid_map_complete[dim + 1](i+0*map_dh.n_dofs()/dim) = + (support_points_map[i+0*map_dh.n_dofs()/dim][2] - point_force_pole(2));
                    N_rigid_map_complete[dim + 1](i+1*map_dh.n_dofs()/dim) = 0;
                    N_rigid_map_complete[dim + 1](i+2*map_dh.n_dofs()/dim) = - (support_points_map[i+2*map_dh.n_dofs()/dim][0] - point_force_pole(0));

                    N_rigid_map_complete[dim + 2](i+0*map_dh.n_dofs()/dim) = - (support_points_map[i+0*map_dh.n_dofs()/dim][1] - point_force_pole(1));
                    N_rigid_map_complete[dim + 2](i+1*map_dh.n_dofs()/dim) = + (support_points_map[i+1*map_dh.n_dofs()/dim][0] - point_force_pole(0));
                    N_rigid_map_complete[dim + 2](i+2*map_dh.n_dofs()/dim) = 0;

                  }
              }
          }
      }





    for (unsigned  int i = 0; i<num_rigid; ++i)
      {
        N_rigid[i].compress(VectorOperation::insert);
        N_rigid_complete[i].compress(VectorOperation::insert);
        N_rigid_map[i].compress(VectorOperation::insert);
        N_rigid_map_complete[i].compress(VectorOperation::insert);
      }
    N_flagellum_translation.compress(VectorOperation::insert);
    N_flagellum_torque.compress(VectorOperation::insert);
    rotate_vector(N_flagellum_torque);
    for (unsigned int i=0; i<num_rigid; ++i)
      {
        Mass_Matrix.vmult(N_rigid_dual[i],N_rigid[i]);  //N_rigid_dual[i]=N_rigid[i];//
        Mass_Matrix.vmult(N_rigid_dual_complete[i],N_rigid_complete[i]);
      }
    Mass_Matrix.vmult(N_flagellum_torque_dual,N_flagellum_torque);
    if (this_mpi_process == 0)
      {
        std::ofstream ofs;
        ofs.open ("center_of_mass_position.txt", std::ofstream::out | std::ofstream::app);
        ofs << frame << " " << center_of_mass_body << std::endl;;
        ofs.close();
        pcout << "Center of mass position = " << center_of_mass_body <<std::endl;
      }



  }


  // template <int dim>
  // void BEMProblem<dim>::assemble_preconditioner()
  // {
  //   pcout<<"Initialising preconditioner"<<std::endl;
  //   prec_data.n_cycles = 1;
  //   prec_data.smoother_type = "Amesos-UMFPACK";//"Amesos-Superludist";//"Amesos-UMFPACK";//;"IFPACK";//;"Aztec";//"MLS"; //"self";
  //   prec_data.aggregation_threshold = 0.1;
  //   prec_data.smoother_sweeps = 3;
  //   prec_data.output_details = false;
  //   prec_data.coarse_type = "Amesos-UMFPACK";
  //   prec_data.constant_modes.clear();
  //   prec_data.constant_modes.resize(dim);
  //   for (unsigned int i = 0; i<dim; ++i)
  //     {
  //       prec_data.constant_modes[i].resize(dh_stokes.n_dofs(),false);
  //       for (types::global_dof_index j = dh_stokes.n_dofs()/dim * i; j<dh_stokes.n_dofs()/dim * (i+1); ++j)
  //         prec_data.constant_modes[i][j] = true;
  //
  //     }
  //   preconditioner.initialize(V_matrix, prec_data);
  //
  // }

  // In this function we compute the global component(from 0 to dim-1) of the local degree of freedom.
  template <int dim>
  void BEMProblem<dim>::compute_global_components()
  {
    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values);

    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();

    for (cell = dh_stokes.begin_active(); cell != endc; ++cell)
      {
        fe_stokes_v.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int j = 0; j<fe_stokes_v.dofs_per_cell; ++j)
          {
            unsigned int jdim = fe_stokes->system_to_component_index(j).first;
            global_components(local_dof_indices[j])=jdim;
          }
      }
  }

  // Assemble of a stokes system solver based on collocation method.
  template <int dim>
  void BEMProblem<dim>::assemble_stokes_system(bool correction_on_V)
  {

    Teuchos::TimeMonitor LocalTimer(*AssembleTime);
    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_cell_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);
    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    FullMatrix<double>    local_single_layer(dim, fe_stokes->dofs_per_cell);
    FullMatrix<double>    local_double_layer(dim, fe_stokes->dofs_per_cell);
    FullMatrix<double>    local_hypersingular(dim, fe_stokes->dofs_per_cell);
    std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);

    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();
    unsigned int kernel_wall_orientation=numbers::invalid_unsigned_int;
    for (unsigned int i=0; i<dim; ++i)
      {
        if (wall_spans[0][i]==0)
          kernel_wall_orientation=i;
      }
    fs_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    fs_exterior_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    ns_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    ns_exterior_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    for (cell = dh_stokes.begin_active(); cell != endc; ++cell)
      {
        fe_stokes_v.reinit(cell);
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i=0; i<(dh_stokes.n_dofs() / dim) ; ++i)
          {
            if (this_cpu_set.is_element(i))
              {
                if ( constraints.is_constrained(i)==false )
                  {
                    local_single_layer = 0.;
                    local_double_layer = 0.;
                    bool is_singular = false;
                    unsigned int singular_index = numbers::invalid_unsigned_int;
                    for (unsigned int j=0; j<fe_stokes->dofs_per_cell; ++j)
                      {
                        unsigned int jdim = fe_stokes->system_to_component_index(j).first;
                        if (jdim == 0 && local_dof_indices[j] == i)
                          {
                            is_singular = true;
                            singular_index = j;
                          }
                        if (is_singular)
                          break;
                      }

                    FEValues<dim-1,dim> * internal_fe_v;
                    if (is_singular == true)
                      {
                        Assert(singular_index != numbers::invalid_unsigned_int,
                               ExcInternalError());
                        internal_fe_v  = & get_singular_fe_values(singular_index);// should be correct since they should be 3 equal support_points
                        internal_fe_v->reinit(cell);
                      }
                    else
                      {
                        internal_fe_v = & fe_stokes_v;
                      }

                    const std::vector<Point<dim> > &q_points = internal_fe_v->get_quadrature_points();
                    const std::vector<Tensor<1, dim> > &normals  = internal_fe_v->get_normal_vectors();

                    unsigned int n_q_points = q_points.size();

                    for (unsigned int q=0; q<n_q_points; ++q)
                      {
                        const Tensor<1,dim> R = q_points[q] - support_points[i];
                        Point<dim> support_point_image(support_points[i]);
                        support_point_image[kernel_wall_orientation] -= 2*(support_points[i][kernel_wall_orientation]-wall_positions[0][kernel_wall_orientation]);
                        const Tensor<1,dim> R_image = q_points[q] - support_point_image;

                        Tensor<2,dim> G = compute_G_kernel(R,R_image,stokes_kernel,fs_stokes_kernel,ns_stokes_kernel,reflect_kernel,no_slip_kernel); //stokes_kernel.value_tens(R);
                        Tensor<3,dim> W = compute_W_kernel(R,R_image,stokes_kernel,fs_stokes_kernel,ns_stokes_kernel,reflect_kernel,no_slip_kernel); //stokes_kernel.value_tens2(R);

                        // Tensor<4,dim> D = stokes_kernel.value_tens3(R);
                        Tensor<2, dim> singular_ker = compute_singular_kernel(normals[q], W);

                        for (unsigned int idim=0; idim < dim; ++idim)
                          {
                            Assert(support_points[i].distance(support_points[i+idim*dh_stokes.n_dofs()/dim])<1e-10,
                                   ExcInternalError());

                            for (unsigned int j=0; j<fe_stokes->dofs_per_cell; ++j)
                              {

                                unsigned int jdim = fe_stokes->system_to_component_index(j).first;
                                local_single_layer(idim, j) += ( G[idim][jdim] * //(1-2*use_internal_alpha)*
                                                                 internal_fe_v->shape_value(j,q)     *
                                                                 internal_fe_v->JxW(q)       );
                                // THE MINUS IS FOR THE EXTERIOR PROBLEM STORY! THE CHANGING SIGN WITH RESPECT TO THE THEORY FOR THE SINGLE LAYER
                                // COMES FROM THE ACTION/REACTION PRINCIPLE=>WE ARE RECOVERING THE FORCES ACTING ON THE BODY (NOT ON THE FLUID), THUS
                                // WE NEED THE OPPOSITE NORMAL OR THE OPPOSITE SIGN.
                                local_double_layer(idim, j) -= (singular_ker[idim][jdim] * //(2*use_internal_alpha-1)*
                                                                (internal_fe_v->shape_value(j,q)) * // - (+cval)) * //internal_fe_v->shape_value(j, support_points[i])) *
                                                                internal_fe_v->JxW(q));
                                // local_hypersingular(idim, j) += (hypersingular_ker[idim][jdim] *
                                //                                  (internal_fe_v->shape_value(j,q)) *
                                //                                  internal_fe_v->JxW(q));
                              }
                          }
                      }

                    std::vector<types::global_dof_index> local_dof_indices_row(dim);
                    for (unsigned int idim=0; idim<dim; ++idim)
                      {
                        // local_dof_indices_row[idim]=i+dh_stokes.n_dofs()/dim*idim;
                        for (unsigned int j=0; j<fe_stokes->dofs_per_cell; ++j)
                          {
                            V_matrix.add(i+dh_stokes.n_dofs()/dim*idim,local_dof_indices[j],local_single_layer(idim,j));
                            K_matrix.add(i+dh_stokes.n_dofs()/dim*idim,local_dof_indices[j],local_double_layer(idim,j));
                            // D_matrix.add(i+dh_stokes.n_dofs()/dim*idim,local_dof_indices[j],local_hypersingular(idim,j));
                          }
                      }
                    // constraints.distribute_local_to_global(local_single_layer,local_dof_indices_row,local_dof_indices,V_matrix);
                    // constraints.distribute_local_to_global(local_double_layer,local_dof_indices_row,local_dof_indices,K_matrix);
                    // constraints.distribute_local_to_global(local_hypersingular,local_dof_indices_row,local_dof_indices,D_matrix);

                  }
                // We need to manually take care of the constraints because of the structure of the assembling cycle based on the dim dimensionality. If we simply use distribute local to global only the first components is complete. The other dim-1 lines remains empties.
                else
                  {
                    // We manually take care of every constrained component. Since we have dim scalar unknowns we perform all the stuff in a dim cycle.
                    for (unsigned int idim=0; idim<dim; ++idim)
                      {
                        auto ii=i+dh_stokes.n_dofs()/dim*idim;
                        std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(ii);
                        V_matrix.add(ii,ii,-V_matrix(ii,ii));
                        V_matrix.add(ii,ii,1.);
                        K_matrix.add(ii,ii,-K_matrix(ii,ii));
                        K_matrix.add(ii,ii,1.);
                        // D_matrix.add(ii,ii,-D_matrix(ii,ii));
                        // D_matrix.add(ii,ii,1.);
                        for (unsigned int ic=0; ic<constr_in.size(); ++ic)
                          {
                            V_matrix.add(ii,constr_in[ic].first,-V_matrix(ii,constr_in[ic].first));
                            V_matrix.add(ii,constr_in[ic].first,-constr_in[ic].second);
                            K_matrix.add(ii,constr_in[ic].first,-K_matrix(ii,constr_in[ic].first));
                            K_matrix.add(ii,constr_in[ic].first,-constr_in[ic].second);
                            // D_matrix.add(ii,constr_in[ic].first,-D_matrix(ii,constr_in[ic].first));
                            // D_matrix.add(ii,constr_in[ic].first,-constr_in[ic].second);

                          }
                      }

                  }
              }
          }
      }
    V_matrix.compress(VectorOperation::add);
    K_matrix.compress(VectorOperation::add);
    // D_matrix.compress(VectorOperation::add);

    // Now we can perform the normal test on the single layer operator V_matrix
    V_matrix.vmult(V_x_normals, normal_vector);
    V_matrix.vmult(V_x_normals_body, normal_vector_pure);
    constraints.close();
    constraints.distribute(V_x_normals);
    constraints.distribute(V_x_normals_body);

    TrilinosWrappers::MPI::Vector foo(this_cpu_set, mpi_communicator);
    tangential_projector_body(normal_vector_pure, foo);
    pcout << "Check on the tangential projection (should be zero): " << foo.linfty_norm() << std::endl;
    pcout << "Check on the V operator Norm (should be zero): " << V_x_normals.linfty_norm() << std::endl;
    pcout << "Check on the V operator Norm (should be zero) pure: " << V_x_normals_body.linfty_norm() << std::endl;

    // Now we can correct the single layer using the normal vector and a Grahm-Schmidt like procedure.
    if (correction_on_V)
      {
        Vector<double> local_M_normal_vector_pure(M_normal_vector_pure);
        // Vector<double> local_normal_vector(normal_vector);
        for (auto i : this_cpu_set)
          {
            // We correct only if we don't have constraints. Otherwise we destroy the effects of having the constraints.
            if (!constraints.is_constrained(i))
              {
                for (unsigned int j=0; j<dh_stokes.n_dofs(); ++j)
                  // V_matrix(i,j) += first_evec(i) * M_first_evec(j)/l2normGamma_evec;
                  V_matrix.add(i,j,(-V_x_normals_body[i]+1.*normal_vector_pure(i)) * local_M_normal_vector_pure(j)/l2normGamma_pure);
              }
          }
        V_matrix.compress(VectorOperation::add);
      }
    pcout<<"Corrected V"<<std::endl;
    // pcout<<M_normal_vector_pure*normal_vector_pure/l2normGamma_pure<<std::endl;

    V_matrix.vmult(V_x_normals_body, normal_vector_pure);

    // The following check works correctly only if we don't have any constraint.
    pcout << "Check on the V operator Norm post (should be one) pure: " << V_x_normals_body *normal_vector_pure/(body_cpu_set.n_elements()/dim) << std::endl;



    // At this point we cant try to correct the double layer to include the solid angle. We need the three versors.
    std::vector<TrilinosWrappers::MPI::Vector > VersorMatrix(dim);
    std::vector<TrilinosWrappers::MPI::Vector > CheckMatrix_2(dim);
    CheckMatrix.resize(dim);

    for (unsigned int i = 0; i<dim; ++i)
      {
        VersorMatrix[i].reinit(this_cpu_set, mpi_communicator);
        CheckMatrix[i].reinit(this_cpu_set, mpi_communicator);
        CheckMatrix_2[i].reinit(this_cpu_set, mpi_communicator);
      }

    pcout<<"building versor matrix"<<std::endl;
    for (unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
      {
        if (this_cpu_set.is_element(i))
          for (unsigned int j=0; j<dim; ++j)
            VersorMatrix[j][i+j*dh_stokes.n_dofs()/dim]=1.;
      }
    for (unsigned int i = 0; i<dim; ++i)
      {
        VersorMatrix[i].compress(VectorOperation::insert);
        constraints.distribute(VersorMatrix[i]);
      }

    for (unsigned int i = 0; i<dim; ++i)
      {
        K_matrix.vmult(CheckMatrix[i],VersorMatrix[i]);
        constraints.distribute(CheckMatrix[i]);
      }

    pcout<<"correcting K matrix"<<std::endl;

    for (unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
      {
        if (this_cpu_set.is_element(i) && !constraints.is_constrained(i))
          for (unsigned int j=0; j<dim; ++j)
            {
              for (unsigned int k=0; k<dim; ++k)
                {
                  // if(use_internal_alpha)
                  K_matrix.add(i+j*dh_stokes.n_dofs()/dim,i+k*dh_stokes.n_dofs()/dim,-CheckMatrix[k][i+j*dh_stokes.n_dofs()/dim]);
                  if (j == k && !(use_internal_alpha))
                    K_matrix.add(i+j*dh_stokes.n_dofs()/dim,i+k*dh_stokes.n_dofs()/dim,1.);

                }
            }
      }

    K_matrix.compress(VectorOperation::add);
    for (unsigned int i = 0; i<dim; ++i)
      {
        K_matrix.vmult(CheckMatrix_2[i],VersorMatrix[i]);
        constraints.distribute(CheckMatrix_2[i]);
        pcout<<"check with versor vector : "<< i <<" l_infty : "<<CheckMatrix_2[i].linfty_norm()<<std::endl;
      }
    // In principle we can correct the double layer even further to account for all its properties on rigid rotations.
    // At the time being it does not work.
    // if (false)
    //   {
    //     pcout<<"correcting K matrix for rotations using Gram–Schmidt "<<std::endl;
    //     TrilinosWrappers::MPI::Vector foo_vel(this_cpu_set,mpi_communicator);
    //     TrilinosWrappers::MPI::Vector foo_vel_2(this_cpu_set,mpi_communicator);
    //     double eigen_factor = 0.;
    //     if (!use_internal_alpha)
    //       eigen_factor=1.;
    //     for (unsigned int i=0; i<num_rigid-dim; ++i)
    //       {
    //         Vector<double> local_ith_N_rigid_dual(N_rigid_dual[i+dim]);
    //         K_matrix.vmult(foo_vel_2,N_rigid[i+dim]);
    //         K_matrix.vmult(foo_vel,N_rigid_complete[i+dim]);
    //         pcout<<foo_vel_2.linfty_norm()<<" "<<foo_vel.linfty_norm()<<std::endl;
    //         double normaliser=N_rigid_dual[i+dim]*N_rigid[i+dim];
    //       }
    //     K_matrix.compress(VectorOperation::add);
    //   }

    std::vector<TrilinosWrappers::MPI::Vector> tmp_N;
    std::vector<TrilinosWrappers::MPI::Vector> tmp_N_2;
    tmp_N.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    tmp_N_2.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));

    TrilinosWrappers::MPI::Vector tmp_shape_vel(this_cpu_set, mpi_communicator), tmp_shape_vel2(this_cpu_set, mpi_communicator);
    TrilinosWrappers::MPI::Vector tmp_flagellum(this_cpu_set,mpi_communicator);
    tangential_projector_body(shape_velocities, tmp_shape_vel2);
    tmp_shape_vel = tmp_shape_vel2;
    K_matrix.vmult(tmp_shape_vel2, tmp_shape_vel);
    tangential_projector_body(tmp_shape_vel2, tmp_shape_vel);
    // tmp_shape_vel = tmp_shape_vel2;
    constraints.distribute(tmp_shape_vel);

    for (unsigned int i=0; i<num_rigid; ++i)
      {
        tangential_projector_body(N_rigid[i], tmp_N[i]);
        // pcout<<N_rigid[i].l2_norm()<<" RIGID 1 "<<tmp_N[i].l2_norm()<<" ";
        K_matrix.vmult(tmp_N_2[i], tmp_N[i]);
        // pcout<<tmp_N_2[i].l2_norm()<<"  ";
        tangential_projector_body(tmp_N_2[i], tmp_N[i]);
        constraints.distribute(tmp_N[i]);
        // pcout<<tmp_N[i].l2_norm()<<std::endl;

      }
    TrilinosWrappers::MPI::Vector tmp_flagellum_2(this_cpu_set,mpi_communicator);
    tangential_projector_body(N_flagellum_torque, tmp_flagellum);
    K_matrix.vmult(tmp_flagellum_2, tmp_flagellum);
    tangential_projector_body(tmp_flagellum_2, tmp_flagellum);

    pcout<<"monolithic building"<<std::endl;
    Vector<double> local_normal_vector(normal_vector);
    if (monolithic_bool)
      {
        for (auto i : this_cpu_set)
          {
            if (constraints.is_constrained(i))
              {
                std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);

                monolithic_system_matrix.set(i,i,1.);
                // monolithic_system_preconditioner_matrix.set(i,i,1.);
                monolithic_rhs[i] = 0.;

                for (unsigned int ic=0; ic<constr_in.size(); ++ic)
                  {
                    // We copy the constrained entries of the corresponding matrix.
                    // if (body_cpu_set.is_element(constr_in[ic].first) || wall_no_slip_cpu_set.is_element(constr_in[ic].first))
                    //   {
                    //     monolithic_system_matrix.set(i,constr_in[ic].first,V_matrix(i,constr_in[ic].first));
                    //     monolithic_system_preconditioner_matrix.set(i,constr_in[ic].first,V_matrix(i,constr_in[ic].first));
                    //
                    //   }
                    // else if (wall_free_surface_cpu_set.is_element(constr_in[ic].first))
                    //   {
                    //     monolithic_system_matrix.set(i,constr_in[ic].first,K_matrix(i,constr_in[ic].first));
                    //     monolithic_system_preconditioner_matrix.set(i,constr_in[ic].first,K_matrix(i,constr_in[ic].first));
                    //
                    //   }
                    monolithic_system_matrix.set(i,constr_in[ic].first,-constr_in[ic].second);
                    // monolithic_system_preconditioner_matrix.set(i,constr_in[ic].first,-constr_in[ic].second);

                  }
              }
            else
              {
                // If we solve a swimming proble the rhs is given by the shape velocities.
                if (grid_type != "Real")
                  monolithic_rhs[i] = 0.;
                else
                  monolithic_rhs[i] = tmp_shape_vel[i];
                if (solve_with_torque)
                  monolithic_rhs[i] = 0.;
                // If we are on the body the unknowns are the stresses.
                for (auto j : body_cpu_set)
                  {
                    monolithic_system_matrix.set(i,j,V_matrix(i,j));
                    // monolithic_system_preconditioner_matrix.set(i,j,V_matrix(i,j));
                  }
                // If we are on the no spil wall the unknowns are the stresses.
                for (auto j : wall_no_slip_cpu_set)
                  {
                    monolithic_system_matrix.set(i,j,V_matrix(i,j));
                    // monolithic_system_preconditioner_matrix.set(i,j,V_matrix(i,j));
                  }
                // If we are on the free surface wall the unknowns are the stresses in the normal direction and the velocity otherwise.
                for (auto j : wall_free_surface_cpu_set)
                  {
                    if (std::abs(std::abs(local_normal_vector(j))-1) < 5e-2)
                      {
                        monolithic_system_matrix.set(i,j,V_matrix(i,j));
                        // monolithic_system_preconditioner_matrix.set(i,j,V_matrix(i,j));
                      }
                    else
                      {
                        monolithic_system_matrix.set(i,j,-K_matrix(i,j));
                        // monolithic_system_preconditioner_matrix.set(i,j,-K_matrix(i,j));
                      }
                  }
                // If we are on the do nothing wall the unknowns are the stresses in the tangential directions and the velocity otherwise.
                for (auto j : wall_do_nothing_cpu_set)
                  {
                    if (std::abs(std::abs(local_normal_vector(j))-1) < 5e-2)
                      {
                        monolithic_system_matrix.set(i,j,-K_matrix(i,j));
                        // monolithic_system_preconditioner_matrix.set(i,j,-K_matrix(i,j));
                      }
                    else
                      {
                        monolithic_system_matrix.set(i,j,V_matrix(i,j));
                        // monolithic_system_preconditioner_matrix.set(i,j,V_matrix(i,j));

                      }
                  }
                // The dirichlet wall is a no slip wall representing the infinity
                for (auto j : wall_dirichlet_cpu_set)
                  {
                    monolithic_system_matrix.set(i,j,V_matrix(i,j));
                    // monolithic_system_preconditioner_matrix.set(i,j,V_matrix(i,j));
                  }
                // The neumann wall is a neumann wall representing the infinity
                for (auto j : wall_neumann_cpu_set)
                  {
                    monolithic_system_matrix.set(i,j,-K_matrix(i,j));
                    // monolithic_system_preconditioner_matrix.set(i,j,-K_matrix(i,j));
                  }
                // for what concerns the rigid movements we can apply a scaling to achieve a better conditioning of the matrices. We apply the scaling factor here ans on the rhs. Consequently even the solution will be rescaled.
                for (types::global_dof_index j = dh_stokes.n_dofs(); j < dh_stokes.n_dofs()+num_rigid; ++j)
                  {
                    monolithic_system_matrix.set(i,j,-assemble_scaling*tmp_N[j-dh_stokes.n_dofs()][i]);
                    // monolithic_system_preconditioner_matrix.set(i,j,-assemble_scaling*tmp_N[j-dh_stokes.n_dofs()][i]);
                  }
                if (solve_with_torque)
                  {

                    monolithic_system_matrix.set(i,dh_stokes.n_dofs()+num_rigid,-assemble_scaling*tmp_flagellum[i]);
                    // monolithic_system_preconditioner_matrix.set(i,dh_stokes.n_dofs()+num_rigid,-assemble_scaling*tmp_flagellum[i]);
                  }

              }
            // if(wall_no_slip_cpu_set.is_element(i))
            // {
            //   if(constraints.is_constrained(i))
            //     pcout<<"CONSTRAINED "<<i<<" ";
            //   else
            //     pcout<<"NON-CONSTRAINED "<<i<<" ";
            //
            //   if(monolithic_system_matrix(i,0)==0)
            //     // pcout<<"ZEROS";
            //     for(auto j : this_cpu_set)
            //       pcout<<V_matrix(i,j)<<" ";
            //   pcout<<std::endl;
            // }


          }

        // We can impose either a rigid movement or an external force. We can also rotate it
        Point<dim> original_imposed_stuff, rotated_imposed_stuff;
        if (imposed_component<dim)
          original_imposed_stuff[imposed_component]=1.;
        else
          original_imposed_stuff[imposed_component-dim]=1.;

        rotated_imposed_stuff=original_imposed_stuff;

        if (dim == 3 && extra_debug_info==true)
          {
            Point<dim> axis;
            axis[0]=1;
            apply_rotation_along_axis(rotated_imposed_stuff,original_imposed_stuff,axis,angle_for_imposed_stuff);
            dpcout<<"original imposed vector (V or Omega) "<<original_imposed_stuff<<std::endl;
            dpcout<<"angle for imposed stuff "<<angle_for_imposed_stuff<<std::endl;
          }
        dpcout<<"effective imposed vector (V or Omega) "<<rotated_imposed_stuff<<std::endl;

        // Now we can take care of the stuff regarding the num_rigid momentum balances. It only happens on proc 0.
        for (types::global_dof_index i = dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
          {
            Vector<double> loc_N_rigid_dual(N_rigid_dual[i-dh_stokes.n_dofs()]);
            if (this_mpi_process == 0)
              {
                monolithic_rhs[i] = 0.;
                // monolithic_system_preconditioner_matrix.set(i,i,1.);
                if (grid_type != "Real")
                  {
                    if (imposed_component < dim && i-dh_stokes.n_dofs()<dim)
                      monolithic_rhs[i] = rotated_imposed_stuff[i-dh_stokes.n_dofs()];
                    else if (imposed_component >= dim && i-dh_stokes.n_dofs()>=dim)
                      monolithic_rhs[i] = rotated_imposed_stuff[i-dh_stokes.n_dofs()-dim];
                    else
                      monolithic_rhs[i] = 0.;
                    if (grid_type == "ImposedVelocity")
                      monolithic_system_matrix.set(i,i,1.*assemble_scaling);
                    else
                      {
                        for (types::global_dof_index j = 0; j<dh_stokes.n_dofs(); ++j)
                          monolithic_system_matrix.set(i,j,loc_N_rigid_dual[j]);
                        for (types::global_dof_index j = dh_stokes.n_dofs(); j<dh_stokes.n_dofs()+num_rigid; ++j)
                          monolithic_system_matrix.set(i,j,0.);

                      }
                  }
                else
                  {
                    for (types::global_dof_index j = 0; j<dh_stokes.n_dofs(); ++j)
                      monolithic_system_matrix.set(i,j,assemble_scaling*loc_N_rigid_dual[j]);
                    for (types::global_dof_index j = dh_stokes.n_dofs(); j<dh_stokes.n_dofs()+num_rigid; ++j)
                      monolithic_system_matrix.set(i,j,0.);


                  }




              }
            dpcout<<"monolithic rhs final components "<< monolithic_rhs[i]<<std::endl;

          }
        if (solve_with_torque)
          {
            Vector<double> loc_N_flagellum_torque_dual(N_flagellum_torque_dual);
            if (this_mpi_process == 0 )
              {
                for (types::global_dof_index j = 0; j<dh_stokes.n_dofs(); ++j)
                  {
                    monolithic_system_matrix.set(dh_stokes.n_dofs()+num_rigid,j,assemble_scaling*loc_N_flagellum_torque_dual[j]);
                    monolithic_system_preconditioner_matrix.set(dh_stokes.n_dofs()+num_rigid,j,assemble_scaling*loc_N_flagellum_torque_dual[j]);
                  }
                monolithic_system_preconditioner_matrix.set(dh_stokes.n_dofs()+num_rigid,dh_stokes.n_dofs()+num_rigid,1.0);
                monolithic_rhs[dh_stokes.n_dofs()+num_rigid]=-2.;
              }
          }

        monolithic_rhs.compress(VectorOperation::insert);

        monolithic_system_matrix.compress(VectorOperation::insert);
        // pcout<<"JJJ "<<monolithic_system_matrix(dh_stokes.n_dofs(),dh_stokes.n_dofs())<<std::endl;
        assemble_monolithic_preconditioner();
      }
    // If we are debugging we may want to take a look into the matrices themselves so we save them in txt.
    if (true)//(extra_debug_info)
      {
        std::ofstream ofs_monolithic;
        std::string filename_monolithic;
        filename_monolithic="monolithic_matrix.txt";
        ofs_monolithic.open (filename_monolithic, std::ofstream::out);
        std::ofstream ofs_M;
        std::string filename_M;
        filename_M="Mass_matrix.txt";
        ofs_M.open (filename_M, std::ofstream::out);
        std::ofstream ofs_normals;
        std::string filename_normals;
        filename_normals="normals_matrix.txt";
        ofs_normals.open (filename_normals, std::ofstream::out);
        std::ofstream ofs_V;
        std::string filename_V;
        filename_V="V_matrix.txt";
        ofs_V.open (filename_V, std::ofstream::out);

        std::ofstream ofs_K;
        std::string filename_K;
        filename_K="K_matrix.txt";
        ofs_K.open (filename_K, std::ofstream::out);
        for (auto i : monolithic_cpu_set)
          {
            for (auto j : monolithic_cpu_set)
              {
                if (monolithic_full_sparsity_pattern.exists(i,j))
                  ofs_monolithic<<monolithic_system_matrix(i,j)<<" ";
                else
                  ofs_monolithic<<0.<<" ";

              }
            ofs_monolithic << std::endl;
          }
        ofs_monolithic.close();
        for (auto i : this_cpu_set)
          {
            for (auto j : this_cpu_set)
              {
                if (sparsity_pattern.exists(i,j))
                  {
                    ofs_M<<Mass_Matrix(i,j)<<" ";
                  }
                else
                  {
                    ofs_M<<0<<" ";
                  }
                if (full_sparsity_pattern.exists(i,j))
                  {
                    ofs_V<<V_matrix(i,j)<<" ";
                    ofs_K<<K_matrix(i,j)<<" ";
                  }
                else
                  {
                    ofs_V<<0.<<" ";
                    ofs_K<<0.<<" ";
                  }
              }
            ofs_normals<<normal_vector_pure[i]<<" "<<M_normal_vector_pure[i]<<std::endl;
            ofs_M << std::endl;
            ofs_V << std::endl;
            ofs_K << std::endl;
          }
        ofs_V.close();
        ofs_K.close();
        ofs_M.close();
        ofs_normals.close();
      }

    K_matrix.vmult(stokes_rhs, shape_velocities);


  }

  template <int dim>
  void BEMProblem<dim>::assemble_monolithic_preconditioner()
  {
    if (bandwith_preconditioner)
      {
        for (auto i : monolithic_cpu_set)
          {
            if (false) //preconditioner_type=="Jacobi")
              {
                monolithic_system_preconditioner_matrix.set(i,i,monolithic_system_matrix(i,i));

              }
            else
              {
                if (constraints.is_constrained(i))
                  {
                    std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);
                    monolithic_system_preconditioner_matrix.set(i,i,1.);
                    for (unsigned int ic=0; ic<constr_in.size(); ++ic)
                      {
                        monolithic_system_preconditioner_matrix.set(i,constr_in[ic].first,-constr_in[ic].second);
                      }
                  }
                else
                  {
                    types::global_dof_index my_start, my_end;
                    if (i > bandwith)
                      my_start = i-bandwith;
                    else
                      my_start = 0;
                    if (i+bandwith > dh_stokes.n_dofs()+num_rigid)
                      my_end = dh_stokes.n_dofs()+num_rigid;
                    else
                      my_end = i+bandwith;
                    for (types::global_dof_index j=my_start; j<my_end; ++j)
                      monolithic_system_preconditioner_matrix.set(i,j,monolithic_system_matrix(i,j));
                    // if (this_mpi_process == 0)
                    //   for (types::global_dof_index i = 0; i<num_rigid; ++i)
                    //     monolithic_system_preconditioner_matrix.set(i+dh_stokes.n_dofs(),i+dh_stokes.n_dofs(),1.0);
                  }
              }
          }

      }
    else
      {
        for (auto i : monolithic_cpu_set)
          {
            if (constraints.is_constrained(i))
              {
                std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);
                monolithic_system_preconditioner_matrix.set(i,i,1.);
                for (unsigned int ic=0; ic<constr_in.size(); ++ic)
                  {
                    monolithic_system_preconditioner_matrix.set(i,constr_in[ic].first,-constr_in[ic].second);
                  }
              }
            else
              for (types::global_dof_index j = 0 ; j<dh_stokes.n_dofs()+num_rigid; ++j)
                monolithic_system_preconditioner_matrix.set(i,j,monolithic_system_matrix(i,j));
          }
        // if (this_mpi_process == 0)
        //   for (types::global_dof_index i = 0; i<num_rigid; ++i)
        //     monolithic_system_preconditioner_matrix.set(i+dh_stokes.n_dofs(),i+dh_stokes.n_dofs(),1.0);
      }

    monolithic_system_preconditioner_matrix.compress(VectorOperation::insert);

  }

  template <int dim>
  void BEMProblem<dim>::assemble_stokes_system_galerkin(bool correction_on_V)
  {
    correction_on_V = false;
    std::cout<<"Dummy cout"<<correction_on_V<<std::endl;
//     // // Check on the support points in vectorial and scalar case.
//     // std::vector<Point<dim-1> > pointvect(fe_stokes->dofs_per_cell);
//     // std::vector<Point<dim-1> > pointscal(fe.dofs_per_cell);
//     // pointvect=fe_stokes->get_unit_support_points();
//     // pointscal=fe.get_unit_support_points();
//
//     // std::cout<<"Scalar unit supp points"<<std::endl;
//     // for (unsigned int i =0 ; i<fe.dofs_per_cell; ++i)
//     // {
//     //   std::cout<<pointscal[i]<<" "<<std::endl;
//     //   if(i==fe.dofs_per_cell-1)
//     //     std::cout<<"\n"<<std::endl;
//     // }
//     // std::cout<<"Vect unit supp points"<<std::endl;
//
//     // for (unsigned int i =0 ; i<fe_stokes->dofs_per_cell; ++i)
//     // {
//     //   std::cout<<pointvect[i]<<" "<<std::endl;
//     //   if(i==fe_stokes->dofs_per_cell-1)
//     //     std::cout<<"\n"<<std::endl;
//     // }
//
//     // pointvect=fe_stokes->get_unit_support_points();
//     // pointscal=fe.get_unit_support_points();
//     Teuchos::TimeMonitor LocalTimer(*AssembleTime);
//     FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
//                                     update_values |
//                                     update_cell_normal_vectors |
//                                     update_quadrature_points |
//                                     update_JxW_values);
//     FEValues<dim-1,dim> fe_stokes_2_v(*mappingeul, *fe_stokes, quadrature_ext,
//                                     update_values |
//                                     update_cell_normal_vectors |
//                                     update_quadrature_points |
//                                     update_JxW_values);
//     /*
//       const FEValuesExtractors::Vector<dim> val(0);
//     */
//     std::vector<types::global_dof_index> local_dof_indices_ext(fe_stokes->dofs_per_cell);
//     std::vector<types::global_dof_index> local_dof_indices_int(fe_stokes->dofs_per_cell);
//
//     // Carefule with the number of q points
//     // std::vector<Vector<double> > cell_vel(n_q_points, Vector<double>(dim) );
//
//     FullMatrix<double>    local_single_layer(fe_stokes->dofs_per_cell, fe_stokes->dofs_per_cell);
//     FullMatrix<double>    local_double_layer(fe_stokes->dofs_per_cell, fe_stokes->dofs_per_cell);
//     // std::vector<Point<dim> > support_points(dh_stokes.n_dofs());
//     // DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, support_points);
//
//     typename DoFHandler<dim-1,dim>::active_cell_iterator
//     cell = dh_stokes.begin_active(),
//     endc = dh_stokes.end();
//
//     typename DoFHandler<dim-1,dim>::active_cell_iterator
//     cell_2 = dh_stokes.begin_active(),
//     endc_2 = dh_stokes.end();
//
//     for (cell = dh_stokes.begin_active(); cell != endc; ++cell)
//     {
//         // pcout<<"Work on a cell "<<std::endl;
//         fe_stokes_v.reinit(cell);
//         cell->get_dof_indices(local_dof_indices_int);
//         for (cell_2 = dh_stokes.begin_active(); cell_2 != endc; ++cell_2)
//         {
//           if(cell_2->material_id()==this_mpi_process)
//           {
//             // if(constraints.is_constrained(i)==false)
//
//             fe_stokes_2_v.reinit(cell_2);
//             cell_2->get_dof_indices(local_dof_indices_ext);
//
//               local_single_layer = 0.;
//               local_double_layer = 0.;
//               bool is_singular = false;
//               // std::vector<unsigned int> singular_indices(dim, numbers::invalid_unsigned_int);
//
//               if(*cell == *cell_2)
//               {
//                 is_singular = true;
//               }
//               // for (unsigned int j=0; j<fe_stokes->dofs_per_cell/dim; ++j)//fe.dofs_per_cell WHY???
//               // {
//               //     unsigned int jdim = fe_stokes->system_to_component_index(j).first;
//               //     if (local_dof_indices[j] == i+jdim*dh_stokes.n_dofs()/dim)
//               //     {
//               //         singular_indices[jdim] = j;
//               //         is_singular = true;
//               //     }
//               //     // if(is_singular)
//               //     //   break;
//               //
//               // }
//
//               FEValues<dim-1,dim> * internal_fe_v;
//               // The external stuff should always be the same since we have already integrated the
//               // singularity with the interior fe_v.
//               FEValues<dim-1, dim> * external_fe_v;
//
//               external_fe_v = & fe_stokes_2_v;
//               // external_fe_v->reinit(cell_2);
//               const std::vector<Point<dim> > &t_points = external_fe_v->get_quadrature_points();
//               unsigned int n_t_points = t_points.size();
//
//               // the i should be the internal index (the coloumn).
//               for(unsigned int i = 0; i<fe_stokes->dofs_per_cell; ++i)
//               {
//                 // We need to select tha singular quadrature if we are singular. If the two pointer are equals we need
//                 // Telles' formula to properly convolve kernels. Thus the internal fe shoul be the Telles' one.
//                 if (is_singular == true) {
//                     internal_fe_v  = & get_singular_fe_values(i);// should be correct since they should be 3 equal support_points
//                     internal_fe_v->reinit(cell);
//                 }
//                 else
//                 {
//                     // we already reinit the fe_stokes_v.
//                     internal_fe_v = & fe_stokes_v;
//                 }
//                 // We can collect everything belonging to the internal fe.
//                 const std::vector<Point<dim> > &q_points = internal_fe_v->get_quadrature_points();
//                 const std::vector<Tensor<1, dim> > &normals  = internal_fe_v->get_normal_vectors();
//                 unsigned int n_q_points = q_points.size();
//
//                 for(unsigned int t=0; t<n_t_points; ++t)
//                 {
//                   for (unsigned int q=0; q<n_q_points; ++q)
//                   {
//                     const Tensor<1,dim> R = q_points[q] - t_points[t];
//                     Tensor<2,dim> G = stokes_kernel.value_tens(R) ;
//                     Tensor<3,dim> W = stokes_kernel.value_tens2(R) ;
//                     Tensor<2,dim> singular_ker = compute_singular_kernel(normals[q], W) ;
//
//                     // the j stands for the external cycle. It should represent the row index
//                     for (unsigned int j=0; j<fe_stokes->dofs_per_cell; ++j)
//                     {
//                         // jdim and idim are needed to select the component of the kernel.
//                         unsigned int jdim = fe_stokes->system_to_component_index(j).first;
//                         unsigned int idim = fe_stokes->system_to_component_index(i).first;
//
//                         local_single_layer(j, i) +=  ( G[jdim][idim] *
//                                                           internal_fe_v->shape_value(i,q)     *
//                                                           internal_fe_v->JxW(q)               *
//                                                           external_fe_v->shape_value(j,t)     *
//                                                           external_fe_v->JxW(t)               );
//                         // if(jdim != idim)
//                           // pcout<<internal_fe_v->shape_value(i,q)<<" "<<external_fe_v->shape_value(j,t) <<" "<<local_single_layer(j, i)<<std::endl;
//                         // THE MINUS IS FOR THE EXTERIOR PROBLEM STORY!
//                         local_double_layer(j, i) -= (singular_ker[jdim][idim] *
//                                                         (internal_fe_v->shape_value(i,q)) * // - (+cval)) * //internal_fe_v->shape_value(j, support_points[i])) *
//                                                         internal_fe_v->JxW(q) *
//                                                         external_fe_v->shape_value(j,t) *
//                                                         external_fe_v->JxW(t));
//                     }
//
//                   }
//                 }
//
//
//               }
//               constraints.distribute_local_to_global(local_single_layer, local_dof_indices_ext, local_dof_indices_int, V_matrix);
//               constraints.distribute_local_to_global(local_double_layer, local_dof_indices_ext, local_dof_indices_int, K_matrix);
//           }
//         }
//     }
//
//     // compute_normal_vector();
//     //includere matrice massa
//     //check normale
//     V_matrix.vmult(V_x_normals, normal_vector);
//     TrilinosWrappers::MPI::Vector foo(this_cpu_set, mpi_communicator);
//     tangential_projector_body(normal_vector, foo);
//     pcout << "Check on the tangential projection (should be zero): " << foo.linfty_norm() << std::endl;
//     //
//     //
//     //
//     // //if()
//     pcout << "Check on the V operator Norm (should be zero): " << V_x_normals.linfty_norm() << std::endl;
//     // double Vnn = V_x_normals * normal_vector;
//     if(correction_on_V)
//     {
//     // for(unsigned int i=0; i<dh_stokes.n_dofs()/dim;++i)
//     // {
//     //   for(unsigned int idim=0; idim<dim; ++idim)
//     //     for(unsigned int jdim=0; jdim<dim; ++jdim)
//     //       V_matrix(i+idim*dh_stokes.n_dofs()/dim,i+jdim*dh_stokes.n_dofs()/dim) += normal_vector(i+idim*dh_stokes.n_dofs()/dim) * normal_vector(i+jdim*dh_stokes.n_dofs()/dim);
//     //
//     // }
//     //
//     //
//       Vector<double> local_M_normal_vector(M_normal_vector);
//       // Vector<double> local_normal_vector(normal_vector);
//       for(auto i : this_cpu_set)
//       {
//         // if(i<dh_stokes.n_dofs()/dim)
//         //   for(unsigned int idim=0; idim<dim; ++idim)
//         //     for(unsigned int jdim=0; jdim<dim; ++jdim)
//         //         // V_matrix(i,j) += first_evec(i) * M_first_evec(j)/l2normGamma_evec;
//         //         V_matrix.add(i+idim*dh_stokes.n_dofs()/dim,i+jdim*dh_stokes.n_dofs()/dim,
//         //         normal_vector(i+idim*dh_stokes.n_dofs()/dim) * local_M_normal_vector(i+jdim*dh_stokes.n_dofs()/dim));
//         for(unsigned int j=0; j<dh_stokes.n_dofs(); ++j)
//             // V_matrix(i,j) += first_evec(i) * M_first_evec(j)/l2normGamma_evec;
//             V_matrix.add(i,j,normal_vector(i) * local_M_normal_vector(j)/l2normGamma);
//       }
//       V_matrix.compress(VectorOperation::add);
//     }                         //iss- V_x_normals(i)*V_x_normals(j)/Vnn;
//     // V_matrix(i+idim*dh_stokes.n_dofs()/dim,i+jdim*dh_stokes.n_dofs()/dim) += normal_vector(i+idim*dh_stokes.n_dofs()/dim) * M_normal_vector(i+jdim*dh_stokes.n_dofs()/dim)/l2normGamma;
//     // NdiadicN(i+idim*dh_stokes.n_dofs()/dim,i+jdim*dh_stokes.n_dofs()/dim) += normal_vector(i+idim*dh_stokes.n_dofs()/dim) * M_normal_vector(i+jdim*dh_stokes.n_dofs()/dim);
//     // std::cout<<"[";
//     // for(unsigned int i = 0; i<dh_stokes.n_dofs(); ++i)
//     // {
//     //     std::cout<<"[ ";
//     //     for(unsigned int j = 0; j<dh_stokes.n_dofs(); ++j)
//     //         std::cout<<V_matrix(i,j)<<", ";
//     //     std::cout<<"],"<<std::endl;
//     // }
//     // std::cout<<"]"<<std::endl;
//     pcout<<"Corrected V"<<std::endl;
//
//
//     V_matrix.vmult(V_x_normals, normal_vector);
//
//     // foo.sadd(0.,1., V_x_normals,-1., normal_vector) ;
//     // std::cout << "Check on the V operator Norm post (should be one): " << V_x_normals.linfty_norm() << std::endl;
//     // std::cout << "Check on the V_x_norm post (should be zero): " << foo.linfty_norm() << std::endl;
//
//
//
//
//     // Test for the singular kernel
//
//     std::vector<TrilinosWrappers::MPI::Vector > VersorMatrix(dim);
//     std::vector<TrilinosWrappers::MPI::Vector > CheckMatrix(dim);
//
//     for(unsigned int i = 0; i<dim; ++i)
//     {
//       VersorMatrix[i].reinit(this_cpu_set, mpi_communicator);
//       CheckMatrix[i].reinit(this_cpu_set, mpi_communicator);
//     }
//
//
//     // TrilinosWrappers::SparseMatrix CheckMatrix(this_cpu_set, complete_index_set(dim), mpi_communicator);
//
//     pcout<<"building versor matrix"<<std::endl;
//     for(unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
//     {
//       if(this_cpu_set.is_element(i))
//         for(unsigned int j=0; j<dim; ++j)
//             VersorMatrix[j][i+j*dh_stokes.n_dofs()/dim]=1.;
//     }
//     for(unsigned int i = 0; i<dim; ++i)
//       VersorMatrix[i].compress(VectorOperation::insert);
//
//     // TODO CHANGE MMULT IN 3 VMULT. PARALLEL ISSUE.
//     for(unsigned int i = 0; i<dim; ++i)
//       K_matrix.vmult(CheckMatrix[i],VersorMatrix[i]);
//
//     pcout<<"correcting K matrix"<<std::endl;
//
//     for(unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
//     {
//       if(this_cpu_set.is_element(i) && !constraints.is_constrained(i))
//         for(unsigned int j=0; j<dim; ++j)
//         {
//             for(unsigned int k=0; k<dim; ++k)
//             {
//                 // if(i==45)
//                 //   std::cout<<"K ("<<j<<", "<<k<<") = "<<CheckMatrix(+j*dh_stokes.n_dofs()/dim,k)<<std::endl;
//                 // if(i+j*dh_stokes.n_dofs()/dim == 232 && k == 1)
//                 //   pcout<<"che te lo dico a fare"<<std::endl;
//
//                 K_matrix.add(i+j*dh_stokes.n_dofs()/dim,i+k*dh_stokes.n_dofs()/dim,-CheckMatrix[k][i+j*dh_stokes.n_dofs()/dim]);
//                 if(j == k)
//                     K_matrix.add(i+j*dh_stokes.n_dofs()/dim,i+k*dh_stokes.n_dofs()/dim,1.);
//
//             }
//         }
//     }
//     K_matrix.compress(VectorOperation::add);
//
//     std::vector<TrilinosWrappers::MPI::Vector> tmp_N;
//     tmp_N.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
//
//     TrilinosWrappers::MPI::Vector tmp_shape_vel(this_cpu_set, mpi_communicator);
//     tangential_projector_body(shape_velocities, tmp_shape_vel);
//     shape_velocities = tmp_shape_vel;
//     K_matrix.vmult(tmp_shape_vel, shape_velocities);
//     for(unsigned int i=0; i<num_rigid; ++i)
//     {
//       K_matrix.vmult(tmp_N[i], N_rigid[i]);
//       // for(auto j : this_cpu_set)
//       //   pcout<<tmp_N[i][j]<<" "<<N_rigid[i][j]<<std::endl;
//     }
//
//     pcout<<"monolithic building"<<std::endl;
//     Vector<double> local_normal_vector(normal_vector);
//     for(auto i : this_cpu_set)
//     {
//       if(constraints.is_constrained(i))
//       {
//         std::vector<std::pair<types::global_dof_index, double> > constr_in = *constraints.get_constraint_entries(i);
//         monolithic_system_matrix.set(i,i,1.);
//         monolithic_rhs[i] = 0.;
//
//         for(unsigned int ic=0; ic<constr_in.size(); ++ic)
//         {
//           if(body_cpu_set.is_element(constr_in[ic].first) || wall_no_slip_cpu_set.is_element(constr_in[ic].first))
//             monolithic_system_matrix.set(i,constr_in[ic].first,V_matrix(i,constr_in[ic].first));
//           else if(wall_free_surface_cpu_set.is_element(constr_in[ic].first) || wall_do_nothing_cpu_set.is_element(constr_in[ic].first))
//             monolithic_system_matrix.set(i,constr_in[ic].first,-K_matrix(i,constr_in[ic].first));
//         }
//       }
//       else
//       {
//         // if(grid_type == "ImposedForce")
//
//         monolithic_rhs[i] = tmp_shape_vel[i];
//         // else
//         //   monolithic_rhs[i] = 0.;
//         for(auto j : body_cpu_set)//(types::global_dof_index j = 0; j < dh_stokes.n_dofs(); ++j)
//         {
//           monolithic_system_matrix.set(i,j,V_matrix(i,j));
//         }
//         for(auto j : wall_no_slip_cpu_set)
//         {
//           monolithic_system_matrix.set(i,j,V_matrix(i,j));
//         }
//
//         for(auto j : wall_free_surface_cpu_set)
//         {
//           if(std::abs(std::abs(local_normal_vector(j))-1) < 1e-3)
//             monolithic_system_matrix.set(i,j,V_matrix(i,j));
//           else
//             monolithic_system_matrix.set(i,j,-K_matrix(i,j));
//         }
//
//         for(types::global_dof_index j = dh_stokes.n_dofs(); j < dh_stokes.n_dofs()+num_rigid; ++j)
//           monolithic_system_matrix.set(i,j,-tmp_N[j-dh_stokes.n_dofs()][i]);
//       }
//     }
//     for(types::global_dof_index i = dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
//     {
//       if(this_mpi_process == 0)
//       {
//         monolithic_rhs[i] = 0.;
//         if(i==dh_stokes.n_dofs() && grid_type != "Real")
//           monolithic_rhs[i] = 1.;
//       }
//       Vector<double> loc_N_rigid_dual(N_rigid_dual[i-dh_stokes.n_dofs()]);
//       if(this_mpi_process == 0)
//         for(types::global_dof_index j = 0; j<dh_stokes.n_dofs(); ++j)
//           monolithic_system_matrix.set(i,j,loc_N_rigid_dual[j]);
//
//     }
//
//     monolithic_rhs.compress(VectorOperation::insert);
//     constraints.distribute(monolithic_rhs);
//     monolithic_system_matrix.compress(VectorOperation::insert);
//
//     // PROBLEM WITH SPARSITY PATTERN
//     // pcout<<CheckMatrix(232,1)<<std::endl;
//     // K_matrix.mmult(CheckMatrix,VersorMatrix);
//     // pcout<<CheckMatrix(232,1)<<std::endl;
//
//     // for(unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
//     // {
//     //   if(this_cpu_set.is_element(i))
//     //     for(unsigned int j=0; j<dim; ++j)
//     //     {
//     //         for(unsigned int k=0; k<dim; ++k)
//     //         {
//     //             if(k==j)
//     //             {
//     //                 if(std::abs(CheckMatrix(i+k*dh_stokes.n_dofs()/dim,j)-1.)>=1e-7)
//     //                 {
//     //                     std::cout<<"something is wrong in matrix assemblement CheckMatrix( "<<i+k*dh_stokes.n_dofs()/dim<<", "<<j<<") = "<<CheckMatrix(i+k*dh_stokes.n_dofs()/dim,j)<<" != 1"<<std::endl;
//     //                 }
//     //             }
//     //             else
//     //             {
//     //                 if(std::abs(CheckMatrix(i+k*dh_stokes.n_dofs()/dim,j))>=1e-7)
//     //                 {
//     //                     std::cout<<"something is wrong in matrix assemblement CheckMatrix( "<<i+k*dh_stokes.n_dofs()/dim<<", "<<j<<") = "<<CheckMatrix(i+k*dh_stokes.n_dofs()/dim,j)<<" != 0"<<std::endl;
//     //                 }
//     //             }
//     //         }
//     //     }
//     // }
//
//     K_matrix.vmult(stokes_rhs, shape_velocities);
//
// //    for(unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
// //      for(unsigned int d=0; d<dim; ++d)
//     // vnormals(i+d*dh_stokes.n_dofs()/dim) = support_points[i+d*dh_stokes.n_dofs()/dim][d];
//
// //    V_matrix.vmult(V_x_normals, vnormals);
// //    std::cout << "Norm: " << V_x_normals.linfty_norm() << std::endl;
//
//
//
//     // for (unsigned int i = 0; i < dh_stokes.n_dofs(); ++i)
//     //    for (unsigned int j = 0; j < dh_stokes.n_dofs(); ++j)
//     //     {
//     //       std::cout<<V_matrix(i,j)<<"  ";
//     //       if (j == dh_stokes.n_dofs() -1 )
//     //       {
//     //         std::cout<<std::endl;
//     //       }
//     //     }
//     // V_matrix.print(std::cout);
//     // K_matrix.print(std::cout);
  }
  template <int dim>
  void BEMProblem<dim>::compute_normal_vector()
  {

    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_cell_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);
    /*
    const FEValuesExtractors::Vector<dim> val(0);
    */
    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);

    // Carefule with the number of q points
    // std::vector<Vector<double> > cell_vel(n_q_points, Vector<double>(dim) );

    TrilinosWrappers::MPI::Vector   normal_rhs(this_cpu_set, mpi_communicator);

    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();

    for (cell = dh_stokes.begin_active(); cell != endc; ++cell)
      {
        if (cell->subdomain_id()==this_mpi_process)
          {
            fe_stokes_v.reinit(cell);
            Vector<double> cell_normal_rhs(fe_stokes->dofs_per_cell);

            cell->get_dof_indices(local_dof_indices);
            const std::vector<Point<dim> > &q_points = fe_stokes_v.get_quadrature_points();
            const std::vector<Tensor<1, dim> > &normals = fe_stokes_v.get_normal_vectors();

            for (unsigned int q_index=0; q_index<q_points.size(); ++q_index)
              for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
                {
                  cell_normal_rhs(i) += (fe_stokes_v.shape_value (i, q_index) *
                                         normals[q_index][fe_stokes->system_to_component_index(i).first] *
                                         fe_stokes_v.JxW (q_index));
                }
            //std::cout<< cell_normal_rhs << std::endl;
            // for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
            //   normal_rhs[local_dof_indices[i]] += cell_normal_rhs[i];
            constraints.distribute_local_to_global(cell_normal_rhs,local_dof_indices,normal_rhs);
          }
      }
    normal_rhs.compress(VectorOperation::add);
    // std::cout<<normal_rhs<<std::endl;
    TrilinosWrappers::SolverCG solver (solver_control);
    //PreconditionJacobi<FullMatrix<double> > precondition;
    //precondition.initialize(V_matrix);
    //K_matrix.vmult(partial_vel, input_vel);
    // Mass_Matrix.print(std::cout);
    TrilinosWrappers::PreconditionAMG mass_prec;
    mass_prec.initialize(Mass_Matrix);
    solver.solve (Mass_Matrix, normal_vector, normal_rhs, mass_prec);
    pcout << "   Iterations needed to solve mass matrix:         " << solver_control.last_step() << std::endl;
    constraints.distribute(normal_vector);
    // normal_vector /= normal_vector.linfty_norm();
    for (unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
      {
        if (this_cpu_set.is_element(i))
          {
            double normal_norm = 0.;
            for (unsigned int idim=0; idim < dim; ++idim)
              normal_norm += normal_vector(i+idim*dh_stokes.n_dofs()/dim)*normal_vector(i+idim*dh_stokes.n_dofs()/dim);
            normal_norm = std::sqrt(normal_norm);
            if (body_cpu_set.is_element(i))
              {
                for (unsigned int idim=0; idim < dim; ++idim)
                  normal_vector_pure(i+idim*dh_stokes.n_dofs()/dim) = normal_vector(i+idim*dh_stokes.n_dofs()/dim)/normal_norm;
              }
            for (unsigned int idim=0; idim < dim; ++idim)
              normal_vector(i+idim*dh_stokes.n_dofs()/dim) = normal_vector(i+idim*dh_stokes.n_dofs()/dim)/normal_norm;

          }

      }

    Mass_Matrix.vmult(M_normal_vector, normal_vector);
    Mass_Matrix.vmult(M_normal_vector_pure, normal_vector_pure);
    l2normGamma = M_normal_vector * normal_vector;
    l2normGamma_pure = M_normal_vector_pure * normal_vector_pure;





  }


  // template<int dim>
  // void BEMProblem<dim>::compute_first_evec(Vector<double> &evec, Vector<double> &M_evec)
  // {
  // Problems on Ulysses with 64 bits

  // PETScWrappers::SparseMatrix             V;
  // std::vector<PETScWrappers::MPI::Vector> eigenfunctions;
  // std::vector<double>                     eigenvalues;
  //
  // IndexSet eigenfunction_index_set = dh_stokes.locally_owned_dofs ();
  //
  // V.reinit (dh_stokes.n_dofs(),
  //           dh_stokes.n_dofs(),
  //           dh_stokes.n_dofs());
  //
  // eigenfunctions.resize(1);
  // for (unsigned int i=0; i<eigenfunctions.size (); ++i)
  //     eigenfunctions[i].reinit (eigenfunction_index_set, MPI_COMM_WORLD);
  //
  // // std::cout<<"[";
  // for(unsigned int i=0 ; i<dh_stokes.n_dofs(); ++i)
  // {
  //   // std::cout<<"[ ";
  //   for(unsigned int j=0 ; j<dh_stokes.n_dofs(); ++j)
  //   {
  //     V.set(i,j,V_matrix(i,j));
  //     // std::cout<<bem_problem_3d.V_matrix(i,j)<<", ";
  //   }
  //   // std::cout<<"],"<<std::endl;
  // }
  // // std::cout<<"]"<<std::endl;
  //
  // V.compress(VectorOperation::insert);
  //
  // SolverControl solver_control (dh_stokes.n_dofs(), 1e-9);
  // SLEPcWrappers::SolverKrylovSchur eigensolver (solver_control);
  // eigensolver.set_which_eigenpairs (EPS_SMALLEST_REAL);
  // eigensolver.set_problem_type (EPS_GHEP);
  // eigensolver.solve (V,
  //                eigenvalues, eigenfunctions,
  //                eigenfunctions.size());
  // for (unsigned int i=0; i<eigenfunctions.size(); ++i)
  // {
  //   eigenfunctions[i] /= eigenfunctions[i].linfty_norm ();
  //   std::cout<<eigenvalues[i]<<std::endl;
  // }
  // for(unsigned int i=0; i<dh_stokes.n_dofs(); ++i)
  //   evec[i] = eigenfunctions[0][i];
  //
  // Mass_Matrix.vmult(M_evec, evec);
  // l2normGamma_evec = M_evec * evec;



  // }


// We define the DN operator. This will be very very handy in the solve system method.
  template <int dim>
  void BEMProblem<dim>::dirichlet_to_neumann_operator(const TrilinosWrappers::MPI::Vector &input_vel,  TrilinosWrappers::MPI::Vector &output_force)
  {
    //Vector<double> output_force(dh_stokes.n_dofs());
    CustomOperator<TrilinosWrappers::MPI::Vector, TrilinosWrappers::SparseMatrix> my_operator(normal_vector, M_normal_vector, l2normGamma, V_matrix);

    TrilinosWrappers::MPI::Vector partial_vel_1(this_cpu_set, mpi_communicator);
    TrilinosWrappers::MPI::Vector partial_vel_2(this_cpu_set, mpi_communicator);
    TrilinosWrappers::MPI::Vector partial_force(this_cpu_set, mpi_communicator);
    // PreconditionJacobi<FullMatrix<double> > precondition;
    // precondition.initialize(V_matrix);
    // PreconditionIdentity precondition;
    tangential_projector_body(input_vel, partial_vel_1);
    K_matrix.vmult(partial_vel_2, partial_vel_1);
    tangential_projector_body(partial_vel_2, partial_vel_1);

    // V_sp.copy_from(V_matrix);
    // V_sp.compress();
    // V_matrix_sparse.reinit(V_sp);
    //
    // for(unsigned int i=0; i<dh_stokes.n_dofs(); ++i)
    //   for(unsigned int j=0; j<dh_stokes.n_dofs(); ++j)
    //     V_matrix_sparse.set(i,j,V_matrix(i,j));
    // std::vector<unsigned int> V_sizes(dh_stokes.n_dofs(), dh_stokes.n_dofs());
    //

    // SparseDirectUMFPACK  V_direct;
    // V_direct.initialize(V_matrix_sparse);
    //
    // V_direct.vmult (partial_force, partial_vel_1);

    // TrilinosWrappers::SolverDirect solvy(solver_control);
    // solvy.solve(V_matrix, partial_force, partial_vel_1);
    // TrilinosWrappers::PreconditionILU preconditioner_foo;
    if (solve_directly)
      {
        // TrilinosWrappers::SolverDirect::AdditionalData::AdditionalData add_data(true,"Amesos-Mumps");
        TrilinosWrappers::SolverDirect solvy(solver_control);
        solvy.initialize(V_matrix);
        solvy.solve(partial_force, partial_vel_1);//monolithic_system_matrix,
      }
    else
      {
        SolverGMRES<TrilinosWrappers::MPI::Vector > solver (solver_control, gmres_additional_data);
        solver.solve (V_matrix, partial_force, partial_vel_1, direct_trilinos_preconditioner);
      }
    pcout << "   Iterations needed to solve DN:         " << solver_control.last_step() << std::endl;
    tangential_projector_body(partial_force, output_force);


    // Vector<double> buff(dh_stokes.n_dofs());
    // SolverGMRES<Vector<double> > solverN (solver_control);
    // precondition.initialize(NdiadicN);
    // solver.solve (NdiadicN, buff, partial_vel, precondition);
    //
    // output_force.sadd(1.,+1.,buff);
    return;
  }


  template <int dim>
  void BEMProblem<dim>::tangential_projector(const TrilinosWrappers::MPI::Vector &input_vel,  TrilinosWrappers::MPI::Vector &output_vel)
  {
    // double pippo;
    // pippo = M_normal_vector * normal_vector;
    output_vel.sadd(0.,1.,input_vel);
    output_vel.sadd(1., -(M_normal_vector_pure*input_vel)/l2normGamma, normal_vector);
    return;
  }

  template <int dim>
  void BEMProblem<dim>::tangential_projector_body(const TrilinosWrappers::MPI::Vector &input_vel,  TrilinosWrappers::MPI::Vector &output_vel)
  {
    // double pippo;
    // pippo = M_normal_vector * normal_vector;
    // dpcout<<"GUARDA QUI : "<<M_normal_vector_pure.linfty_norm()<<" "<<input_vel.linfty_norm()<<" " <<M_normal_vector_pure*input_vel<<std::endl;
    output_vel.sadd(0.,1.,input_vel);
    output_vel.sadd(1., -(M_normal_vector_pure*input_vel)/l2normGamma_pure, normal_vector_pure);
    return;
  }

// @sect4{BEMProblem::solve_system}

// The next function simply solves the linear system, and compute the rigid_velocities.


  template <int dim>
  void BEMProblem<dim>::solve_system(bool monolithic_booly)
  {
    Teuchos::TimeMonitor LocalTimer(*LacSolveTime);
    pcout << "Solving system" << std::endl;
    if (!monolithic_booly)
      {
        SolverGMRES<Vector<double> > solver (solver_control);
        FullMatrix<double>  final_matrix(num_rigid,num_rigid);
        Vector<double> final_rhs(num_rigid);
        // Vector<double> final_rhs_new(num_rigid);
        // if (reassemble_preconditoner)
        //   assemble_preconditioner();

        dirichlet_to_neumann_operator(shape_velocities, stokes_forces);
        // dirichlet_to_neumann_operator(N_rigid[dim], stokes_forces);
        pcout<< "pressure test = "<< M_normal_vector *stokes_forces << std::endl;
        pcout<< "shape test = "<< M_normal_vector *shape_velocities << std::endl;
        pcout<< "l2gamma = "<< l2normGamma << std::endl;
        if (grid_type == "Convergence")
          {
            return;
          }
        for (unsigned int i = 0; i<num_rigid; ++i)
          {
            dirichlet_to_neumann_operator(N_rigid[i], DN_N_rigid[i]);
            final_rhs(i) = -(N_rigid_dual[i] * stokes_forces);
          }
        for (unsigned int i = 0; i<num_rigid; ++i)
          {
            if (grid_type=="ImposedForce")
              {
                if (i==imposed_component)
                  final_rhs[imposed_component]+=1.;
                for (unsigned int j = 0; j<num_rigid; ++j)
                  final_matrix(i,j) = N_rigid_dual[i] * DN_N_rigid[j];

              }
            else if (grid_type=="ImposedVelocity")
              {
                final_matrix(i,i)=1.;
                final_rhs[i]=0.;
                if (i==imposed_component)
                  final_rhs[imposed_component]+=1.;

              }
            else
              {
                for (unsigned int j = 0; j<num_rigid; ++j)
                  final_matrix(i,j) = N_rigid_dual[i] * DN_N_rigid[j];

              }

          }

        if (solver_control.last_step()>(solver_control.max_steps()/2))
          reassemble_preconditoner = true;
        else
          reassemble_preconditoner = true;
        // final_matrix.invert(final_matrix);//gauss_jordan();
        // final_matrix.vmult(rigid_velocities, final_rhs);
        if (this_mpi_process==0)
          {
            pcout<<"MATRICE FINALE !!!"<<std::endl;
            std::string filename_matrix;
            std::ofstream ofs_matrix;
            filename_matrix="final_matrix.txt";
            std::string filename_rhs;
            std::ofstream ofs_rhs;
            filename_rhs="final_rhs.txt";

            ofs_matrix.open (filename_matrix, std::ofstream::out | std::ofstream::app);
            ofs_rhs.open (filename_rhs, std::ofstream::out | std::ofstream::app);
            for (unsigned int ii=0; ii<num_rigid; ++ii)
              {
                for (unsigned int jj=0; jj<num_rigid; ++jj)
                  pcout<<final_matrix(ii,jj)<<" ";
                pcout<<std::endl;
                for (unsigned int jj=0; jj<num_rigid; ++jj)
                  ofs_matrix<<final_matrix(ii,jj)<<" ";
                ofs_matrix<<std::endl;
                ofs_rhs<<final_rhs[ii]<<std::endl;
              }
            ofs_matrix.close();
            ofs_rhs.close();
          }
        solver.solve(final_matrix, rigid_velocities, final_rhs, PreconditionIdentity());
        Vector<double> foo(num_rigid);
        final_matrix.vmult(foo,rigid_velocities);
        for (unsigned int i = 0; i<num_rigid; ++i)
          stokes_forces.sadd(1.,rigid_velocities[i],DN_N_rigid[i]);
        for (unsigned int i = 0; i<num_rigid; ++i)
          {
            double rigid_force = 0.;
            for (unsigned int j = 0; j<num_rigid; ++j)
              rigid_force+=rigid_velocities[j] * (N_rigid_dual[i]*DN_N_rigid[j]);
            pcout<<rigid_force<<" "<<rigid_velocities[i]<<" "<<foo[i]<<std::endl;
          }
        pcout << "   Iterations needed to solve final:         " << solver_control.last_step() << std::endl;
      }
    else
      {
        if (solve_directly)
          {
            // TrilinosWrappers::SolverDirect::AdditionalData::AdditionalData add_data(true,"Amesos-Mumps");
            TrilinosWrappers::SolverDirect solvy(solver_control);
            solvy.initialize(monolithic_system_matrix);
            solvy.solve(monolithic_solution, monolithic_rhs);//monolithic_system_matrix,
          }
        else
          {
            // TrilinosWrappers::SolverGMRES monolithic_solver (solver_control);
            SolverGMRES<TrilinosWrappers::MPI::Vector > monolithic_solver (solver_control, gmres_additional_data);
            TrilinosWrappers::PreconditionILU foo_ilu;
            TrilinosWrappers::PreconditionAMG foo_amg;
            TrilinosWrappers::PreconditionJacobi foo_jacobi;
            TrilinosWrappers::PreconditionSOR  foo_sor;
            TrilinosWrappers::PreconditionSSOR foo_ssor;
            // pcout<<"JJJ "<<dh_stokes.n_dofs()<<" "<<dh_stokes.n_dofs()<<" "<<monolithic_system_matrix(dh_stokes.n_dofs(),dh_stokes.n_dofs())<<std::endl;
            // pcout<<monolithic_system_preconditioner_matrix(dh_stokes.n_dofs(),dh_stokes.n_dofs())<<" "<<monolithic_system_matrix(dh_stokes.n_dofs(),dh_stokes.n_dofs())<<std::endl;
            for (auto i : monolithic_cpu_set)
              for (types::global_dof_index j=0; j<monolithic_cpu_set.size(); ++j)
                if (monolithic_preconditioner_sparsity_pattern->exists(i,j) && monolithic_system_preconditioner_matrix(i,j)!=monolithic_system_matrix(i,j))
                  pcout<<"HHH "<<i<<" "<<j<<" "<<monolithic_system_preconditioner_matrix(i,j)<<" "<<monolithic_system_matrix(i,j)<<std::endl;
            if (preconditioner_type == "ILU")
              {
                foo_ilu.initialize(monolithic_system_preconditioner_matrix);
                monolithic_system_preconditioner = &foo_ilu;
              }
            else if (preconditioner_type == "AMG")
              {
                foo_amg.initialize(monolithic_system_preconditioner_matrix);
                monolithic_system_preconditioner = &foo_amg;
              }
            else if (preconditioner_type == "Jacobi")
              {
                TrilinosWrappers::PreconditionJacobi::AdditionalData add_data_jac(1.0,0.,1);
                foo_jacobi.initialize(monolithic_system_preconditioner_matrix, add_data_jac);
                monolithic_system_preconditioner = &foo_jacobi;
              }
            else if (preconditioner_type == "SOR")
              {
                foo_sor.initialize(monolithic_system_preconditioner_matrix);
                monolithic_system_preconditioner = &foo_sor;
              }
            else if (preconditioner_type == "SSOR")
              {
                foo_ssor.initialize(monolithic_system_preconditioner_matrix);
                monolithic_system_preconditioner = &foo_ssor;
              }
            else if (preconditioner_type == "Direct")
              {
                AssertThrow(bandwith_preconditioner == false, ExcNotImplemented());
                monolithic_system_preconditioner = &direct_trilinos_preconditioner;
              }
            // prec_data.elliptic=false;
            // prec_data.n_cycles = 1;
            // // prec_data.coase_solver = "Amesos-KLU";
            // // prec_data.smoother_type = "Amesos-KLU";//"Amesos-Superludist";//"Amesos-UMFPACK";//"SuperLU";//;"Amesos-KLU";//;"IFPACK";//;"Aztec";//"MLS"; //"self";
            // prec_data.aggregation_threshold = 1.;
            // prec_data.smoother_sweeps = 10;
            // prec_data.output_details = true;
            // prec_data.smoother_overlap = 1;
            // prec_data.n_cycles = 10;
            // prec_data.w_cycle = true;
            // TrilinosWrappers::PreconditionAMG::AdditionalData::AdditionalData add_data_amg(false,
            //                               false,1,false,1e-4,std::vector<std::vector<bool> > (0),
            //                               2,0,false,"Jacobi","Amesos-KLU");
            // CustomOperator<TrilinosWrappers::MPI::Vector, TrilinosWrappers::SparseMatrix > monolithic_operator(monolithic_normal, monolithic_normal_dual, l2normGamma_pure, monolithic_system_matrix);
            // monolithic_system_preconditioner->initialize(monolithic_system_preconditioner_matrix);
            // direct_trilinos_preconditioner.initialize(monolithic_system_matrix);
            // direct_trilinos_preconditioner.vmult(monolithic_solution, monolithic_rhs);
            pcout<<"preconditioner_type = "<<preconditioner_type<<std::endl;
            monolithic_solver.solve(monolithic_system_matrix, monolithic_solution, monolithic_rhs, *monolithic_system_preconditioner);//direct_trilinos_preconditioner);//monolithic_prec

            monolithic_system_preconditioner = NULL;
            pcout << "   Iterations needed to solve monolithic:         " << solver_control.last_step() << std::endl;
            if (solver_control.last_step() > 100)
              {
                reassemble_preconditoner=true;
              }
          }


        TrilinosWrappers::MPI::Vector dummy_foo(monolithic_cpu_set,mpi_communicator);
        monolithic_system_matrix.vmult(dummy_foo,monolithic_solution);
        dummy_foo.sadd(1.,-1.,monolithic_rhs);
        pcout<<"FINAL CHECK 0 "<<dummy_foo.linfty_norm()<<" : "<<dummy_foo.l2_norm()<<std::endl;

        TrilinosWrappers::MPI::Vector stokes_forces_foo(this_cpu_set, mpi_communicator);
        wall_velocities=0.;

        for (auto i : this_cpu_set)
          {
            // if (!constraints.is_constrained(i))
            {
              if (monolithic_system_matrix(i,i)==-K_matrix(i,i)) //(wall_free_surface_cpu_set.is_element(i) && std::abs(std::abs(normal_vector(i)) - 1.)>=1e-3 ) ||
                //  (wall_do_nothing_cpu_set.is_element( i ) && std::abs(std::abs(normal_vector(i)) - 0.)>=1e-3 ))
                {
                  wall_velocities[i] = monolithic_solution[i];
                  stokes_forces_foo[i] = 0.;
                }
              else if (monolithic_system_matrix(i,i)==V_matrix(i,i))
                {
                  stokes_forces_foo[i] = monolithic_solution[i];
                  wall_velocities[i] = 0.;
                }
            }

          }
        stokes_forces_foo.compress(VectorOperation::insert);
        wall_velocities.compress(VectorOperation::insert);
        constraints.distribute(stokes_forces_foo);
        constraints.distribute(wall_velocities);
        stokes_forces = stokes_forces_foo;
        // pcout<<stokes_forces.linfty_norm()<<" QUI "<<stokes_forces.l2_norm()<<std::endl;
        // pcout<<monolithic_solution.linfty_norm()<<" QUI "<<monolithic_solution.l2_norm()<<std::endl;
        double motor_torque = N_flagellum_torque_dual*stokes_forces;
        if (this_mpi_process==0 && solve_with_torque)
          {
            std::string filename_torque;
            std::ofstream ofs_torque;
            filename_torque="motor_torque.txt";

            ofs_torque.open (filename_torque, std::ofstream::out | std::ofstream::app);
            ofs_torque<<motor_torque<<std::endl;
            ofs_torque.close();
          }
        // tangential_projector_body(stokes_forces_foo, stokes_forces);

        // Vector<double> loc_mon(monolithic_solution);
        //
        // for(types::global_dof_index i = dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
        //   pcout<<loc_mon[i]<<std::endl;
        if (this_mpi_process == 0)
          {
            for (types::global_dof_index i = dh_stokes.n_dofs(); i<dh_stokes.n_dofs()+num_rigid; ++i)
              {
                rigid_velocities[i-dh_stokes.n_dofs()]=monolithic_solution[i];
                baricenter_rigid_velocities[i-dh_stokes.n_dofs()]=monolithic_solution[i];
              }
            if (solve_with_torque)
              {
                flagellum_omega = monolithic_solution[dh_stokes.n_dofs()+num_rigid];
                // shape_velocities.sadd(0.,flagellum_omega,N_flagellum_torque);
              }
          }
        // Vector<double> foo_force(dh_stokes.n_dofs());
        // // for(auto i : foo_force.locally_owned_elements())
        // //   foo_force[i] = loc_mon[i];
        if (solve_with_torque)
          {
            MPI_Bcast(&flagellum_omega, 1, MPI_DOUBLE, 0, mpi_communicator);
            shape_velocities.sadd(0.,flagellum_omega,N_flagellum_torque);

          }
        MPI_Bcast(&rigid_velocities[0], num_rigid, MPI_DOUBLE, 0, mpi_communicator);
        MPI_Bcast(&baricenter_rigid_velocities[0], num_rigid, MPI_DOUBLE, 0, mpi_communicator);

        TrilinosWrappers::MPI::Vector check(monolithic_cpu_set, mpi_communicator);
        monolithic_system_matrix.vmult(check, monolithic_solution);

        for (auto i : monolithic_cpu_set)
          check[i] -= monolithic_rhs[i];
        // for(unsigned int i =0; i<num_rigid; ++i)
        //   pcout<<stokes_forces * N_rigid_dual[i]<<" "<<rigid_velocities[i]<<std::endl;
        TrilinosWrappers::MPI::Vector dummy_2(this_cpu_set, mpi_communicator);
        TrilinosWrappers::MPI::Vector dummy_1(this_cpu_set, mpi_communicator);
        TrilinosWrappers::MPI::Vector dummy_3(this_cpu_set, mpi_communicator);
        TrilinosWrappers::MPI::Vector dummy_4(this_cpu_set, mpi_communicator);
        TrilinosWrappers::MPI::Vector dummy_5(this_cpu_set, mpi_communicator);
        TrilinosWrappers::MPI::Vector dummy_6(this_cpu_set, mpi_communicator);

        dummy_2 *= 0.;
        std::vector<TrilinosWrappers::MPI::Vector> tmp_N;
        std::vector<TrilinosWrappers::MPI::Vector> tmp_N_2;
        tmp_N.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
        tmp_N_2.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
        for (unsigned int i=0; i<num_rigid; ++i)
          {
            tangential_projector_body(N_rigid[i], tmp_N[i]);
            K_matrix.vmult(tmp_N_2[i], tmp_N[i]);
            tangential_projector_body(tmp_N_2[i], tmp_N[i]);
            // constraints.distribute(tmp_N[i]);
            dummy_2.sadd(1.,assemble_scaling*rigid_velocities[i],tmp_N[i]);
          }
        K_matrix.vmult(dummy_4, wall_velocities);
        constraints.distribute(dummy_4);
        tangential_projector_body(shape_velocities, dummy_5);
        K_matrix.vmult(dummy_6, dummy_5);
        tangential_projector_body(dummy_6, dummy_5);
        constraints.distribute(dummy_5);
        V_matrix.vmult(dummy_1, stokes_forces);
        dummy_3.sadd(0.,-1.,dummy_1);
        dummy_3.sadd(1.,1.,dummy_4);
        dummy_3.sadd(1.,1.,dummy_5);
        dummy_3.sadd(1.,1.,dummy_2);
        constraints.distribute(dummy_3);
        pcout<<"FINAL CHECK 1 "<<dummy_3.linfty_norm()<<" : "<<dummy_3.l2_norm()<<std::endl;
      }
    std::ofstream ofs_vel_cg;
    std::string filename_vel_cg;
    filename_vel_cg="point_velocities.txt";
    ofs_vel_cg.open (filename_vel_cg, std::ofstream::out | std::ofstream::app);
    for (unsigned int i=0; i<num_rigid; ++i)
      {
        rigid_velocities[i]*=assemble_scaling;
        baricenter_rigid_velocities[i]*=assemble_scaling;
      }
    for (unsigned int i =0; i<num_rigid; ++i)
      {
        rigid_total_forces[i]=stokes_forces * N_rigid_dual[i];
        rigid_total_forces_complete[i]=stokes_forces * N_rigid_dual_complete[i];
        if (this_mpi_process==0)
          {

            ofs_vel_cg<<rigid_total_forces[i]<<" "<<baricenter_rigid_velocities[i]<<" "<<rigid_total_forces_complete[i]<<std::endl;
          }
      }
    // ofs_vel_cg << std::endl;
    ofs_vel_cg.close();
    pcout<<"Transforming velocities from the force pole to origin O"<<std::endl;
    if (force_pole!="Origin")
      {
        rigid_velocities[0] += baricenter_rigid_velocities[4] * (0.-point_force_pole[2]) - baricenter_rigid_velocities[5] * (0.-point_force_pole[1]);
        rigid_velocities[1] += baricenter_rigid_velocities[5] * (0.-point_force_pole[0]) - baricenter_rigid_velocities[3] * (0.-point_force_pole[2]);
        rigid_velocities[2] += baricenter_rigid_velocities[3] * (0.-point_force_pole[1]) - baricenter_rigid_velocities[4] * (0.-point_force_pole[0]);
      }
    if (this_mpi_process==0)
      {
        std::ofstream ofs_vel_0;
        std::string filename_vel_0;
        filename_vel_0="origin_velocities.txt";
        ofs_vel_0.open (filename_vel_0, std::ofstream::out | std::ofstream::app);
        for (unsigned int i =0; i<num_rigid; ++i)
          {
            ofs_vel_0<<rigid_total_forces[i]<<" "<<rigid_velocities[i]<<std::endl;
          }
        // ofs_vel_0 << std::endl;
        ofs_vel_0.close();
      }
    pcout << "solved system" << std::endl;
  }


  template<int dim>
  void BEMProblem<dim>::compute_rotation_matrix_from_quaternion(FullMatrix<double> &rotation, const Vector<double> &q)
  {
    rotation.reinit(3,3);
    rotation[0][0] = 1. - 2 * (q[3]*q[3] + q[2]*q[2]);
    rotation[0][1] = - 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    rotation[0][2] = + 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    rotation[1][0] = + 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    rotation[1][1] = 1. - 2 * (q[3]*q[3] + q[1]*q[1]);
    rotation[1][2] = - 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    rotation[2][0] = - 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    rotation[2][1] = + 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    rotation[2][2] = 1. - 2 * (q[1]*q[1] + q[2]*q[2]);

  }

  template<int dim>
  void BEMProblem<dim>::update_rotation_matrix(FullMatrix<double> &rotation, const Vector<double> omega, const double dt, const bool forward_euler, const double theta)
  {
    // pcout << "Updating the rotation matrix using quaternions" << std::endl;
    // Firstly we need to reconstruct the original quaternion given the rotation matrix
    // FullMatrix<double> dummy(dim,dim);
    // dummy[0][0]=1.;
    // dummy[2][1]=1.;
    // dummy[1][2]=-1.;
    // rotation = dummy;
    // rotation.print(std::cout);
    Vector<double> q(dim+1);
    double q_dummy;
    // IT SEEMS OK.
    q_dummy = 1.;
    for (int i = 0; i<dim; ++i)
      q_dummy += (rotation[i][i]);

    q[0] = std::pow(q_dummy, 0.5)/2;


    q[1] = 1 / q[0] * 0.25 * (rotation[2][1] - rotation[1][2]);
    q[2] = 1 / q[0] * 0.25 * (rotation[0][2] - rotation[2][0]);
    q[3] = 1 / q[0] * 0.25 * (rotation[1][0] - rotation[0][1]);

    double foo = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    // std::cout<<foo<<std::endl;
    q/=foo;//std::sqrt(foo);
    // q.print(std::cout);
    Vector<double> qdot(dim+1), omega_plus(dim+1);
    for (unsigned int i=0; i<dim; ++i)
      omega_plus[i+1] = omega[i];
    // omega_plus[0] = 0.;
    // omega_plus[3] = 1.;
    //Next we can update the quaternion using qdot = S^-1 omega_plus
    FullMatrix<double> S_inv(dim+1,dim+1);
    //CHECK 0.5
    S_inv[0][0] = q[0];

    S_inv[1][0] = q[1];
    S_inv[2][0] = q[2];
    S_inv[3][0] = q[3];

    S_inv[0][1] = - q[1];
    S_inv[0][2] = - q[2];
    S_inv[0][3] = - q[3];

    S_inv[1][1] = q[0];
    S_inv[2][2] = q[0];
    S_inv[3][3] = q[0];

    S_inv[1][1] += 0.;
    S_inv[1][2] = +q[3];
    S_inv[1][3] = -q[2];

    S_inv[2][1] = -q[3];
    S_inv[2][2] += 0.;
    S_inv[2][3] = +q[1];

    S_inv[3][1] = +q[2];
    S_inv[3][2] = -q[1];
    S_inv[3][3] += 0.;


    // S_inv[0][0] = q[0];
    //
    // S_inv[1][0] = q[1];
    // S_inv[2][0] = q[2];
    // S_inv[3][0] = q[3];
    //
    // S_inv[0][1] = - q[1];
    // S_inv[0][2] = - q[2];
    // S_inv[0][3] = - q[3];
    //
    // S_inv[1][1] = q[0];
    // S_inv[2][2] = q[0];
    // S_inv[3][3] = q[0];
    //
    // S_inv[1][1] += 0.;
    // S_inv[1][2] = +q[3];
    // S_inv[1][3] = -q[2];
    //
    // S_inv[2][1] = -q[3];
    // S_inv[2][2] += 0.;
    // S_inv[2][3] = +q[1];
    //
    // S_inv[3][1] = -q[2];
    // S_inv[3][2] = +q[1];
    // S_inv[3][3] += 0.;

    S_inv *= 0.5;

    //   for(unsigned int j=0; j<dim+1; ++j)
    //     S_inv[i][j] = omega_plus[i] * q[j];

    // omega_plus.print(std::cout);
    S_inv.vmult(qdot, omega_plus);
    // // S_inv.print_formatted(std::cout);
    // // std::cout<<"old quat"<<std::endl;
    // q.print(std::cout);
    // std::cout<<"velocity"<<std::endl;
    // qdot.print(std::cout);
    // Vector<double> q_new(dim+1);
    // qdot.print(std::cout);
    // q.print(std::cout);

    if (forward_euler)
      {
        q.sadd(1.,dt,qdot);
        // q.print(std::cout);
      }
    else
      {
        FullMatrix<double> A_CN(dim+1,dim+1);
        A_CN[0][0] = 1.+theta*dt*omega_plus[0]*0.5;

        A_CN[0][1] = theta*dt*omega_plus[1]*0.5;
        A_CN[0][2] = theta*dt*omega_plus[2]*0.5;
        A_CN[0][3] = theta*dt*omega_plus[3]*0.5;

        A_CN[1][0] = -theta*dt*omega_plus[1]*0.5;
        A_CN[2][0] = -theta*dt*omega_plus[2]*0.5;
        A_CN[3][0] = -theta*dt*omega_plus[3]*0.5;

        A_CN[1][1] = 1.+theta*dt*omega_plus[0]*0.5;
        A_CN[1][2] = -theta*dt*omega_plus[3]*0.5;
        A_CN[1][3] = theta*dt*omega_plus[2]*0.5;

        A_CN[2][1] = theta*dt*omega_plus[3]*0.5;
        A_CN[2][2] = 1.+theta*dt*omega_plus[0]*0.5;
        A_CN[2][3] = -theta*dt*omega_plus[1]*0.5;

        A_CN[3][1] = -theta*dt*omega_plus[2]*0.5;
        A_CN[3][2] = theta*dt*omega_plus[1]*0.5;
        A_CN[3][3] = 1.+theta*dt*omega_plus[0]*0.5;

        Vector<double> rhs(dim+1);
        rhs.sadd(0.,1.,q);
        // qdot=Sinv*q (OK)
        rhs.sadd(1.,(1-theta)*dt,qdot);
        SolverGMRES<Vector<double> > solvy_quat(solver_control);
        // A_CN.print_formatted(std::cout);
        // q.print(std::cout);
        // qdot.print(std::cout);
        solvy_quat.solve(A_CN,q,rhs,PreconditionIdentity());
        // q.print(std::cout);
      }

    foo = std::sqrt(q[0]*q[0]+q[1]*q[1]+q[2]*q[2]+q[3]*q[3]);
    // std::cout<<foo<<std::endl;
    q /= (foo);

    // std::cout<<"new quat"<<std::endl;

    // q.print(std::cout);

    compute_rotation_matrix_from_quaternion(rotation,q);

    //Once we have the new quaternion we can update the rotation matrix using R = I + 2q0qx + 2qxqx
    // rotation[0][0] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[2]*q[2]);
    // rotation[0][1] = - 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    // rotation[0][2] = + 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    // rotation[1][0] = + 2 * q[0] * q[3] + 2 * (q[1] * q[2]);
    // rotation[1][1] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[1]*q[1]);
    // rotation[1][2] = - 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    // rotation[2][0] = - 2 * q[0] * q[2] + 2 * (q[1] * q[3]);
    // rotation[2][1] = + 2 * q[0] * q[1] + 2 * (q[3] * q[2]);
    // rotation[2][2] = 1. + 2 * q[0] * 0 + 2 * -(q[3]*q[3] + q[2]*q[2]);

    // IT SEEMS OK!


    FullMatrix<double> foo_mat(dim,dim);

    rotation.Tmmult(foo_mat, rotation);

    double tol = 1e-7;
    for (unsigned int i=0; i<dim; ++i)
      for (unsigned int j=0; j<dim; ++j)
        {
          if (i == j)
            {
              if (std::fabs(foo_mat[i][j]-1) >= tol)
                pcout<<"Something Wrong in Rotations, on the diagonal "<<std::fabs(foo_mat[i][j]-1)<<std::endl;
            }
          else
            {
              if (std::fabs(foo_mat[i][j]) >=tol )
                pcout<<"Something Wrong in Rotations, out the diagonal "<<std::fabs(foo_mat[i][j])<<std::endl;

            }
        }

  }


// @sect4{BEMProblem::update_system_state}

// This function updates the system state. Namely it integrates the rigid velocities obtained with BEM.

  template<int dim>
  void BEMProblem<dim>::update_system_state(bool compute, unsigned int frame, bool consider_rotations, bool consider_displacements, std::string res_system)
  {
    // The back up vectors for the predictot corrector

    // if we are not Computing the real solution we need to set up fake angular and linear velocities.
    if (compute == false)
      {
        compute_normal_vector();
        Vector<double> vel_amplitude(dim);
        Vector<double> ome_amplitude(dim);
        if (dim == 3)
          {
            vel_amplitude(0) = 0;
            vel_amplitude(1) = 0;
            vel_amplitude(2) = 0;

            ome_amplitude(0) = 1;
            ome_amplitude(1) = 1;
            ome_amplitude(2) = 1;
            for (unsigned int i=0; i < dim; ++i)
              {
                rigid_velocities(i) = vel_amplitude(i) * sin(2 * 4 * numbers::PI * 1/140 * frame);
                rigid_velocities(i + dim) = ome_amplitude(i);// * sin(2 * numbers::PI * 1/140 * frame);
              }

          }
        else
          {
            vel_amplitude(0) = 1;
            vel_amplitude(1) = 0;

            ome_amplitude(0) = 0;

            for (unsigned int i=0; i < dim; ++i)
              {
                rigid_velocities(i) = vel_amplitude(i) * sin(2 * 4 * numbers::PI * 1/140 * frame);
              }
            rigid_velocities(dim) = ome_amplitude(0) * sin(2 * numbers::PI * 1/140 * frame);

          }
      }
    // rigid_velocities.print(std::cout);
    if (res_system=="Heun" && res_strategy=="Heun")
      {
        // recover of the back up for the corrector
        rotation_matrix=old_rotation_matrix;
        rigid_displacements_for_sim=old_rigid_displacements_for_sim;
        rigid_velocities.sadd(0.5, 0.5, old_rigid_velocities);
      }
    else if (res_system=="Forward" && res_strategy=="Heun")
      {
        // back up of the system state for the corrector
        old_rigid_velocities=rigid_velocities;
        old_rotation_matrix=rotation_matrix;
        old_rigid_displacements_for_sim=rigid_displacements_for_sim;
      }
    rigid_puntual_velocities=0.;
    for (unsigned int i=0; i<dim; ++i)
      rigid_puntual_velocities.sadd(1., assemble_scaling*baricenter_rigid_velocities(i), N_rigid[i]);
    rigid_puntual_translation_velocities = rigid_puntual_velocities;
    for (unsigned int i=dim; i<num_rigid; ++i)
      rigid_puntual_velocities.sadd(1., assemble_scaling*baricenter_rigid_velocities(i), N_rigid[i]);
    // rigid_puntual_velocities.compress(VectorOperation::add);
    // pcout<<rigid_puntual_velocities.linfty_norm()<<std::endl;
    // for(unsigned int i=0; i<dim; ++i)
    //     rigid_puntual_translation_velocities.sadd(1., rigid_velocities(i), N_rigid_map[i]);

    bool fake = false;
    if (dim == 3 && consider_rotations)
      {
        Vector<double> omega(dim);
        if (fake)
          {
            pcout << "considering a FAKE angular velocity" << std::endl;
            omega[0] = 0.1;//rigid_velocities[3];
          }
        else
          omega[0] = rigid_velocities[3];
        omega[1] = rigid_velocities[4];
        omega[2] = rigid_velocities[5];
        // we can update the rotation matrix in 3d using quaternions.
        update_rotation_matrix(rotation_matrix, omega, time_step);

      }
    // We have used the angular velocities to update the quaternion. Now it is time to update the positions.
    // We simply integrate the rigid translation velocities using a forward euler scheme.
    // If we prescribe apredictor corrector scheme we have already computed a mean velocity to be integrated.
    // DOUBT: maybe we should use rigid_puntual_translation_velocities together with the rotation matrix.
    next_rigid_puntual_displacements.sadd(0., time_step, rigid_puntual_translation_velocities);
    // total_euler_vec.sadd(0.,1.,euler_vec);
    // total_euler_vec.sadd(1.,1.,next_rigid_puntual_displacements);
    if (consider_displacements)
      {
        Vector<double> loc_rigid_puntual_displacements(next_rigid_puntual_displacements);
        if (bool_dipl_x)
          {
            for (types::global_dof_index i=0; i<loc_rigid_puntual_displacements.size()/dim; ++i)
              {
                rigid_displacements_for_sim[i] += loc_rigid_puntual_displacements[i];
              }
          }
        if (bool_dipl_y)
          {
            for (types::global_dof_index i=loc_rigid_puntual_displacements.size()/dim; i<2*loc_rigid_puntual_displacements.size()/dim; ++i)
              {
                rigid_displacements_for_sim[i] += loc_rigid_puntual_displacements[i];
              }
          }
        if (dim == 3 && bool_dipl_z)
          {
            for (types::global_dof_index i=2*loc_rigid_puntual_displacements.size()/dim; i<3*loc_rigid_puntual_displacements.size()/dim; ++i)
              {
                rigid_displacements_for_sim[i] += loc_rigid_puntual_displacements[i];
              }
          }

      }

  }

// Singular integration requires a careful selection of the quadrature
// rules. In particular the deal.II library provides quadrature rules which
// are tailored for logarithmic singularities (QGaussLog, QGaussLogR), as
// well as for 1/R singularities (QGaussOneOverR).
//
// Singular integration is typically obtained by constructing weighted
// quadrature formulas with singular weights, so that it is possible to
// write
//
// \f[ \int_K f(x) s(x) dx = \sum_{i=1}^N w_i f(q_i) \f]
//
// where $s(x)$ is a given singularity, and the weights and quadrature
// points $w_i,q_i$ are carefully selected to make the formula above an
// equality for a certain class of functions $f(x)$.
//
// In all the finite element examples we have seen so far, the weight of the
// quadrature itself (namely, the function $s(x)$), was always constantly
// equal to 1.  For singular integration, we have two choices: we can use
// the definition above, factoring out the singularity from the integrand
// (i.e., integrating $f(x)$ with the special quadrature rule), or we can
// ask the quadrature rule to "normalize" the weights $w_i$ with $s(q_i)$:
//
// \f[ \int_K f(x) s(x) dx = \int_K g(x) dx = \sum_{i=1}^N
//   \frac{w_i}{s(q_i)} g(q_i) \f]
//
// We use this second option, through the @p factor_out_singularity
// parameter of both QGaussLogR and QGaussOneOverR.
//
// These integrals are somewhat delicate, especially in two dimensions, due
// to the transformation from the real to the reference cell, where the
// variable of integration is scaled with the determinant of the
// transformation.
//
// In two dimensions this process does not result only in a factor appearing
// as a constant factor on the entire integral, but also on an additional
// integral altogether that needs to be evaluated:
//
// \f[ \int_0^1 f(x)\ln(x/\alpha) dx = \int_0^1 f(x)\ln(x) dx - \int_0^1
//  f(x) \ln(\alpha) dx.  \f]
//
// This process is taken care of by the constructor of the QGaussLogR class,
// which adds additional quadrature points and weights to take into
// consideration also the second part of the integral.
//
// A similar reasoning should be done in the three dimensional case, since
// the singular quadrature is tailored on the inverse of the radius $r$ in
// the reference cell, while our singular function lives in real space,
// however in the three dimensional case everything is simpler because the
// singularity scales linearly with the determinant of the
// transformation. This allows us to build the singular two dimensional
// quadrature rules only once and, reuse them over all cells.
//
// In the one dimensional singular integration this is not possible, since
// we need to know the scaling parameter for the quadrature, which is not
// known a priori. Here, the quadrature rule itself depends also on the size
// of the current cell. For this reason, it is necessary to create a new
// quadrature for each singular integration.
//
// The different quadrature rules are built inside the
// get_singular_quadrature, which is specialized for dim=2 and dim=3, and
// they are retrieved inside the assemble_system function. The index given
// as an argument is the index of the unit support point where the
// singularity is located.

  template<>
  const Quadrature<2> &BEMProblem<3>::get_singular_quadrature(const unsigned int index) const
  {
    Assert(index < fe_stokes->dofs_per_cell,
           ExcIndexRange(0, fe_stokes->dofs_per_cell, index));

    static std::vector<Quadrature<2> > quadratures;
    if (quadratures.size() == 0)
      {
        if (singular_quadrature_type == "Duffy")
          for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
            {
              // if (fe_stokes->degree>1)
              //   quadratures.push_back(QIterated<dim-1>(QGauss<1> (singular_quadrature_order),fe_stokes->degree));
              // else
              quadratures.push_back(QSplit<2> (QDuffy (singular_quadrature_order,1.),fe_stokes->get_unit_support_points()[i]));//,true
              //  quadratures.push_back(QTelles<dim-1>(singular_quadrature_order,  //QTelles<dim-1>, QGaussOneOverR<dim-1>
              //                                   fe_stokes->get_unit_support_points()[i]));

            }
        else if (singular_quadrature_type == "Mixed")
          for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
            {
              if (fe_stokes->degree>1)
                quadratures.push_back(QIterated<2>(QGauss<1> (singular_quadrature_order),fe_stokes->degree));
              else
                quadratures.push_back(QGaussOneOverR<2>(singular_quadrature_order,  //QTelles<dim-1>, QGaussOneOverR<dim-1>
                                                        fe_stokes->get_unit_support_points()[i],true));

            }
        else if (singular_quadrature_type == "Telles")
          for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
            quadratures.push_back(QTelles<2>(singular_quadrature_order,  //QTelles<dim-1>, QGaussOneOverR<dim-1>
                                             fe_stokes->get_unit_support_points()[i]));
      }
    return quadratures[index];
  }
  template<>
  const Quadrature<1> &BEMProblem<2>::get_singular_quadrature(const unsigned int index) const
  {
    Assert(index < fe_stokes->dofs_per_cell,
           ExcIndexRange(0, fe_stokes->dofs_per_cell, index));

    static std::vector<Quadrature<1> > quadratures;
    if (quadratures.size() == 0)
      for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
        {
          if (fe_stokes->degree>1)
            quadratures.push_back(QIterated<1>(QGauss<1> (singular_quadrature_order),fe_stokes->degree));
          else
            quadratures.push_back(QTelles<1>(singular_quadrature_order,
                                             fe_stokes->get_unit_support_points()[i]));
        }

    return quadratures[index];
  }


  template<int dim>
  FEValues<dim-1,dim> & BEMProblem<dim>::get_singular_fe_values(const unsigned int index) const
  {
    static std::vector<FEValues<dim-1, dim> *> fe_values;

    if (index == numbers::invalid_unsigned_int)
      {
        if (fe_values.size())
          for (unsigned int i=0; i<fe_values.size(); ++i)
            delete fe_values[i];
        fe_values.resize(0);
        return *static_cast<FEValues<dim-1,dim> *>(NULL);
      }
    else
      {
        // std::cout<<"ORTOMIO "<<index<<" "<<fe_stokes->dofs_per_cell<<" "<<fe_values.size()<<std::endl;

        Assert(index < fe_stokes->dofs_per_cell,
               ExcIndexRange(0, fe_stokes->dofs_per_cell, index));

        if (fe_values.size() == 0)
          for (unsigned int i=0; i<fe_stokes->dofs_per_cell; ++i)
            fe_values.push_back(new FEValues<dim-1,dim> (*mappingeul, *fe_stokes,
                                                         get_singular_quadrature(index),
                                                         update_jacobians |
                                                         update_values |
                                                         update_cell_normal_vectors |
                                                         update_quadrature_points ));
        return *fe_values[index];
      }
  }
  template<int dim>
  Tensor<2, dim> BEMProblem<dim>::compute_G_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image, const StokesKernel<dim> &stokes_kernel, const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel, const NoSlipWallStokesKernel<dim> &ns_stokes_kernel, const bool reflect, const bool no_slip) const
  {
    Tensor<2, dim> G;
    if (reflect)
      {
        G = fs_stokes_kernel.value_tens_image(R,R_image);
        //  pcout<<"CLARABELLA"<<std::endl;

      }
    else if (no_slip)
      {
        G = ns_stokes_kernel.value_tens_image(R,R_image);
        // pcout<<"PIPPO"<<std::endl;

      }
    else
      {
        G = stokes_kernel.value_tens(R);
      }
    return G;
  }

  template<int dim>
  Tensor<3, dim> BEMProblem<dim>::compute_W_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image, const StokesKernel<dim> &stokes_kernel, const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel, const NoSlipWallStokesKernel<dim> &ns_stokes_kernel, const bool reflect, const bool no_slip) const
  {
    Tensor<3, dim> W;
    if (reflect)
      {
        W = fs_stokes_kernel.value_tens_image2(R,R_image);
        //  pcout<<"MINNI"<<std::endl;
      }
    else if (no_slip)
      {
        W = ns_stokes_kernel.value_tens_image2(R,R_image);
        // pcout<<"TOPOLINO"<<std::endl;
      }
    else
      {
        W = stokes_kernel.value_tens2(R);
      }
    return W;
  }

  template<int dim>
  Tensor<2, dim> BEMProblem<dim>::compute_singular_kernel(const Tensor<1, dim> &normal, const Tensor<3, dim> &W) const
  {
    Tensor<2,dim> singular_kernel;
    for (unsigned  int i=0; i<dim; ++i)
      for (unsigned  int j=0; j<dim; ++j)
        for (unsigned  int k=0; k<dim; ++k)
          {
            singular_kernel[i][j] += W[i][j][k] * normal[k];
          }
    //std::cout<<"result ="<<*result<<" singular_kernel="<< singular_kernel<<std::endl;
    return singular_kernel;
  }

  template<int dim>
  Tensor<2, dim> BEMProblem<dim>::compute_hypersingular_kernel(const Tensor<1, dim> &normal_y,
      const Tensor<1,dim> &normal_x,
      const Tensor<4,dim> &D) const
  {
    Tensor<2,dim> hypersingular_kernel;
    for (unsigned  int i=0; i<dim; ++i)
      for (unsigned  int j=0; j<dim; ++j)
        for (unsigned  int k=0; k<dim; ++k)
          for (unsigned  int m=0; m<dim; ++m)
            hypersingular_kernel[i][j] += D[i][j][k][m] * normal_y[k] * normal_x[m];
    //std::cout<<"result ="<<*result<<" singular_kernel="<< singular_kernel<<std::endl;
    return hypersingular_kernel;
  }


// @sect4{BEMProblem::output_save_stokes_results}

// Outputting the results of our computations is a rather mechanical
// tasks. All the components of this function have been discussed before.
  template<int dim>
  void BEMProblem<dim>::save_rotation_matrix(const FullMatrix<double> &rotation, const unsigned int frame)
  {
    Vector<double> q(dim*dim);
    for (int i = 0; i<dim; ++i)
      for (int j = 0; j<dim; ++j)
        q[i*dim+j] += (rotation[i][j]);

    std::string file_name1;
    file_name1 = "rotation_matrix_" + Utilities::int_to_string(frame) + ".bin";
    std::ofstream rot_mat (file_name1.c_str());
    q.block_write(rot_mat);
  }
  template<int dim>
  void BEMProblem<dim>::read_rotation_matrix(FullMatrix<double> &rotation, const unsigned int frame)
  {

    Vector<double> q(dim*dim);
    std::string file_name1;
    file_name1 = "rotation_matrix_" + Utilities::int_to_string(frame) + ".bin";
    std::ifstream rot_mat (file_name1.c_str());
    q.block_read(rot_mat);

    for (int i = 0; i<dim; ++i)
      for (int j = 0; j<dim; ++j)
        rotation[i][j]=q[i*dim+j];

  }

  template <int dim>
  void BEMProblem<dim>::output_save_stokes_results(const unsigned int cycle)
  {

    Vector<double> loc_stokes_forces(stokes_forces);
    Vector<double> loc_shape_velocities(shape_velocities);
    const Vector<double> loc_normal_vector(normal_vector);
    const Vector<double> loc_rigid_puntual_velocities(rigid_puntual_velocities);
    const Vector<double> loc_rigid_puntual_displacements(rigid_puntual_displacements);
    const Vector<double> loc_next_rigid_puntual_displacements(next_rigid_puntual_displacements);
    const Vector<double> loc_wall_velocities(wall_velocities);
    const Vector<double> loc_total_velocities(total_velocities);
    const Vector<double> loc_final_test(final_test);
    // pcout<<final_test.linfty_norm()<<"!!!"<<std::endl;
    std::vector<Vector<double> > loc_alpha(CheckMatrix.size());
    for (unsigned int i=0; i<dim; ++i)
      {
        loc_alpha[i] = CheckMatrix[i];
      }

    std::vector<Vector<double> > loc_N_rigid(N_rigid.size());
    std::vector<Vector<double> > loc_N_rigid_dual(N_rigid.size());
    std::vector<Vector<double> > loc_DN_N_rigid(DN_N_rigid.size());
    const Vector<double> loc_N_flagellum_torque(N_flagellum_torque);
    for (unsigned int i=0; i<num_rigid; ++i)
      {
        loc_N_rigid[i] = N_rigid[i];
        loc_DN_N_rigid[i] = DN_N_rigid[i];
        loc_N_rigid_dual[i] = N_rigid_dual[i];//DN_N_rigid[i];
      }
    if (this_mpi_process == 0)
      {
        Vector<double> first_original_eig(dh_stokes.n_dofs());
        Vector<double> second_original_eig(dh_stokes.n_dofs());
        Vector<double> first_corrected_eig(dh_stokes.n_dofs());

        // read_eig_vector(first_original_eig,"original_first_eig.txt");
        // read_eig_vector(second_original_eig,"original_second_eig.txt");
        // read_eig_vector(first_corrected_eig,"corrected_first_eig.txt");
        // MappingFEField<dim-1, dim> mapping_tot(map_dh,total_euler_vec);
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
        data_component_interpretation
        (dim, DataComponentInterpretation::component_is_part_of_vector);
        std::vector<Point<dim> > grid_support_points(dh_stokes.n_dofs());
        DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, grid_support_points);
        Vector<double> support_position(dh_stokes.n_dofs());
        for (unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
          {
            for (unsigned int idim=0; idim<dim; ++idim)
              {
                support_position[i+idim*dh_stokes.n_dofs()/dim] = grid_support_points[i][idim];
              }
          }
        DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, grid_support_points);
        DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;

        dataout.attach_dof_handler(dh_stokes);
        dataout.add_data_vector(loc_stokes_forces, std::vector<std::string > (dim,"global_stokes_forces"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        //dataout.add_data_vector(stokes_rhs, std::vector<std::string > (dim,"K_x_shape_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        //dataout.add_data_vector(V_x_normals, std::vector<std::string > (dim,"V_x_n"), DataOut<dim-1,_ DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_normal_vector, std::vector<std::string > (dim,"n"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        // dataout.add_data_vector(first_original_eig, std::vector<std::string > (dim,"eig 1 o"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        // dataout.add_data_vector(second_original_eig, std::vector<std::string > (dim,"eig 2 o"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        // dataout.add_data_vector(first_corrected_eig, std::vector<std::string > (dim,"eig 1 c"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_rigid_puntual_velocities, std::vector<std::string > (dim,"overall_rigid_vel"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_rigid_puntual_displacements, std::vector<std::string > (dim,"rigid_displacement"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_shape_velocities, std::vector<std::string > (dim,"shape_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_total_velocities, std::vector<std::string > (dim,"total_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_wall_velocities, std::vector<std::string > (dim,"wall_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.add_data_vector(loc_final_test, std::vector<std::string > (dim,"final_test"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);

        for (unsigned int i=0; i<dim; ++i)
          {
            dataout.add_data_vector(loc_alpha[i], std::vector<std::string > (dim,"alpha_"+Utilities::int_to_string(i)), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
          }
        dataout.add_data_vector(support_position, std::vector<std::string > (dim,"supp_points"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        for (unsigned int i=0; i<num_rigid; ++i)
          {
            dataout.add_data_vector(loc_N_rigid_dual[i], std::vector<std::string > (dim,"dual_rigid_vel_"+Utilities::int_to_string(i)), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
            dataout.add_data_vector(loc_N_rigid[i], std::vector<std::string > (dim,"rigid_vel_"+Utilities::int_to_string(i)), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
            dataout.add_data_vector(loc_DN_N_rigid[i], std::vector<std::string > (dim,"DN_rigid_vel_"+Utilities::int_to_string(i)), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);

          }
        dataout.add_data_vector(loc_N_flagellum_torque, std::vector<std::string > (dim,"rigid_flagellum"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        dataout.build_patches(*mappingeul,
                              fe_stokes->degree + n_subdivisions,
                              DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

        std::string filename = ( Utilities::int_to_string(dim) +
                                 "d_stokes_boundary_solution_" +
                                 Utilities::int_to_string(cycle) +
                                 ".vtu" );
        std::ofstream file(filename.c_str());

        dataout.write_vtu(file);

        MappingQ<dim-1, dim>      mapping_foo(fe_stokes->degree, true);

        DataOut<dim-1, DoFHandler<dim-1, dim> > foo_dataout;
        Vector<double> pippo(loc_rigid_puntual_velocities.size());
        // for(auto i : map_wall_no_slip_cpu_set)
        // {
        //   if(i>map_dh.n_dofs()/dim && i<=map_dh.n_dofs()/dim*2)
        //     pcout<<i<<" "<<euler_vec[i]<<" ";
        // }
        // euler_vec.print(std::cout);
        // body_cpu_set.print(std::cout);
        foo_dataout.attach_dof_handler(dh_stokes);
        foo_dataout.add_data_vector(loc_rigid_puntual_velocities, std::vector<std::string > (dim,"rigid_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        foo_dataout.add_data_vector(loc_shape_velocities, std::vector<std::string > (dim,"shape_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        foo_dataout.add_data_vector(loc_total_velocities, std::vector<std::string > (dim,"total_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        foo_dataout.add_data_vector(loc_rigid_puntual_displacements, std::vector<std::string > (dim,"rigid_displacement"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        if (fe_map->get_name() == fe_stokes->get_name())
          foo_dataout.add_data_vector(euler_vec, std::vector<std::string > (dim,"euler"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        // foo_dataout.add_data_vector(euler_vec, std::vector<std::string > (dim,"euler_vector"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        // foo_dataout.add_data_vector(total_euler_vec, std::vector<std::string > (dim,"total_euler_vector"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
        foo_dataout.add_data_vector(loc_normal_vector, std::vector<std::string > (dim,"normal_vector"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);

        foo_dataout.build_patches(mapping_foo,
                                  mapping_foo.get_degree(),
                                  DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

        std::string foo_filename = ( Utilities::int_to_string(dim) +
                                     "d_foo_vel_" +
                                     Utilities::int_to_string(cycle) +
                                     ".vtu" );
        std::ofstream foo_file(foo_filename.c_str());

        foo_dataout.write_vtu(foo_file);

        pcout<<"saving"<<std::endl;
        std::string file_name1;
        file_name1 = "stokes_forces_" + Utilities::int_to_string(cycle) + ".bin";
        std::ofstream forces (file_name1.c_str());
        // for(auto ii : wall_free_surface_cpu_set)
        //   if(std::abs(loc_normal_vector[ii]-1)>1e-3)
        //   {
        //     loc_shape_velocities[ii]+=loc_stokes_forces[ii];
        //     loc_stokes_forces[ii]=0.;
        //   }
        loc_stokes_forces.block_write(forces);
        std::string file_name2;
        file_name2 = "shape_velocities_" + Utilities::int_to_string(cycle) + ".bin";
        std::ofstream velocities (file_name2.c_str());
        loc_shape_velocities.block_write(velocities);
        std::string file_name_total_vel;
        file_name_total_vel = "total_velocities_" + Utilities::int_to_string(cycle) + ".bin";
        std::ofstream total_velocities (file_name_total_vel.c_str());
        loc_total_velocities.block_write(total_velocities);
        save_rotation_matrix(rotation_matrix, cycle);
        for (unsigned int r=0; r<num_rigid; ++r)
          {
            std::string file_name_rm;
            file_name_rm = "DN_rigid_mode_" + Utilities::int_to_string(r) + "_frame_"+Utilities::int_to_string(cycle)+".bin";
            std::ofstream rigid_mode (file_name_rm.c_str());
            loc_DN_N_rigid[r].block_write(rigid_mode);
          }
        std::string filename_rigid_vel_short;
        filename_rigid_vel_short = "4_6_rigid_velocities_"+Utilities::int_to_string(cycle)+".bin";
        std::ofstream rv46 (filename_rigid_vel_short.c_str());
        rigid_velocities.block_write(rv46);
        std::string filename_overall_forces;
        filename_rigid_vel_short = "4_6_overall_forces_"+Utilities::int_to_string(cycle)+".bin";
        std::ofstream of46 (filename_rigid_vel_short.c_str());
        rigid_total_forces.block_write(of46);
        // We now compute the rigid rotations considering rotations as well.
        std::string file_name3, file_name4, filename_displ, filename_euler, filename_normals;
        file_name3 = "stokes_rigid_displ_" + Utilities::int_to_string(cycle) + ".bin";
        file_name4 = "stokes_rigid_vel_" + Utilities::int_to_string(cycle) + ".bin";
        filename_euler = "euler_vec_" + Utilities::int_to_string(cycle) + ".bin";
        filename_normals = "normal_vector" + Utilities::int_to_string(cycle) + ".bin";

        std::ofstream rdispl (file_name3.c_str());
        loc_next_rigid_puntual_displacements.block_write(rdispl);

        std::ofstream rvel (file_name4.c_str());
        loc_rigid_puntual_velocities.block_write(rvel);

        std::ofstream reuler (filename_euler.c_str());
        euler_vec.block_write(reuler);

        std::ofstream rnorm (filename_normals.c_str());
        loc_normal_vector.block_write(rnorm);

        std::ofstream ofs_point;
        filename_displ="point_0_on_proc_0_displacement_frame_"+Utilities::int_to_string(cycle)+".txt";
        ofs_point.open (filename_displ);
        ofs_point << cycle << " ";
        for (unsigned int idim = 0; idim<dim; ++idim)
          {
            ofs_point << rigid_puntual_displacements(this_cpu_set.nth_index_in_set(0)) << " ";
          }
        ofs_point << std::endl;
        ofs_point.close();
      }
  }

  template <int dim>
  void BEMProblem<dim>::approximate_velocity_gradient(
    const std::vector<Point<dim> > &val_points,
    const Vector<double> &vel,
    const Vector<double> &forces,
    const double h,
    std::vector<Tensor<2,dim> > &val_gradients)
  {
    if (val_gradients.size() != val_points.size()*dim)
      val_gradients.resize(val_points.size());

    for (types::global_dof_index i=0; i<val_points.size(); ++i)
      {
        std::vector<Point<dim> > grad_points(dim*2);
        Vector<double> grad_vel(grad_points.size()*dim);
        for (unsigned int j=0; j<dim; ++j)
          {
            grad_points[j*2]=val_points[i];
            grad_points[j*2][j]+=h;
            grad_points[j*2+1]=val_points[i];
            grad_points[j*2+1][j]-=h;
          }
        evaluate_stokes_bie(grad_points,vel,forces,grad_vel);
        for (unsigned int j=0; j<dim; ++j)
          {
            for (unsigned int k=0; k<dim; ++k)
              {
                val_gradients[i][j][k]=(grad_vel[j*grad_points.size()+k*2]-grad_vel[j*grad_points.size()+k*2+1])/h;
              }
          }

      }

  }
  template <int dim>
  void BEMProblem<dim>::evaluate_stokes_bie(
    const std::vector<Point<dim> > &val_points,
    const Vector<double> &vel,
    const Vector<double> &forces,
    Vector<double> &val_velocities)
  {
    if (val_velocities.size() != val_points.size()*dim)
      val_velocities.reinit(val_points.size()*dim);
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();



    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_cell_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

    const unsigned int n_q_points_stokes = fe_stokes_v.n_quadrature_points;

    dpcout<<" Stokes Solution norms "<<forces.linfty_norm()<<" "<<forces.l2_norm()<<" , "
          <<vel.linfty_norm()<<" "<<vel.l2_norm()<<" "<<std::endl;

    std::vector<Vector<double> > stokes_local_forces(n_q_points_stokes, Vector<double> (dim));
    std::vector<Vector<double> > stokes_local_velocities(n_q_points_stokes, Vector<double> (dim));
    unsigned int kernel_wall_orientation=numbers::invalid_unsigned_int;
    for (unsigned int i=0; i<dim; ++i)
      {
        if (wall_spans[0][i]==0)
          kernel_wall_orientation=i;
      }
    fs_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    fs_exterior_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    ns_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    ns_exterior_stokes_kernel.set_wall_orientation(kernel_wall_orientation);
    for (cell = dh_stokes.begin_active(); cell != dh_stokes.end(); ++cell)
      {
        fe_stokes_v.reinit(cell);

        const std::vector<Point<dim> > &q_points = fe_stokes_v.get_quadrature_points();
        const std::vector<Tensor<1, dim> > &normals = fe_stokes_v.get_normal_vectors();

        fe_stokes_v.get_function_values(forces, stokes_local_forces);
        fe_stokes_v.get_function_values(vel, stokes_local_velocities);


        for (types::global_dof_index  i=0; i<val_velocities.size()/dim; ++i)
          {
            for (unsigned int q=0; q<n_q_points_stokes; ++q)
              {

                const Tensor<1,dim> R = q_points[q] - val_points[i];
                Point<dim> support_point_image(val_points[i]);
                support_point_image[kernel_wall_orientation] -= 2*(val_points[i][kernel_wall_orientation]-wall_positions[0][kernel_wall_orientation]);
                const Tensor<1,dim> R_image = q_points[q] - support_point_image;
                // std::cout<<q_image<<" "<<q_points[q]<<" "<<q_image - q_points[q]<<std::endl;
                // Point<dim> q_image(q_points[q]);
                // q_image[1] -= 2*(q_points[q][1]-wall_positions[0][1]);
                // const Tensor<1,dim> R_image = q_image - val_points[i];
                // // std::cout<<q_image<<" "<<q_points[q]<<" "<<q_image - q_points[q]<<std::endl;
                Tensor<2,dim> G = compute_G_kernel(R, R_image, exterior_stokes_kernel, fs_exterior_stokes_kernel, ns_exterior_stokes_kernel, reflect_kernel, no_slip_kernel); //exterior_stokes_kernel.value_tens(R) ;
                Tensor<3,dim> W = compute_W_kernel(R, R_image, exterior_stokes_kernel, fs_exterior_stokes_kernel, ns_exterior_stokes_kernel, reflect_kernel, no_slip_kernel);//exterior_stokes_kernel.value_tens2(R) ;
                // Tensor<2,dim> G = fs_exterior_stokes_kernel.value_tens(R) ;
                // Tensor<3,dim> W = fs_exterior_stokes_kernel.value_tens2(R) ;
                // Tensor<2,dim> G = fs_exterior_stokes_kernel.value_tens_image(R,R_image) ;
                // Tensor<3,dim> W = fs_exterior_stokes_kernel.value_tens_image2(R,R_image) ;
                Tensor<2,dim> singular_ker = compute_singular_kernel(normals[q], W) ;

                for (unsigned int idim = 0; idim < dim; ++idim)
                  {
                    for (unsigned int jdim = 0; jdim < dim; ++jdim)
                      {
                        val_velocities(i+val_velocities.size()/dim*idim) +=  G[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                                                                             stokes_local_forces[q](jdim) *
                                                                             fe_stokes_v.JxW(q) ;
                        val_velocities(i+val_velocities.size()/dim*idim) -= singular_ker[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                                                                            stokes_local_velocities[q](jdim) *
                                                                            fe_stokes_v.JxW(q) ;
                      }
                  }
              }
          }
      }
  }

  template <int dim>
  void BEMProblem<dim>::evaluate_stokes_bie_on_boundary(
    const std::vector<Point<dim> > &val_points,
    const Vector<double> &vel,
    const Vector<double> &forces,
    Vector<double> &val_velocities)
  {
    typename DoFHandler<dim-1,dim>::active_cell_iterator
    cell = dh_stokes.begin_active(),
    endc = dh_stokes.end();


    FEValues<dim-1,dim> fe_stokes_v(*mappingeul, *fe_stokes, quadrature,
                                    update_values |
                                    update_cell_normal_vectors |
                                    update_quadrature_points |
                                    update_JxW_values);

    FEValues<dim-1,dim> * internal_fe_v;

    const unsigned int n_q_points_stokes = fe_stokes_v.n_quadrature_points;


    std::vector<Vector<double> > stokes_normal_local_forces(n_q_points_stokes, Vector<double> (dim));
    std::vector<Vector<double> > stokes_normal_local_velocities(n_q_points_stokes, Vector<double> (dim));

    std::vector<Vector<double> > stokes_singular_local_forces(get_singular_fe_values(0).n_quadrature_points, Vector<double> (dim));
    std::vector<Vector<double> > stokes_singular_local_velocities(get_singular_fe_values(0).n_quadrature_points, Vector<double> (dim));

    std::vector<Vector<double> > *stokes_local_forces;
    std::vector<Vector<double> > *stokes_local_velocities;
    std::vector<types::global_dof_index> local_dof_indices(fe_stokes->dofs_per_cell);
    std::vector<Point<dim> > grid_support_points(dh_stokes.n_dofs());
    DoFTools::map_dofs_to_support_points<dim-1, dim>( *mappingeul, dh_stokes, grid_support_points);

    double tol=1e-3;
    for (unsigned int i=0; i<val_velocities.size()/dim; ++i)
      {

        for (cell = dh_stokes.begin_active(); cell != dh_stokes.end(); ++cell)
          {
            cell->get_dof_indices(local_dof_indices);
            bool is_singular = false;
            std::vector<unsigned int> singular_indices(dim, numbers::invalid_unsigned_int);

            for (unsigned int j=0; j<fe_stokes->dofs_per_cell; ++j)
              {
                unsigned int jdim = fe_stokes->system_to_component_index(j).first;
                if (val_points[i].distance(grid_support_points[local_dof_indices[j]]) <= tol)
                  {
                    singular_indices[jdim] = j;
                    is_singular = true;
                  }
                if (is_singular)
                  break;
              }

            if (is_singular == true)
              {
                Assert(singular_indices[0] != numbers::invalid_unsigned_int,
                       ExcInternalError());
                internal_fe_v  = & get_singular_fe_values(singular_indices[0]);


                stokes_local_forces = &stokes_singular_local_forces;
                stokes_local_velocities = &stokes_singular_local_velocities;

              }
            else
              {
                internal_fe_v = & fe_stokes_v;
                stokes_local_forces = &stokes_normal_local_forces;
                stokes_local_velocities = &stokes_normal_local_velocities;

              }

            internal_fe_v->reinit(cell);

            const std::vector<Point<dim> > &q_points = internal_fe_v->get_quadrature_points();
            const std::vector<Tensor<1, dim> > &normals = internal_fe_v->get_normal_vectors();

            internal_fe_v->get_function_values(forces, *stokes_local_forces);
            internal_fe_v->get_function_values(vel, *stokes_local_velocities);

            for (unsigned int q=0; q<internal_fe_v->n_quadrature_points; ++q)
              {

                const Tensor<1,dim> R =   (q_points[q] - val_points[i]);
                Tensor<2,dim> G = exterior_stokes_kernel.value_tens(R) ;
                Tensor<3,dim> W = exterior_stokes_kernel.value_tens2(R) ;
                Tensor<2,dim> singular_ker = compute_singular_kernel(normals[q], W) ;
                for (unsigned int idim = 0; idim < dim; ++idim)
                  {
                    for (unsigned int jdim = 0; jdim < dim; ++jdim)
                      {
                        val_velocities(i+val_velocities.size()/dim*idim) +=  G[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                                                                             (*stokes_local_forces)[q](jdim) *
                                                                             internal_fe_v->JxW(q) ;
                        val_velocities(i+val_velocities.size()/dim*idim) -= singular_ker[idim][jdim] * //my_stokes_kernel.value(R, idim * dim * dim + jdim * dim) *
                                                                            (*stokes_local_velocities)[q](jdim) *
                                                                            internal_fe_v->JxW(q) ;
                      }
                  }
              }
          }
      }

  }

// Just a helper funtion to convert the parsed bool parameters to arrays of bools
  template <int dim>
  void BEMProblem<dim>::convert_bool_parameters()
  {
    wall_bool[0] = wall_bool_0;
    wall_bool[1] = wall_bool_1;
    wall_bool[2] = wall_bool_2;
    wall_bool[3] = wall_bool_3;
    wall_bool[4] = wall_bool_4;
    wall_bool[5] = wall_bool_5;
    wall_bool[6] = wall_bool_6;
    wall_bool[7] = wall_bool_7;
    flip_normal_wall_bool[0] = flip_normal_wall_bool_0;
    flip_normal_wall_bool[1] = flip_normal_wall_bool_1;
    flip_normal_wall_bool[2] = flip_normal_wall_bool_2;
    flip_normal_wall_bool[3] = flip_normal_wall_bool_3;
    flip_normal_wall_bool[4] = flip_normal_wall_bool_4;
    flip_normal_wall_bool[5] = flip_normal_wall_bool_5;
    flip_normal_wall_bool[6] = flip_normal_wall_bool_6;
    flip_normal_wall_bool[7] = flip_normal_wall_bool_7;
  }
  // In 2D it is a necessary function to avoid strange behaviours of the wall related stuff
  template <>
  void BEMProblem<2>::convert_bool_parameters()
  {
    wall_bool[0] = false;
    wall_bool[1] = false;
    wall_bool[2] = false;
    wall_bool[3] = false;
    wall_bool[4] = false;
    wall_bool[5] = false;
    wall_bool[6] = false;
    wall_bool[7] = false;
    flip_normal_wall_bool[0] = flip_normal_wall_bool_0;
    flip_normal_wall_bool[1] = flip_normal_wall_bool_1;
    flip_normal_wall_bool[2] = flip_normal_wall_bool_2;
    flip_normal_wall_bool[3] = flip_normal_wall_bool_3;
    flip_normal_wall_bool[4] = flip_normal_wall_bool_4;
    flip_normal_wall_bool[5] = flip_normal_wall_bool_5;
    flip_normal_wall_bool[6] = flip_normal_wall_bool_6;
    flip_normal_wall_bool[7] = flip_normal_wall_bool_7;
    create_box_bool=false;
    cylinder_create_bool=false;
    cylinder_import_bool=false;
    use_flagellum_handler=false;
    reflect_kernel=false;
    no_slip_kernel=false;
    first_index_box=0;
  }

  // Simply a function to rotate a Vector related to a FESystem with dofreordering
  template <int dim>
  void BEMProblem<dim>::rotate_vector(TrilinosWrappers::MPI::Vector &vec)
  {
    for (types::global_dof_index i = 0; i<vec.size()/dim; ++i)
      {
        if (vec.locally_owned_elements().is_element(i))
          {
            Vector<double> pos(dim), new_pos(dim);
            for (unsigned int idim=0; idim<dim; ++idim)
              pos[idim] = vec[i+idim*vec.size()/dim];
            rotation_matrix.vmult(new_pos, pos);
            // pcout<<pos<<" "<<new_pos<<std::endl;
            // Assert(new_pos.l2_norm()==pos.l2_norm(),ExcMessage("PIRLA"));
            for (unsigned int idim=0; idim<dim; ++idim)
              vec[i+idim*vec.size()/dim] = new_pos[idim];
          }
      }
    vec.compress(VectorOperation::insert);
  }
// @sect4{BEMProblem::run}
// This is the main function. It should be self explanatory in its
// briefness:
  template <int dim>
  void BEMProblem<dim>::run(unsigned int start_frame, unsigned int end_frame)
  {
    {
      Teuchos::TimeMonitor LocalTimer(*RunTime);

      //As first step we convert the bool parameters into vectors.
      convert_bool_parameters();

      // We retrieve the two Finite Element Systems, er translate the ParsedFiniteElement in shared_ptr to easu their usage
      fe_stokes = SP(parsed_fe_stokes());
      fe_map = SP(parsed_fe_mapping());

      // Secondly we need to read the reference domain, from which we will "evolve" our simulation
      read_domain();


      // We reinitialise the variables
      reinit();

      // if requested by the user we set up the flagellum handler
      if (use_flagellum_handler)
        flagellum_handler.compute_reference_euler_constant_spiral(reference_euler_vec);
      if (start_frame != 0)
        {
          // We may need to read from memory the previous state. This is in case of a restart.
          if (use_previous_state)
            {
              pcout<<"getting old stuff"<<std::endl;
              Vector<double> loc_rigid_puntual_displacements(dh_stokes.n_dofs());
              std::string file_name_displ = "stokes_rigid_displ_" + Utilities::int_to_string(start_frame-1) + ".bin";
              pcout<<"rigid displacement : "<<file_name_displ<<std::endl;
              std::ifstream if_displ(file_name_displ.c_str());
              loc_rigid_puntual_displacements.block_read(if_displ);
              for (auto iii : rigid_puntual_displacements.locally_owned_elements())
                rigid_puntual_displacements[iii] = loc_rigid_puntual_displacements[iii];
              rigid_puntual_displacements.compress(VectorOperation::insert);
              read_rotation_matrix(rotation_matrix, start_frame-1);
            }
        }

      // Compute the euler vector for the initial computational frame.
      compute_euler_vector(euler_vec, start_frame%n_frames, true);
      // Just a flag on the preconditioner assembly. Essential in case of direct preconditioning.
      reassemble_preconditoner = true;
      // We begin the standard solving loop for the motility BEMStokes.
      for (unsigned int i=start_frame; i<=end_frame; i=i+delta_frame)
        {
          // We need to compute the euler vector for the next step. We do so to compute the shape_velocities vector, which is the real starting point of each simulation.
          // We need to consider the same rotation so we get R*\dot{s}
          compute_euler_vector(next_euler_vec,(i+1)%n_frames, true);
          // Then we can compute the geometrical relevant quantities for the simulation, namely center of mass and rigid modes.
          compute_center_of_mass_and_rigid_modes(i);
          compute_normal_vector();
          // This line is to impose the constraint on the normal vector. We do not actually use it, we just need to set it to zero.
          // If zero it is ignored.
          i_single_layer_constraint = 0;
          // compute_constraints_for_single_layer();

          // If we do not compute a real motility problem we set next_euler_vec = euler_vec to set shape_velocities = 0.
          if (grid_type != "Real" )
            next_euler_vec=euler_vec;
          // shape_velocities computation, we need to take care of the case of discontiuos element. We have two different
          // FESystem for the mapping (to be conitnuos) and the unkwons (arbitrary).
          // If we know a priori the movement of the flagellum we shoould impose them exactly. Otherwise we use a finite difference strategy between two consecutive frames.
          if (imposed_rotation_as_flagellum_shape)
            {
              compute_rotational_shape_velocities(shape_velocities, N_flagellum_torque);
            }
          else if (imposed_traslation_as_flagellum_shape)
            {
              shape_velocities=0.;
              for (unsigned int i=0; i<dh_stokes.n_dofs()/dim; ++i)
                if (flagellum_cpu_set.is_element(i) && this_cpu_set.is_element(i))
                  shape_velocities[i]=1.;
              shape_velocities.compress(VectorOperation::insert);
              // compute_traslational_shape_velocities(shape_velocities, N_flagellum_translation);

            }
          else
            {
              // if (fe_map->get_name() == fe_stokes->get_name())
              //   for (auto i : this_cpu_set)
              //     {
              //       shape_velocities[i]=1./time_step*(next_euler_vec[i]-euler_vec[i]);
              //     }
              // else
              project_shape_velocities();
            }
          if (grid_type!="Real")
            shape_velocities = 0.;

          // A flag for debugging purposes
          bool compute=true;
          // If compute==true we assemble and solve the Stokes system, otherwise we will compute a test case for the rigid displacements
          if (compute)
            {
              pcout<<"Assembling"<<std::endl;
              if (galerkin)
                assemble_stokes_system_galerkin();
              else
                assemble_stokes_system(true);
              // Finally we can solve the system. We need to set the Monolithic resolution strategy to consider walls.
              if (reassemble_preconditoner && solve_directly == false && preconditioner_type=="Direct")
                {
                  Teuchos::TimeMonitor LocalTimer(*RefactTime);
                  pcout<<"refactorizing direct_preconditioner"<<std::endl;
                  direct_trilinos_preconditioner.set_up(solver_control);
                  if (monolithic_bool)
                    direct_trilinos_preconditioner.initialize(monolithic_system_matrix);
                  else
                    direct_trilinos_preconditioner.initialize(V_matrix);

                  reassemble_preconditoner = false;
                }
              solve_system(monolithic_bool);
            }
          // We can choose between two different time integration strategies. Standard Forward Euler or Heun predictor corrector.
          if (res_strategy == "Forward")
            {
              update_system_state(compute,i,bool_rot,bool_dipl,res_strategy);
            }
          else if (res_strategy == "Heun")
            {
              // We create the intermediate state and back up the original state
              old_rigid_velocities.reinit(rigid_velocities.size());
              old_rigid_displacements_for_sim.reinit(rigid_displacements_for_sim.size());

              update_system_state(compute,i,bool_rot,bool_dipl,"Forward");
              // We prepare for the new simulation
              reinit_for_new_time((i+1)%n_frames);
              compute_euler_vector(next_euler_vec,(i+2)%n_frames, true);
              compute_center_of_mass_and_rigid_modes(i+1);
              compute_normal_vector();
              i_single_layer_constraint = 0;
              if (grid_type != "Real" )
                next_euler_vec=euler_vec;
              if (imposed_rotation_as_flagellum_shape)
                {
                  compute_rotational_shape_velocities(shape_velocities, N_rigid[dim]);
                }
              else if (imposed_traslation_as_flagellum_shape)
                {
                  compute_traslational_shape_velocities(shape_velocities, N_rigid[0]);

                }
              else
                {
                  // if (fe_map->get_name() == fe_stokes->get_name())
                  //   for (auto i : this_cpu_set)
                  //     {
                  //       shape_velocities[i]=1./time_step*(next_euler_vec[i]-euler_vec[i]);
                  //     }
                  // else
                  project_shape_velocities();
                }
              bool compute = true;

              if (compute)
                {
                  pcout<<"Assembling"<<std::endl;
                  if (galerkin)
                    assemble_stokes_system_galerkin();
                  else
                    assemble_stokes_system(true);
                  solve_system(monolithic_bool);
                }
              update_system_state(compute,i,bool_rot,bool_dipl,res_strategy);


            }
          else
            {
              Assert(false, ExcNotImplemented());
            }
          // Now we have all the velocities, thus we can compute the total velocity everywhere on the boundary.
          total_velocities.sadd(0.,1.,shape_velocities);
          // for(unsigned int i = 0; i<num_rigid; ++i)
          //   total_velocities.sadd(1.,assemble_scaling*rigid_velocities[i],N_rigid[i]);

          total_velocities.sadd(1.,1.,rigid_puntual_velocities);
          total_velocities.sadd(1.,1.,wall_velocities);
          // We update the puntual displacements and perform the output operations.
          rigid_puntual_displacements = next_rigid_puntual_displacements;
          // We output the results of the simulation at each time step. Every time step is assumed to be independent if we don't consider rigid movements to update the system state.

          // We check that the soluton we have retrieved actually solves the BEM for Stokes.
          if (res_strategy=="Forward")
            {
              TrilinosWrappers::MPI::Vector dummy_1(this_cpu_set, mpi_communicator);
              TrilinosWrappers::MPI::Vector dummy_3(this_cpu_set, mpi_communicator);
              TrilinosWrappers::MPI::Vector dummy_2(this_cpu_set, mpi_communicator);
              V_matrix.vmult(dummy_1, stokes_forces);
              tangential_projector_body(total_velocities, dummy_2);
              K_matrix.vmult(dummy_3, dummy_2);
              tangential_projector_body(dummy_3, dummy_2);
              // WITH A WALL THE SECOND PROJECTOR INFLUENCES THE RESULT!!!
              dummy_3.sadd(0.,-1.,dummy_1);
              dummy_3.sadd(1.,1.,dummy_2);
              constraints.distribute(dummy_3);
              pcout<<"FINAL CHECK "<<dummy_3.linfty_norm()<<" : "<<dummy_3.l2_norm()<<std::endl;
              for (auto i : this_cpu_set)
                final_test[i]=dummy_3[i];
              pcout<<"FINAL CHECK "<<final_test.linfty_norm()<<" : "<<final_test.l2_norm()<<std::endl;

            }
          // We simply need to prepare again everything for a new solution(at a new time step).
          // We use compute_euler_vector to update the euler vector considering the new rotation matrix.
          output_save_stokes_results(i);
          reinit_for_new_time((i+delta_frame)%n_frames);
        }
      // tria.set_manifold(0);
      // tria.set_manifold(99);

      // if (extend_solution == true)
      //     compute_exterior_stokes_solution<2> ();
    }
    Teuchos::TimeMonitor::summarize();
    pcout<<"THE END"<<std::endl;
  }

// In this function we simply prepare all our vectors-matrices for a
// new cycle. We can't simply reinit every unkwown. We update also
// the euler_vec vector.
  template<int dim>
  void BEMProblem<dim>::reinit_for_new_time(const unsigned int frame)
  {
    // euler_vec = next_euler_vec;
    pcout<<"preparing for new time"<<std::endl;
    compute_euler_vector(euler_vec,frame%n_frames,true);
    next_euler_vec = 0.;
    rigid_puntual_velocities = 0.;
    stokes_rhs = 0.;
    for (unsigned int j=0; j<dim; ++j)
      {
        center_of_mass(j) = 0.;
        center_of_mass_body(j) = 0.;
      }
    Mass_Matrix = 0.;
    V_matrix = 0.;
    K_matrix = 0.;
    N_rigid.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_complete.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    DN_N_rigid.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_dual.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_rigid_dual_complete.resize(num_rigid, TrilinosWrappers::MPI::Vector (this_cpu_set,mpi_communicator));
    N_flagellum_torque.reinit(this_cpu_set,mpi_communicator);
    N_flagellum_translation.reinit(this_cpu_set,mpi_communicator);
    N_flagellum_torque_dual.reinit(this_cpu_set,mpi_communicator);
  }


  template <int dim>
  void BEMProblem<dim>::output_composed_stokes_results(const unsigned int cycle,const Vector<double> &loc_stokes_forces,const Vector<double> &loc_normal_vector,const Vector<double> &loc_rigid_puntual_displacements,const Vector<double> &loc_rigid_puntual_velocities,const Vector<double> &loc_shape_velocities)
  {
    MappingFEField<dim-1, dim> mapping_tot(map_dh,total_euler_vec);
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation
    (dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim-1, DoFHandler<dim-1, dim> > dataout;

    dataout.attach_dof_handler(dh_stokes);
    dataout.add_data_vector(loc_stokes_forces, std::vector<std::string > (dim,"global_stokes_forces"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
    dataout.add_data_vector(loc_normal_vector, std::vector<std::string > (dim,"n"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
    dataout.add_data_vector(loc_rigid_puntual_velocities, std::vector<std::string > (dim,"overall_rigid_vel"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
    dataout.add_data_vector(loc_rigid_puntual_displacements, std::vector<std::string > (dim,"rigid_displacement"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
    dataout.add_data_vector(loc_shape_velocities, std::vector<std::string > (dim,"shape_velocity"), DataOut<dim-1, DoFHandler<dim-1, dim> >::type_dof_data, data_component_interpretation);
    dataout.build_patches(*mappingeul,
                          fe_stokes->degree,
                          DataOut<dim-1, DoFHandler<dim-1, dim> >::curved_inner_cells);

    std::string filename = ( Utilities::int_to_string(dim) +
                             "d_stokes_composed_boundary_solution_" +
                             Utilities::int_to_string(cycle) +
                             ".vtu" );
    std::ofstream file(filename.c_str());

    dataout.write_vtu(file);

  }

  // In this function we compose all the solutions. Namely we read the results saved by a standard run and we mount them.
  template<int dim>
  void BEMProblem<dim>::composer(unsigned int start_frame, unsigned int end_frame)
  {
    pcout << "Composing existing solutions"<<std::endl;
    // read_parameters("../parameters.prm");
    convert_bool_parameters();

    fe_stokes = SP(parsed_fe_stokes());
    fe_map = SP(parsed_fe_mapping());

    read_domain();
    pcout<<"reinitialize"<<std::endl;

    reinit();

    // Basically the composer performs read and write from the memory. It is a serial application.
    if (this_mpi_process==0)
      {
        Vector<double> loc_rigid_puntual_displacements(dh_stokes.n_dofs());
        Vector<double> mean_rigid_velocities_short(num_rigid);
        for (unsigned int i=start_frame; i<end_frame; i=i+delta_frame)
          {
            // First we read from the memory.
            Vector<double> partial_rigid_displacement(dh_stokes.n_dofs());
            Vector<double> loc_stokes_forces(dh_stokes.n_dofs());
            Vector<double> loc_rigid_puntual_velocities(dh_stokes.n_dofs());
            Vector<double> loc_shape_velocities(dh_stokes.n_dofs());
            Vector<double> loc_normal_vector(dh_stokes.n_dofs());
            Vector<double> loc_rigid_velocities_short(num_rigid);
            Vector<double> loc_total_velocities(dh_stokes.n_dofs());
            std::string filename_displ, filename_vel, filename_total_vel, filename_shape_vel, filename_forces, filename_euler, filename_normals,filename_point, filename_rigid_vel_short;
            std::ofstream ofs_point;
            pcout<<"Reading data for frame: "<<i<<std::endl;
            filename_forces = "stokes_forces_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream forces(filename_forces.c_str());
            loc_stokes_forces.block_read(forces);
            filename_displ = "stokes_rigid_displ_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream displ(filename_displ.c_str());
            partial_rigid_displacement.block_read(displ);
            loc_rigid_puntual_displacements.sadd(1.,1.,partial_rigid_displacement);
            filename_vel = "stokes_rigid_vel_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream veloc(filename_vel.c_str());
            loc_rigid_puntual_velocities.block_read(veloc);
            filename_total_vel = "total_velocities_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream total_veloc(filename_total_vel.c_str());
            loc_total_velocities.block_read(total_veloc);
            filename_rigid_vel_short = "4_6_rigid_velocities_"+Utilities::int_to_string(i)+".bin";
            std::ifstream rv46 (filename_rigid_vel_short.c_str());
            loc_rigid_velocities_short.block_read(rv46);
            loc_rigid_velocities_short.print(std::cout);
            mean_rigid_velocities_short.sadd(1.,1.,loc_rigid_velocities_short);
            filename_shape_vel = "shape_velocities_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream s_vel(filename_shape_vel.c_str());
            loc_shape_velocities.block_read(s_vel);
            filename_euler = "euler_vec_" + Utilities::int_to_string(i) + ".bin";
            std::ifstream reuler (filename_euler.c_str());
            euler_vec.block_read(reuler);
            filename_normals = "normal_vector" + Utilities::int_to_string(i) + ".bin";
            std::ifstream rnorm (filename_normals.c_str());
            loc_normal_vector.block_read(rnorm);
            total_euler_vec.sadd(0.,1.,euler_vec);
            total_euler_vec.sadd(1.,1.,loc_rigid_puntual_displacements);
            if (i==start_frame)
              {
                mappingeul = SP(new MappingFEField<dim-1, dim>(map_dh,euler_vec));
              }
            pcout<<"Output"<<std::endl;
            // Then we save the results.
            output_composed_stokes_results(i, loc_stokes_forces, loc_normal_vector, loc_rigid_puntual_displacements, loc_rigid_puntual_velocities, loc_shape_velocities);
            filename_point="point_0_position.txt";
            ofs_point.open (filename_point, std::ofstream::out | std::ofstream::app);
            ofs_point << i << " ";
            for (unsigned int idim = 0; idim<dim; ++idim)
              {
                ofs_point << loc_rigid_puntual_displacements(0) << " ";
              }
            ofs_point << std::endl;
            ofs_point.close();
            pcout<<"Composed frame: "<<i<<std::endl;


          }
        std::string filename_mean_vel="mean_velocities_stroke.txt";
        std::ofstream ofs_vel;
        ofs_vel.open (filename_mean_vel, std::ofstream::out | std::ofstream::app);
        for (unsigned int i = 0; i<num_rigid; ++i)
          {
            ofs_vel << mean_rigid_velocities_short(i) << std::endl;
          }
        ofs_vel << std::endl;
        ofs_vel.close();

      }
    pcout<<"THE END"<<std::endl;


  }
}

template class BEMStokes::BEMProblem<2>;
template class BEMStokes::BEMProblem<3>;
