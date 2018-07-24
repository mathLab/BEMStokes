// ---------------------------------------------------------------------
//
// Copyright (C) 2014 - 2018 by the BEMStokes authors.
//
// This file is part of the BEMStokes library.
//
// The BEMStokes is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License version 2.1 as published by the Free Software Foundation.
// The full text of the license can be found in the file LICENSE at
// the top level of the BEMStokes distribution.
//
// Authors: Nicola Giuliani, Luca Heltai, Antonio DeSimone
//
// ---------------------------------------------------------------------

#ifndef __bem_stokes_h
#define __bem_stokes_h
#include <deal.II/base/smartpointer.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/quadrature_selector.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/types.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>



// And here are a few C++ standard header files that we will need:
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>

#include <kernel.h>
#include <repeated_kernel.h>
#include <free_surface_kernel.h>
#include <no_slip_wall_kernel.h>
#include <direct_preconditioner.h>
#include <flagellar_geometry_handler.h>

#include <mpi.h>
#include <deal2lkit/parameter_acceptor.h>
// #include <deal2lkit/error_handler.h>
// #include <deal2lkit/parsed_function.h>
#include <deal2lkit/parsed_finite_element.h>
#include <deal2lkit/parsed_quadrature.h>
#include <deal2lkit/utilities.h>


namespace BEMStokes
{
  /// The last part of this preamble is to import everything in the dealii
  /// namespace into the one into which everything in this program will go
  using namespace dealii;
  using namespace deal2lkit;


  // @sect3{Single and double layer operator kernels}

  // First, let us define a bit of the boundary integral equation machinery.



  // @sect3{The BEMProblem class}

  // The structure of a boundary element method code is very similar to the
  // structure of a finite element code, and so the member functions of this
  // class are like those of most of the other tutorial programs. In
  // particular, by now you should be familiar with reading parameters from an
  // external file, and with the splitting of the different tasks into
  // different modules. The same applies to boundary element methods, and we
  // won't comment too much on them, except on the differences.
  template <int dim>
  class BEMProblem : public ParameterAcceptor
  {
  public:

    /// The setting of the projection type for a future integration of manifold descriptors as CAD files.
    enum ProjectionType
    {
      NormalProjection = 0,
      DirectionalProjection = 1,
      NormalToMeshProjection = 2
    };

    /// The constructor is extremely simple to respect the structure of the parameter handlers of the deal2lkit library.
    BEMProblem(MPI_Comm mpi_commy=MPI_COMM_WORLD);

    /// Simple class destructor.
    virtual ~BEMProblem();

    /// Class declaring all the parameter to be read by parametric file. We have to respect the dealii structure of the ParameterHandler.
    virtual void declare_parameters(ParameterHandler &prm);

    /// Actual parser for the parameters.
    virtual void parse_parameters(ParameterHandler &prm);

    /// This class reads the mesh corresponding to the selected frame. The library stores the path_to_grid, the grid_basename together with its extensions. Thus
    /// it expects the file to be path_to_grid+grid_basename+frame+extension. We accept grid in vtk, msh, inp formats.
    void read_input_mesh_file(unsigned int frame, Triangulation<dim-1,dim> &triangulation);

    /// The function reads the reference domain configuration for the current simulation. It automatically handles 2d and 3d simulations. The function also takes care of
    /// the manifold settings for the library. In particular we can set an iges descriptor for the flagellum (experimental) or SphericalManifolds if we want to perform
    /// benchmark application on spheres.
    void read_domain();

    /// This function takes care of the actual setting of the iges descriptor for the flagellum. At the time being this function is extremely slow and needs revision.
    void apply_flagellum_iges(Triangulation<dim-1,dim> &triangulation, std::string cad_file_name);

    /// This function create a cylindrical wall around the swimmer. This function is only implemented for dim=3. It needs several inputs: wall_type determines the kind
    /// of interface (at the moment free_surface interfaces can't be modelled using a cylinder), the direction of the cylinder axis, the location of the center of the axis,
    /// then the radius and the height. Finally we can choose to apply the CylindricalManifold, if this is not the case we simply get a parallelepiped as wall shape.
    void create_cylindrical_wall(Triangulation<dim-1, dim> &triangulation_wall,
                                 const std::string &wall_type, const Point<dim> &direction, const Point<dim> &point_on_axis,
                                 const double radius, const double heigth, const bool apply_manifold);

    /// The function providing the possibility of importing an existing cylindrical wall, we need to provide: the filename, the kind
    /// of interface (at the moment free_surface interfaces can't be modelled using a cylinder), the direction of the cylinder axis, the location of the center of the axis,
    /// then the radius and the height. Finally we can choose to apply the CylindricalManifold, if this is not the case we simply get a parallelepiped as wall shape.
    void import_cylinder(Triangulation<dim-1, dim> &triangulation_wall, const std::string &filename,
                         const std::string &wall_type, const Point<dim> &direction, const Point<dim> &point_on_axis,
                         const bool apply_manifold, const bool flip_all);

    /// This functions simply creates a coarse wall given the kind of interface the position of the wall center and its spans. This is a very simple function but it used by the create_wall routine
    void create_coarse_wall(Triangulation<dim-1, dim> &triangulation_wall, std::string &wall_type, Point<dim> &position, std::vector<double> &span, bool flip_normal=false) const;

    /// This function creates the numerical box (ideally) surrounding the existing triangulation. It only needs the existing triangulation and the starting_index for the walls to be created and
    /// included in the box. This function only works for dim=3 and it expects 6 different walls.
    void create_box(Triangulation<dim-1, dim> &triangulation, unsigned int starting_index);

    /// This function creates a refined wall of a given kind, it asks for the wall center position, its spans and the kind. The wall number is required to save as inp file the refiend triangulation.
    /// This file is used by the add_wall_to_tria or add_box_to_tria to load the triangulation_wall and add it to the complete tria.
    void create_wall(Triangulation<dim-1, dim> &triangulation, std::string wall_type, Point<dim> position, std::vector<double> span, unsigned int wall_number, bool flip_normal=false);

    /// This function simply add the created triangulation_wall to the existing swimmer triangulation.
    void add_wall_to_tria(Triangulation<dim-1, dim> &triangulation, unsigned int wall_number);

    /// This function simply add the created triangulation_box to the existing swimmer triangulation.
    void add_box_to_tria(Triangulation<dim-1, dim> &triangulation, std::string="box.inp");

    /// This function simply add the created triangulation_cyl to the existing swimmer triangulation.
    void add_cylinder_to_tria(Triangulation<dim-1, dim> &triangulation, bool apply_manifold, std::string="cylinder.inp");

    /// Function that locally refine the walls. It checks for the cells to be inside the max_distance from the center for refinement and to have a diameter larger than the threshold. It
    /// also requires the center of the wall for refinements issues.
    void refine_walls(Triangulation<dim-1, dim> &triangulation, const double max_distance, const double threshold, const Point<dim> &center, bool gradual_refinement=false);

    /// If we consider a generic triangulation we may end up with different material_id. If a hanging  node is present on such an interface we may end up with some discretisation errors.
    /// Therefore we take care of such possibility.
    void remove_hanging_nodes_between_different_material_id(Triangulation<dim-1,dim> &tria_in,
                                                            const bool isotropic=false,
                                                            const unsigned int max_iterations=100);

    /// This functions takes care of the reinitialisation of all the needed vectors and matrices for a fully parallel MPI simulation.
    void reinit();

    /// This functions builds a constraint for the single layer operator to make the solution of a pure dirichelt problem unique. This is due to the fact that in Stokes BIE the
    /// single layer V is singular if you consider the normal vector due to the hydrostatic pressure.
    void compute_constraints_for_single_layer();

    /// Since we study swimmers near interfaces we need to take great care logically splitting all the degrees of freedom. This function takes care of the definition of all the IndexSets
    /// to properly handle all such interfaces.
    void create_wall_body_index_sets();

    /// The euler_vector is the vector defining the Mapping we use in the BEM. The Mapping defines how each cells is maped to the effective geometry. In the present application we
    /// have chosen to using the MappingFEField class, thus the euler_vector represents the actual position of the support points used to define the geometry. We define the FiniteElement
    /// for the mapping from parametric file and it is required to be continuos.
    void compute_euler_vector(Vector<double> &vector, unsigned int frame, bool consider_displacements = false);//, unsigned int mag_factor = 1);

    /// We provide the option of not using isoparametric BEMs, thus we may need to project the shape_velocities vector (which is the real input datum of the problem) from the mapping
    /// discretisation to the finite dimensional space used to discretise the real unknowns of the problem.
    void project_shape_velocities(unsigned int frame = numbers::invalid_unsigned_int);

    /// In many benchamark application we can directly apply a rotational velocity to a part of the swimmer, at the moment such a velocity is applied using the rotation mode selected and
    /// the velocity are computed for the unknowns identified by this_cpu_set and flagellum_cpu_set.
    void compute_rotational_shape_velocities(TrilinosWrappers::MPI::Vector &rotational_shape_vel,const TrilinosWrappers::MPI::Vector &rotation_mode);

    /// In many benchamark application we can directly apply a linear uniform velocity to a part of the swimmer, at the moment such a velocity is applied using the traslation mode selected and
    /// the velocity are computed for the unknowns identified by this_cpu_set and flagellum_cpu_set.
    void compute_traslational_shape_velocities(TrilinosWrappers::MPI::Vector &traslation_shape_vel,const TrilinosWrappers::MPI::Vector &traslation_mode);

    /// This functions computes geometrical properties for the actual configuration of the swimmer. Firstly the geometrical center of mass is computed. Then the function computes the N rigid modes
    /// using the point point_force_pole as pole for the torques. point_force_pole is selected by the user through the parameter file.
    void compute_center_of_mass_and_rigid_modes(unsigned int frame);

    /// In this function we compute the global component(from 0 to dim-1) of the local degree of freedom.
    void compute_global_components();

    /// This functions takes care of the assembling of all the operators using a collocation scheme. We build the single layer operator V, the double layer K and the monolithic_system_matrix A.
    /// The matrix A depends on the different boundary conditions. The monolithic_system_preconditioner_matrix is the same as A with an identity submatrix for what regads the rigid velocity interactions.
    /// This is necessary to provide a proper preconditioner. We can select if we want a correction on the operator V based on the normal vector as prescribed by Olaf Steinbach in "Numerical Approximation
    /// Methods for Elliptic Boundary Value Problems".
    void assemble_stokes_system(bool correction_on_V=true);

    void assemble_monolithic_preconditioner();

    /// This functions takes care of the assembling of all the operators using a Galerkin scheme. We build the single layer operator V, the double layer K and the monolithic_system_matrix A.
    /// The matrix A depends on the different boundary conditions. The monolithic_system_preconditioner_matrix is the same as A with an identity submatrix for what regads the rigid velocity interactions.
    /// This is necessary to provide a proper preconditioner. At the time being this is just experimental and it is not fit for real simulations.
    void assemble_stokes_system_galerkin(bool correction_on_V=true);

    /// This function computes the normal vector. Such a vector is essential because it approximates the first eigenvector of the Single Layer V corresponding to a null eigenvalue.
    void compute_normal_vector();

    /// The dirichlet to neumann map T retrieves the force associated to a given input velocity. T can be approximated by T: V^-1 K, so it is bijective if the Single Layer operator is not singular.
    /// This function is used if we don't prescribe a monolithic resolution. In such a case we literally build the N rigid x N rigid resistive matrix and we solve the problem.
    void dirichlet_to_neumann_operator(const TrilinosWrappers::MPI::Vector &scr_vel, TrilinosWrappers::MPI::Vector &dst_force);

    /// Tangential projector considering the normal vector to the entire domain.
    void tangential_projector(const TrilinosWrappers::MPI::Vector &scr_vel, TrilinosWrappers::MPI::Vector &dst_vel);

    /// Tangential projector considering the normal vector to the sole body.
    void tangential_projector_body(const TrilinosWrappers::MPI::Vector &scr_vel, TrilinosWrappers::MPI::Vector &dst_vel);

    /// This function takes care of the resolution of the linear system. The only choice lies in the monolithic_booly parameters. If true the monolithic_system_matrix is inverted and we
    /// recover directly stresses and rigid_velocities, otherwise we proced by solving the motility assembling the grand resistive matrix through the resolution of N rigid dirichlet_to_neumann_operator.
    /// If the monolithic_booly is true we automatically select the preconditioner following the user requirements.
    void solve_system(bool monolithic_booly=false);

    /// Given a quaternion we compute the corresponding rotaion matrix R.
    void compute_rotation_matrix_from_quaternion(FullMatrix<double> &rotation, const Vector<double> &q);

    /// This function update the rotation matrix R using its representation using quaternions. By default it uses a forward euler scheme to update the quaternion.
    void update_rotation_matrix(FullMatrix<double> &rotation, const Vector<double> omega, const double dt, const bool forward_euler=true, const double theta=0.);

    /// Function rotating a Point along an axis provided the rotation angle.
    void apply_rotation_along_axis(Point<dim> &out, const Point<dim> &in, const Point<dim> &axis, const double angle);

    void update_system_state(bool compute, unsigned int frame, bool consider_rotations = false, bool consider_displacements = false, std::string res_system="Forward");

    /// Function computing the proper singular quadrature formulas for the integration of weakly singular kernels.
    const Quadrature<dim-1> & get_singular_quadrature(const unsigned int index) const;

    /// Function computing the FEValues corresponding built using the singular quadrature formulas for the integration of weakly singular kernels.
    FEValues<dim-1, dim> & get_singular_fe_values(const unsigned int index) const;

    Tensor<2,dim> compute_singular_kernel(const Tensor<1, dim> &normal,
                                          const Tensor<3,dim> &value2) const;

    /// Given the settings of the current simulation we can compute the first Green kernel G. It only uses one of the possible kernels depending on the input bool parameters.
    Tensor<2,dim> compute_G_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image,
                                   const StokesKernel<dim> &stokes_kernel,
                                   const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel,
                                   const NoSlipWallStokesKernel<dim> &ns_stokes_kernel,
                                   const bool reflect=false, const bool no_slip=false) const;

    /// Given the settings of the current simulation we can compute the second Green kernel W. It only uses one of the possible kernels depending on the input bool parameters.
    Tensor<3,dim> compute_W_kernel(const Tensor<1, dim> &R, const Tensor<1, dim> &R_image,
                                   const StokesKernel<dim> &stokes_kernel,
                                   const FreeSurfaceStokesKernel<dim> &fs_stokes_kernel,
                                   const NoSlipWallStokesKernel<dim> &ns_stokes_kernel,
                                   const bool reflect=false, const bool no_slip=false) const;

    /// Given the settings of the current simulation we can compute the third Green hypersingular kernel D. It only uses one of the possible kernels depending on the input bool parameters. NOT WORKING!!!
    Tensor<2,dim> compute_hypersingular_kernel(const Tensor<1, dim> &normal_y,
                                               const Tensor<1, dim> &normal_x,
                                               const Tensor<4, dim> &value3) const;


    /// This function approximate by means of finite differences the velocity gradient. This is useful in testing the specific kernels representing the different interfaces.
    void approximate_velocity_gradient(
      const std::vector<Point<dim> > &val_points,
      const Vector<double> &vel,
      const Vector<double> &forces,
      const double h,
      std::vector<Tensor<2,dim> > &val_gradients);


    /// This function evaluates the Stokes system on a given array of input points given the values of velocities and stresses on the boundary. Note that such vectors are not distributed among processors.
    void evaluate_stokes_bie(
      const std::vector<Point<dim> > &val_points,
      const Vector<double> &vel,
      const Vector<double> &forces,
      Vector<double> &val_velocities);

    /// This function evaluates the Stokes system on a given array of input points ON THE BOUNDARY given the values of velocities and stresses on the boundary. Note that such vectors are not distributed among processors.
    /// The function is slower than evaluate_stokes_bie given the necessity of using the singular quadrature formula for the weakly singular kernels.
    void evaluate_stokes_bie_on_boundary(
      const std::vector<Point<dim> > &val_points,
      const Vector<double> &vel,
      const Vector<double> &forces,
      Vector<double> &val_velocities);

    /// Helper function to save the rotation matrix as a binary file.
    void save_rotation_matrix(const FullMatrix<double> &rotation, const unsigned int frame);

    /// Helper function to load the rotation matrix from a binary file.
    void read_rotation_matrix(FullMatrix<double> &rotation, const unsigned int frame);

    /// The function producing the real output for the BEMStokes simulation. It Basically produces 2 vtk files, the first one represents the real simulation output, while the second one
    /// denoted by "foo" is useful for debugging purposes.
    void output_save_stokes_results(const unsigned int cycle);

    /// Helper function to convert the bool parameter into vector of boolean.
    void convert_bool_parameters();


    /// This function guides the actual BEM simulation for micromotility studies. Basically it loads the grids reporting the shape change frame by frame. It approximates such
    /// velocity shape changes and it recovers the overall rigid motions. It can operate both in 2 d and 3 d.  The output of the simulation are many varying from standard vtk format
    /// files reporting the punctual stresses and velocities together with the complete set of geometric informations as the N rigid modes. Moreover all the rigid velocities are saved
    /// in txt format for post process and analysis purposes. Moreover, it saves as binary files a complete set of vectors that can be used for the stop and restart procedure or for
    /// post process issues.
    void run(unsigned int start_frame=0, unsigned int end_frame=119);

    /// At each time step there are some quantities (vectors and matrices) that need to be reinitialise for the simulation to run properly. This function takes care of such issue.
    void reinit_for_new_time(const unsigned int frame);

    void output_composed_stokes_results(const unsigned int cycle,const Vector<double> &loc_stokes_forces,const Vector<double> &loc_normal_vector,const Vector<double> &loc_rigid_puntual_displacements,const Vector<double> &loc_rigid_puntual_velocities,const Vector<double> &loc_shape_velocities);

    /// The second possibility for the usage of BEMStokes is to simply compose together already existing results. This is what the composer organise. Namely it reads from the memory
    /// the stored results (from the SAME directory) and it produces some overall output using the output_composed_stokes_results (which includes rigid displacements) or mean values as the
    /// mean rifid velocities. This function is particularly helpful if we need to run BEM simulation without considering rigid displacements or rotations. In such a case every
    /// frame is independent and the displacements/rotations are only useful for a posteriori evaluations.
    void composer(unsigned int start_frame=0, unsigned int end_frame=120);

    /// Helper function that rotates a distributed vector using the Rotation matrix R
    void rotate_vector(TrilinosWrappers::MPI::Vector &vec);

    /// Helper function that counter-rotates a distributed vector using the Rotation matrix R TO BE IMPLEMENTED.
    void counter_rotate_vector(TrilinosWrappers::MPI::Vector &vec);


    Triangulation<dim-1, dim>   tria;
    Triangulation<dim-1, dim>   tria_unrefined;

    std::shared_ptr<Manifold<dim-1, dim> > manifold;
    std::shared_ptr<Manifold<dim-1, dim> > manifold_1, manifold_2;
    std::shared_ptr<Manifold<dim-1, dim> > flagellum_manifold;
    std::shared_ptr<Manifold<dim-1, dim> > cylinder_manifold;

    IndexSet this_cpu_set;
    IndexSet map_cpu_set;
    IndexSet monolithic_cpu_set;

    /// IndexSets to split different type of domain (body or wall)
    IndexSet body_cpu_set;
    IndexSet flagellum_cpu_set;
    IndexSet head_cpu_set;
    IndexSet wall_no_slip_cpu_set;
    IndexSet wall_free_surface_cpu_set;
    IndexSet wall_do_nothing_cpu_set;
    IndexSet wall_neumann_cpu_set;
    IndexSet wall_dirichlet_cpu_set;

    /// IndexSets to split different type of domain (body or wall), needed in case of discontinuos galerkin
    IndexSet map_body_cpu_set;
    IndexSet map_flagellum_cpu_set;
    IndexSet map_head_cpu_set;
    IndexSet map_wall_no_slip_cpu_set;
    IndexSet map_wall_free_surface_cpu_set;
    IndexSet map_wall_do_nothing_cpu_set;
    IndexSet map_wall_neumann_cpu_set;
    IndexSet map_wall_dirichlet_cpu_set;


    Vector<double>          euler_vec;
    Vector<double>          reference_euler_vec;

    // Stokes part
    std::unique_ptr<FiniteElement<dim-1, dim> > fe_stokes;
    std::unique_ptr<FiniteElement<dim-1, dim> > fe_map;
    // FESystem<dim-1, dim>  fe_map;
    // FESystem<dim-1, dim>  fe_stokes;
    DoFHandler<dim-1, dim> map_dh;
    DoFHandler<dim-1, dim> dh_stokes;

    FlagellarGeometryHandler<dim> flagellum_handler;

    shared_ptr<Mapping<dim-1, dim> > mappingeul;
    TrilinosWrappers::PreconditionBase *monolithic_system_preconditioner;
    std::vector<Point<dim> > reference_support_points;

    TrilinosWrappers::MPI::Vector              V_x_normals;
    TrilinosWrappers::MPI::Vector              V_x_normals_body;
    TrilinosWrappers::MPI::Vector              vnormals;
    TrilinosWrappers::MPI::Vector              final_test;
    std::vector<TrilinosWrappers::MPI::Vector > CheckMatrix;

    unsigned int singular_quadrature_order;

    ParsedQuadrature<dim-1> quadrature;

    ParsedQuadrature<dim-1> quadrature_ext;

    ParsedFiniteElement<dim-1, dim> parsed_fe_stokes;
    ParsedFiniteElement<dim-1, dim> parsed_fe_mapping;

    SolverControl solver_control;

    unsigned int n_frames;
    unsigned int n_cycles;
    unsigned int n_subdivisions;
    unsigned int external_refinement;
    unsigned int num_rigid;

    unsigned int delta_frame;

    bool rotation_correction;
    bool imposed_rotation_as_flagellum_shape;
    bool imposed_traslation_as_flagellum_shape;
    bool gradual_wall_refinement;
    bool extend_solution;

    bool apply_iges, apply_iges_to_frame, remove_tria_anisotropies;

    bool extra_debug_info;

    bool use_flagellum_handler;

    std::string singular_quadrature_type;
    std::string input_grid_path;
    std::string input_grid_base_name;
    std::string input_grid_format;
    std::string input_velocity_path;
    std::string squirming_velocity_basename;
    std::string input_iges_file_1;
    std::string res_strategy;
    std::string velocity_type;

    bool build_sphere_in_deal;
    unsigned int internal_sphere_refinements;
    unsigned int imposed_component;

    StokesKernel<dim> stokes_kernel;//stokes_kernel;
    StokesKernel<dim> exterior_stokes_kernel;//LH_exterior_stokes_kernel;

    FreeSurfaceStokesKernel<dim> fs_stokes_kernel;
    FreeSurfaceStokesKernel<dim> fs_exterior_stokes_kernel;

    NoSlipWallStokesKernel<dim> ns_stokes_kernel;
    NoSlipWallStokesKernel<dim> ns_exterior_stokes_kernel;

    TrilinosWrappers::SparseMatrix   V_matrix;

    // Needed for UMFPACK
    // SparsityPattern V_sp;
    // SparseMatrix<double> V_matrix_sparse;

    TrilinosWrappers::SparseMatrix    K_matrix;

    // TrilinosWrappers::SparseMatrix    D_matrix;

    TrilinosWrappers::SparseMatrix    monolithic_system_matrix;

    TrilinosWrappers::SparseMatrix    monolithic_system_preconditioner_matrix;

    TrilinosWrappers::SparseMatrix   NdiadicN;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid_complete;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid_map;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid_map_complete;
    std::vector<TrilinosWrappers::MPI::Vector >  DN_N_rigid;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid_dual;
    std::vector<TrilinosWrappers::MPI::Vector >  N_rigid_dual_complete;
    TrilinosWrappers::MPI::Vector N_flagellum_torque, N_flagellum_torque_dual, N_flagellum_translation;
    double flagellum_torque, flagellum_omega;
    double spheres_distance;
    bool two_spheres;
    bool spherical_head;
    // CompressedSimpleSparsityPattern csp_prec;

    TrilinosWrappers::SparsityPattern sparsity_pattern;
    // SparseMatrix<double> V_prec;

    TrilinosWrappers::SparseMatrix    Mass_Matrix;

    TrilinosWrappers::MPI::Vector stokes_forces;
    TrilinosWrappers::MPI::Vector stokes_rhs;
    TrilinosWrappers::MPI::Vector global_components;
    TrilinosWrappers::MPI::Vector shape_velocities;
    TrilinosWrappers::MPI::Vector total_velocities;

    Vector<double>          rigid_displacements_for_sim;
    Vector<double>          old_rigid_displacements_for_sim;
    Vector<double>          next_euler_vec;
    Vector<double>          total_euler_vec;

    Vector<double>          rigid_velocities;
    Vector<double>          baricenter_rigid_velocities;
    Vector<double>          old_rigid_velocities;
    Vector<double>          rigid_total_forces;
    Vector<double>          rigid_total_forces_complete;

    TrilinosWrappers::MPI::Vector rigid_puntual_velocities;
    TrilinosWrappers::MPI::Vector wall_velocities;
    TrilinosWrappers::MPI::Vector rigid_puntual_translation_velocities;
    TrilinosWrappers::MPI::Vector next_rigid_puntual_displacements;
    TrilinosWrappers::MPI::Vector rigid_puntual_displacements;

    TrilinosWrappers::MPI::Vector monolithic_solution;
    TrilinosWrappers::MPI::Vector monolithic_rhs;

    TrilinosWrappers::MPI::Vector normal_vector;
    TrilinosWrappers::MPI::Vector M_normal_vector;
    TrilinosWrappers::MPI::Vector normal_vector_pure;
    TrilinosWrappers::MPI::Vector M_normal_vector_pure;
    double l2normGamma;
    double l2normGamma_pure;

    Vector<double> first_evec;
    Vector<double> M_first_evec;

    double l2normGamma_evec;
    double angle_for_imposed_stuff;

    Point<dim> refinement_center;

    Point<dim> center_of_mass;
    Point<dim> center_of_mass_body;
    Point<dim> point_force_pole;
    Point<dim> force_arbitrary_point;

    double time_step;

    types::global_dof_index preconditioner_bandwith;

    TrilinosWrappers::SparsityPattern preconditioner_sparsity_pattern;

    TrilinosWrappers::SparsityPattern full_sparsity_pattern;

    TrilinosWrappers::SparsityPattern monolithic_full_sparsity_pattern;

    TrilinosWrappers::SparsityPattern *monolithic_preconditioner_sparsity_pattern;


    TrilinosWrappers::SparseMatrix band_system;

    TrilinosWrappers::PreconditionAMG::AdditionalData prec_data;

    SolverGMRES<TrilinosWrappers::MPI::Vector >::AdditionalData gmres_additional_data;

    ConstraintMatrix     constraints;
    // ConstraintMatrix     constraints_null;

    types::global_dof_index i_single_layer_constraint;

    bool reassemble_preconditoner;

    FullMatrix<double> rotation_matrix;
    FullMatrix<double> old_rotation_matrix;

    bool bool_rot;

    bool galerkin;

    unsigned int num_convergence_cycle;
    bool bool_dipl;
    bool bool_dipl_x;
    bool bool_dipl_y;
    bool bool_dipl_z;

    bool monolithic_bool;
    bool solve_directly;
    bool solve_with_torque;

    unsigned int gmres_restart;

    std::string grid_type;
    std::string force_pole;
    std::string preconditioner_type;

    bool bandwith_preconditioner;

    unsigned int bandwith;


    // std::vector<bool> wall_bools;
    bool create_box_bool;
    bool use_internal_alpha;
    unsigned int first_index_box;

    bool cylinder_create_bool, cylinder_import_bool, cylinder_manifold_bool, cylinder_flip_normal_bool;
    double cylinder_radius, cylinder_heigth;
    Point<dim> cylinder_direction, cylinder_point_on_axis;
    std::string cylinder_mesh_filename;
    std::string cylinder_wall_type;

    bool wall_bool_0;
    bool wall_bool_1;
    bool wall_bool_2;
    bool wall_bool_3;
    bool wall_bool_4;
    bool wall_bool_5;
    bool wall_bool_6;
    bool wall_bool_7;

    bool flip_normal_wall_bool_0;
    bool flip_normal_wall_bool_1;
    bool flip_normal_wall_bool_2;
    bool flip_normal_wall_bool_3;
    bool flip_normal_wall_bool_4;
    bool flip_normal_wall_bool_5;
    bool flip_normal_wall_bool_6;
    bool flip_normal_wall_bool_7;

    bool use_previous_state;
    bool reflect_kernel;
    bool no_slip_kernel;

    std::vector<bool> wall_bool;
    std::vector<bool> flip_normal_wall_bool;

    std::vector<std::string > wall_types;

    std::vector<Point<dim> > wall_positions;

    std::vector<std::vector<double> > wall_spans;

    double wall_threshold;

    double assemble_scaling;

    double refine_distance_from_center;

    Vector<double> initial_quaternion;

    DirectPreconditioner direct_trilinos_preconditioner;

    MPI_Comm mpi_communicator;
    unsigned int n_mpi_processes;
    unsigned int this_mpi_process;
    ConditionalOStream pcout;
    ConditionalOStream dpcout;
  };

}

#endif
