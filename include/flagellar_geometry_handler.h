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

#ifndef __flagellar_geometry_handler_h
#define __flagellar_geometry_handler_h

#include <deal.II/base/smartpointer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/types.h>
#include <deal.II/base/index_set.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_control.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/sparsity_pattern.h>
// #include <deal.II/lac/compressed_simple_sparsity_pattern.h>
#include <deal.II/lac/sparse_ilu.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>

#include <deal.II/lac/block_sparsity_pattern.h>
// #include <deal.II/lac/trilinos_block_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
// #include <deal.II/lac/trilinos_block_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>

#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q_eulerian.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/numerics/data_out.h>
//#include <deal.II/numerics/data_out_dof_data.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

#include <cmath>
#include <iostream>
#include <fstream>
#include <string>


// #include <mpi.h>
#include <deal2lkit/parameter_acceptor.h>

// The last part of this preamble is to import everything in the dealii
// namespace into the one into which everything in this program will go:
namespace BEMStokes
{
  using namespace dealii;
  using namespace deal2lkit;



  /// This class difines a possible reconstruction of a flagellar geometry. It follows the geometry described by Thien in 1987, Ramia in 1993 and Shum in 2010.
  /// At the time being the geometry is deformed near the raccording section where the helix amplitude tends to 0. Alternatively we can use a a perfectly helix as flagellum.
  /// In such a case we can compare with the results by Purcell in 1997 are Rodenborm 2013. However we strongly advise to use the import of the grids which have been created already
  /// respecting the geometric constraints.
  template <int dim>
  class FlagellarGeometryHandler : public ParameterAcceptor
  {
  public:

    FlagellarGeometryHandler() {};

    ~FlagellarGeometryHandler()
    {
      if (map_flagellum_cpu_set)
        map_flagellum_cpu_set = NULL;
      if (mapping)
        mapping = NULL;
    };

    /// declare_parameters, parse_parameters are required by the ParameterAcceptor structure.
    virtual void declare_parameters(ParameterHandler &prm);

    virtual void parse_parameters(ParameterHandler &prm);

    /// We create the initial flagellum grid starting from a parallelepiped in codimension 1.
    void create_initial_flagellum_triangulation(Triangulation<dim-1,dim> &tria);

    void set_geometry_cache(const DoFHandler<dim-1,dim> &map_dh_in, const IndexSet *map_flagellum_cpu_set_in, const Mapping<dim-1,dim> *mapping_in);

    /// We compute the reference euler vector for a MappingFEField considering the spiral described in Thien 1987
    void compute_reference_euler(Vector<double> &reference_euler) const;

    /// Given a rotation of the flagellum we can compute the corresponding euler vector compting a rotation along an axis of the reference_euler.
    void compute_euler_at_theta(Vector<double> &euler, const Vector<double> &reference_euler, const double theta) const;

    /// We compute the reference euler vector for a MappingFEField considering the a perfect spiral as flagellum.
    void compute_reference_euler_constant_spiral(Vector<double> &reference_euler) const;

    void save_geometry(const std::string path_save, const std::string basename, const std::string extension, const unsigned int n_frames) const;

  private:
    double Nlambda, alpha, Lx, k, ke, delta_head_flagellum, a;

    SmartPointer<const DoFHandler<dim-1,dim> >         map_dh;
    const IndexSet *map_flagellum_cpu_set;
    const Mapping<dim-1,dim>*        mapping;


  };

}

#endif
