#include <bem_stokes.h>
#include <mpi.h>


template<int my_dim>
void impose_G_as_velocity(const BEMStokes::BEMProblem<my_dim> &bem, const Point<my_dim> &source, TrilinosWrappers::MPI::Vector &G_velocities)
{
  std::vector<Point<my_dim> > support_points(bem.dh_stokes.n_dofs());
  DoFTools::map_dofs_to_support_points<my_dim-1, my_dim> (*bem.mappingeul,
                                                          bem.dh_stokes, support_points);

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
                                                         bem.dh_stokes, support_points);

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
