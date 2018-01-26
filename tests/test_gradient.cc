#ifndef __test_gradient_h__
#define __test_gradient_h__

#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/function_lib.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <fstream>
#include <deal.II/fe/mapping_q.h>

#include <Sacado.hpp>



using namespace dealii;

typedef Sacado::Fad::DFad<double> sacado_double;
typedef Sacado::Fad::DFad<sacado_double> sacado_sacado_double;

template<int dim, class Type>
void TestKernelGradient(const Kernel<dim, Type> &my_kernel, const unsigned int degree = 1, const unsigned int refinement = 3)
{
  HyperShellBoundary<dim> boundary;
  Triangulation<dim> tria;
  GridGenerator::hyper_shell(tria, Point<dim>(), .01, 1.0, 6);
  tria.set_boundary(0, boundary);
  tria.refine_global(refinement);
  FESystem<dim,dim> fe (FE_Q<dim> (degree), my_kernel.n_components);
  DoFHandler<dim> dh(tria);
  dh.distribute_dofs(fe);

//    Vector<double> ker_vec(dh.n_dofs());
//
//    Vector<double> ker_vec_err(dh.n_dofs());

  Vector<double> ker_vec_true(dh.n_dofs());

  Vector<double> ker_grad_err_euclidean_norm(dh.n_dofs());

  std::vector<Point<dim>> support_points(dh.n_dofs());

  MappingQ<dim>        mapping(degree);

//    Vector<double> difference_per_cell(tria.n_active_cells());
//
//    VectorTools::interpolate(dh, my_kernel, ker_vec);

  DoFTools::map_dofs_to_support_points<dim>( mapping, dh, support_points);

  for (unsigned int i=0; i<dh.n_dofs(); ++i)
    {
      Point<dim> gradient_exact;
      LaplaceKernel::kernels(support_points[i], gradient_exact, ker_vec_true(i));
//      Point<dim, sacado_double> sp;
//      for(unsigned int j=0; j<dim; ++j)
//      {
//        sp[j]=(support_points[i][j]);
//      }
//      sacado_double sacado_eval;
//      sacado_eval = my_kernel.t_value(sp);
      Tensor<1,dim> gradient;
      gradient = my_kernel.gradient(support_points[i]);
      Point<dim> grad_error = gradient - gradient_exact;

//      ker_vec_err(i) = sacado_eval.val()-ker_vec(i);
      ker_grad_err_euclidean_norm(i) = grad_error.norm();

    }

//    double linf_value=ker_vec_err.linfty_norm();
//    double l2_value=ker_vec_err.l2_norm();

  double linf_grad=ker_grad_err_euclidean_norm.linfty_norm();
  double l2_grad=ker_grad_err_euclidean_norm.l2_norm();


  std::cout<<"Gradient errors in dim = "<<dim<<std::endl;
  std::cout<<"L2 norm = "<< l2_grad <<" Linfty norm = "<< linf_grad <<std::endl;


}

#endif
