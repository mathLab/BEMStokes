#include "free_surface_kernel.h"

using namespace dealii;

int main(int argc, char **argv)
{
  for (unsigned int i=0; i<3; ++i)
    {
      std::cout<<"Testing a perfect slip kernel using a normal along the "<<i<<" axis"<<std::endl;

      FreeSurfaceStokesKernel<3> fs_kernel;
      double position = 1.;
      double tol = 1e-6;
      fs_kernel.set_wall_orientation(i);
      Point<3> source, source_image;

      Point<3> valuation_point;
      Tensor<1,3> R,R_image,R_image_old;

      valuation_point[i] = position;
      valuation_point[(i+1)%3] = 3.;

      Point<3> val_image(valuation_point);
      source_image = source;
      source_image[i] -= 2*(source[i]-position);
      val_image[i]-= 2*(valuation_point[i]-position);
      R = valuation_point - source;
      R_image = val_image - source;
      R_image_old = valuation_point -source_image;
      Tensor<2,3> G = fs_kernel.value_tens_image(R,R_image);
      Tensor<2,3> G_old = fs_kernel.value_tens_image_old(R,R_image_old);

      for (unsigned int ii=0; ii<3; ++ii)
        for (unsigned int jj=0; jj<3; ++jj)
          {
            if (std::abs(G[ii][jj]-G_old[ii][jj])<tol)
              std::cout<<"OK"<<std::endl;
            else
              std::cout<<"ERROR at ii = "<<ii<<" jj = "<<jj<<" , "<<G[ii][jj]<<" "<<G_old[ii][jj]<<std::endl;

          }
    }
}
