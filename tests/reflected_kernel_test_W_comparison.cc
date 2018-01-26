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
      Point<3> source;
      Point<3> source_image;
      Point<3> valuation_point;
      Tensor<1,3> R,R_image,R_image_old;

      valuation_point[i] = position+3.;
      valuation_point[(i+1)%3] = 3.;

      Point<3> val_image(valuation_point);
      source_image = source;
      val_image[i]-= 2*(valuation_point[i]-position);
      source_image[i]-= 2*(source[i]-position);
      R = valuation_point - source;
      R_image = val_image - source;
      R_image_old = valuation_point -source_image;
      // std::cout<<R_image<<" "<<R_image_old<<std::endl;
      Tensor<3,3> W = fs_kernel.value_tens_image2(R,R_image);
      Tensor<3,3> W_old = fs_kernel.value_tens_image2_old(R,R_image_old);

      for (unsigned int ii=0; ii<3; ++ii)
        {
          for (unsigned int jj=0; jj<3; ++jj)
            {
              for (unsigned int kk=0; kk<3; ++kk)
                {
                  if (std::abs(W[ii][jj][kk]-W_old[ii][jj][kk])>tol)
                    std::cout<<"ERROR at ii = "<<ii<<" jj = "<<jj<<" kk = "<<kk<<" , "<<W[ii][jj][kk]<<" , "<<W_old[ii][jj][kk]<<std::endl;
                  else
                    {
                      std::cout<<"OK"<<std::endl;
                    }

                }
            }
        }
    }
}
