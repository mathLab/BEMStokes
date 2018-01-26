#include "no_slip_wall_kernel.h"
using namespace dealii;

int main(int argc, char **argv)
{
  for (unsigned int i=0; i<3; ++i)
    {
      std::cout<<"Testing a no slip kernel using a normal along the "<<i<<" axis"<<std::endl;

      NoSlipWallStokesKernel<3> ns_kernel;
      double position = 1.;
      double tol = 1e-6;
      ns_kernel.set_wall_orientation(i);
      Point<3> source;
      Point<3> source_image;
      Point<3> valuation_point;
      Tensor<1,3> R,R_image,R_image_old;

      valuation_point[i] = position;
      valuation_point[(i+1)%3] = 3.667;
      valuation_point[(i+2)%3] = 0.214456;
      source[i] = 6.67;
      source[(i+1)%3] = 3.234;
      source[(i+2)%3] = 9.234;
      std::cout<<"h_0 = "<<source[i]-position<<std::endl;
      Point<3> val_image(valuation_point);
      source_image = source;
      val_image[i]-= 2*(valuation_point[i]-position);
      source_image[i]-= 2*(source[i]-position);
      std::cout<<valuation_point<<" "<<val_image<<std::endl;
      R = valuation_point - source;
      R_image = val_image - source;
      R_image_old = valuation_point -source_image;
      std::cout<<"h_0_bis = "<<(R[i]-R_image[i])*0.5<<std::endl;
      // std::cout<<R_image-R<<std::endl;
      // std::cout<<R_image<<" "<<R_image_old<<std::endl;
      Tensor<3,3> W = ns_kernel.value_tens_image2(R,R_image);
      // for(unsigned int ii=0; ii<3; ++ii)
      // {
      //   for(unsigned int jj=0; jj<3; ++jj)
      //   {
      //     for(unsigned int kk=0; kk<3; ++kk)
      //     {
      //       if(std::abs(W[ii][jj][kk]-W_old[ii][jj][kk])>tol)
      //         std::cout<<"ERROR at ii = "<<ii<<" jj = "<<jj<<" kk = "<<kk<<" , "<<W[ii][jj][kk]<<" , "<<W_old[ii][jj][kk]<<std::endl;
      //       else
      //       {
      //         std::cout<<"OK"<<std::endl;
      //       }
      //
      //     }
      //   }
      // }
      for (unsigned int j=0; j<3; ++j)
        {
          for (unsigned int k=0; k<3; ++k)
            {
              for (unsigned int t=0; t<3; ++t)
                {
                  // std::cout<<W[j][k][t]<<" ";
                  if (std::abs(W[j][k][t])<tol)
                    std::cout<<"OK"<<std::endl;
                  else
                    std::cout<<"ERROR at j = "<<j<<" k = "<<k<<" , "<<W[j][k][t]<<std::endl;
                }
            }

        }
    }
}
