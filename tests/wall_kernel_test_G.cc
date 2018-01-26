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

      Tensor<2,3> G = ns_kernel.value_tens_image(R,R_image);

      for (unsigned int j=0; j<3; ++j)
        {
          for (unsigned int k=0; k<3; ++k)
            {
              if (std::abs(G[j][k])<tol)
                std::cout<<"OK"<<std::endl;
              else
                std::cout<<"ERROR at j = "<<j<<" , "<<" : k = "<<k<<" , "<<G[j][k]<<std::endl;
            }

        }
    }
}
