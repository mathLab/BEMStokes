#include <free_surface_kernel.h>

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

      Point<3> valuation_point;
      Tensor<1,3> R,R_image;

      valuation_point[i] = position;
      valuation_point[(i+1)%3] = 3.;

      Point<3> val_image(valuation_point);
      val_image[i]-= 2*(valuation_point[i]-position);
      R = valuation_point - source;
      R_image = val_image - source;

      Tensor<2,3> G = fs_kernel.value_tens_image(R,R_image);

      for (unsigned int j=0; j<3; ++j)
        {
          if (std::abs(G[i][j])<tol)
            std::cout<<"OK"<<std::endl;
          else
            std::cout<<"ERROR at j = "<<j<<" , "<<G[i][j]<<std::endl;

        }
    }
}
