// ---------------------------------------------------------------------
// $Id: integrate_one_over_r.cc 30338 2013-08-18 22:02:27Z heltai $
//
// Copyright (C) 2005 - 2015 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE at
// the top level of the deal.II distribution.
//
// ---------------------------------------------------------------------

// integrates the function *f(x,y)/R, where f(x,y) is a power of x and
// y on the set [0,1]x[0,1]. dim = 2 only.

#include <fstream>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

// all include files needed for the program
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <string>

#include <math.h>

using namespace std;
using namespace dealii;




// We test the integration of singular kernels with a singularity of kind 1/R
// We multiply this function with a polynomial up to degree 6.

double
exact_integral_one_over_r (
  const unsigned int i, const unsigned int j,
  const unsigned int vertex_index);

// ofstream logfile("output");

int
main ()
{
  // std::cout.attach(logfile);
  // std::cout << std::fixed;

  std::cout << std::endl
            << "Calculation of the integral of f(x,y)*1/R on [0,1]x[0,1]" << std::endl
            << "for f(x,y) = x^i y^j, with i,j ranging from 0 to 2, and R being" << std::endl
            << "the distance from (x,y) to nine support points of a Q2 element." << std::endl
            << std::endl;


  std::vector<Point<2> > vertices = FE_Q<2>(2).get_unit_support_points();
  for (unsigned int i=0; i<vertices.size(); ++i)
    {
      std::cout<<i<<" "<<vertices[i]<<std::endl;
    }

  for (unsigned int m=3; m<20; ++m)
    {
      std::cout << " =========Quadrature Order: " << m
                << " =============================== " << std::endl;
      std::cout << " ============================================================ " << std::endl;
      for (unsigned int index=0; index<vertices.size(); ++index)//vertices.size()
        {
          std::cout << " ===============Vertex: " << vertices[index]
                    << " ============================= " << std::endl;

          QTelles<2> quad(m, vertices[index]);
          QGaussOneOverR<2> quad2(m, vertices[index], true);
          QIterated<2> quad3(QGauss<1> (m), 2);
          QSplit<2> quad4(QDuffy (m,1.),vertices[index]);
          // for(unsigned int i = 0; i < quad4.size(); ++i)
          //   std::cout<<quad4.point(i)" "<<std::endl;


          for (unsigned int i = 0; i < 3; ++i)
            {
              for (unsigned int j = 0; j < 3; ++j)
                {
                  // std::cout<<"retrieving exact value"<<std::endl;
                  double exact_integral  = exact_integral_one_over_r(index, i,j);
                  double approx_integral = 0;
                  double approx_integral_2 = 0;
                  double approx_integral_3 = 0;
                  double approx_integral_4 = 0;
                  // std::cout<<"computing quadrature value"<<std::endl;

                  for (unsigned int q=0; q< quad.size(); ++q)
                    {
                      double x = quad.point(q)[0];
                      double y = quad.point(q)[1];
                      double R = sqrt((x-vertices[index][0] )*(x-vertices[index][0] )+(y-vertices[index][1])*(y-vertices[index][1]));
                      approx_integral += ( pow(( x-vertices[index][0] ), (double)i) *
                                           pow(( y-vertices[index][1] ), (double)j) / R *
                                           quad.weight(q) );
                    }

                  for (unsigned int q=0; q< quad2.size(); ++q)
                    {
                      double x = quad2.point(q)[0];
                      double y = quad2.point(q)[1];
                      double R = sqrt((x-vertices[index][0] )*(x-vertices[index][0] )+(y-vertices[index][1])*(y-vertices[index][1]));
                      approx_integral_2 += ( pow(( x-vertices[index][0] ), (double)i) *
                                             pow(( y-vertices[index][1] ), (double)j)  *
                                             quad2.weight(q) );
                    }

                  for (unsigned int q=0; q< quad3.size(); ++q)
                    {
                      double x = quad3.point(q)[0];
                      double y = quad3.point(q)[1];
                      double R = sqrt((x-vertices[index][0] )*(x-vertices[index][0] )+(y-vertices[index][1])*(y-vertices[index][1]));
                      approx_integral_3 += ( pow(( x-vertices[index][0] ), (double)i) *
                                             pow(( y-vertices[index][1] ), (double)j) / R *
                                             quad3.weight(q) );
                    }
                  for (unsigned int q=0; q< quad4.size(); ++q)
                    {
                      double x = quad4.point(q)[0];
                      double y = quad4.point(q)[1];
                      double R = sqrt((x-vertices[index][0] )*(x-vertices[index][0] )+(y-vertices[index][1])*(y-vertices[index][1]));
                      approx_integral_4 += ( pow(( x-vertices[index][0] ), (double)i) *
                                             pow(( y-vertices[index][1] ), (double)j) / R *
                                             quad4.weight(q) );
                    }

                  std::cout << "f(x,y) = x^" << i
                            << " y^" << j
                            << ", Errors = Telles "
                            << std::abs(exact_integral - approx_integral)
                            << ", LWGaussOneR "
                            << std::abs(exact_integral - approx_integral_2)
                            << ", QIteraded(QGauss) "
                            << std::abs(exact_integral - approx_integral_3)
                            << ", QDuffy "
                            << std::abs(exact_integral - approx_integral_4)
                            << ", exact value "
                            << exact_integral
                            << std::endl;
                }
            }
        }
    }
}

double exact_integral_one_over_r(const unsigned int vertex_index,
                                 const unsigned int i,
                                 const unsigned int j)
{
  Assert(vertex_index < 9, ExcInternalError());
  Assert(i<3, ExcNotImplemented());
  Assert(j<3, ExcNotImplemented());

// The integrals are computed using the following maple snippet of
// code:
//
//      singint := proc(index, N, M)
//         if index = 0 then
//            return int(int(x^N *y^M/sqrt(x^2+y^2), x=0.0..1.0), y=0.0..1.0);
//         elif index = 1 then
//            return int(int(x^N *y^M/sqrt((x-1)^2+y^2), x=0.0..1.0), y=0.0..1.0);
//         elif index = 2 then
//            return int(int(x^N *y^M/sqrt(x^2+(y-1)^2), x=0.0..1.0), y=0.0..1.0);
//         elif index = 3 then
//            return int(int((1-x)^N *(1-y)^M/sqrt(x^2+y^2), x=0.0..1.0), y=0.0..1.0);
//         end if;
//      end proc;
//      Digits := 20;
//      for i from 3 to 3 do
//         for n from 0 to 5 do
//          for m from 0 to 5 do
//               v[i+1][n+1][m+1] = sing_int(i, n, m);
//            end do;
//         end do;
//      end do;
//      C(v)

  static double v[3][3][9] =
  {
    {
      { 0}
    }
  };
  if (v[0][0][0] == 0)
    {
      v[0][0][0] = 1.7627459673175378541;
      v[0][0][1] = 1.7627459673175374100;
      v[0][0][2] = 1.7627459673175374100;
      v[0][0][3] = 1.7627459673175365218;
      v[0][0][4] = 2.4060543385174839592;
      v[0][0][5] = 2.4060543385174835151;
      v[0][0][6] = 2.4060543385174835151;
      v[0][0][7] = 2.4060543385174835151;
      v[0][0][8] = 3.5254752792359669300;
      v[0][1][0] = 0.64779381295037530286;
      v[0][1][1] = 0.64779381295037530286;
      v[0][1][2] = -0.64779381295037541388;
      v[0][1][3] = -0.64779381295037530286;
      v[0][1][4] = 0;
      v[0][1][5] = 0;
      v[0][1][6] = 0.79022927532552311725;
      v[0][1][7] = -0.79022927532552311725;
      v[0][1][8] = 0;
      v[0][2][0] = 0.38259779614627675848;
      v[0][2][1] = 0.38259779614627675848;
      v[0][2][2] = 0.38259779614627681399;
      v[0][2][3] = 0.38259779614627681399;
      v[0][2][4] = 0.14623788528177569801;
      v[0][2][5] = 0.14623788528177567025;
      v[0][2][6] = 0.44699527788817983165;
      v[0][2][7] = 0.44699527788817983165;
      v[0][2][8] = 0.19129866779540399824;
      v[1][0][0] = 0.64779381295037530286;
      v[1][0][1] = -0.64779381295037530286;
      v[1][0][2] = 0.64779381295037530286;
      v[1][0][3] = -0.64779381295037530286;
      v[1][0][4] = 0.79022927532552311725;
      v[1][0][5] = -0.79022927532552311725;
      v[1][0][6] = 0;
      v[1][0][7] = 0;
      v[1][0][8] =0;
      v[1][1][0] = 0.27614241419806495603;
      v[1][1][1] = -0.27614241419806490052;
      v[1][1][2] = -0.27614241419806490052;
      v[1][1][3] = 0.27614241419806490052;
      v[1][1][4] = 0;
      v[1][1][5] = 0;
      v[1][1][6] =  0;
      v[1][1][7] =   0;
      v[1][1][8] = 0;
      v[1][2][0] = 0.17015839103434349244;
      v[1][2][1] = -0.17015839103434346469;
      v[1][2][2] = 0.17015839103434346469;
      v[1][2][3] = -0.17015839103434349244;
      v[1][2][4] = 0.058078420977436817640;
      v[1][2][5] = -0.058078420977436810702;
      v[1][2][6] = 0;
      v[1][2][7] = 0;
      v[1][2][8] = 0;
      v[2][0][0] = 0.38259779614627675848;
      v[2][0][1] = 0.38259779614627686950;
      v[2][0][2] = 0.38259779614627675848;
      v[2][0][3] = 0.38259779614627686950;
      v[2][0][4] = 0.44699527788817983165;
      v[2][0][5] = 0.44699527788817983165;
      v[2][0][6] = 0.14623788528177567025;
      v[2][0][7] = 0.14623788528177567025;
      v[2][0][8] = 0.19129866779540399824;
      v[2][1][0] = 0.17015839103434349244;
      v[2][1][1] = 0.17015839103434349244;
      v[2][1][2] = -0.17015839103434346469;
      v[2][1][3] = -0.17015839103434349244;
      v[2][1][4] = 0;
      v[2][1][5] =  0;
      v[2][1][6] = 0.058078420977436810702;
      v[2][1][7] = -0.058078420977436817640;
      v[2][1][8] =   0;
      v[2][2][0] = 0.10656799659662753721;
      v[2][2][1] = 0.10656799659662753721;
      v[2][2][2] = 0.10656799659662753721;
      v[2][2][3] = 0.10656799659662755109;
      v[2][2][4] = 0.034489161848319446757;
      v[2][2][5] = 0.034489161848319439818;
      v[2][2][6] = 0.034489161848319439818;
      v[2][2][7] = 0.034489161848319446757;
      v[2][2][8] = 0.013321000164368355240;
    }
  return v[i][j][vertex_index];
}
