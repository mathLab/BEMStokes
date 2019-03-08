#include <flagellar_geometry_handler.h>

namespace BEMStokes
{
  using namespace dealii;
  using namespace deal2lkit;

  void compute_rotation_matrix_given_theta(FullMatrix<double> &R, const double theta)
  {
    R.reinit(3,3);
    R.set(0,0,1);
    R.set(0,1,0);
    R.set(0,2,0);

    R.set(1,0,0);
    R.set(1,1,cos(theta));
    R.set(1,2,-sin(theta));

    R.set(2,0,0);
    R.set(2,1,+sin(theta));
    R.set(2,2,cos(theta));
  }


  template<int dim>
  void FlagellarGeometryHandler<dim>::declare_parameters (ParameterHandler &prm)
  {
    add_parameter(prm, &Nlambda,"Number of turns for the spiral","1.5",Patterns::Double());
    add_parameter(prm, &Lx,"Length over x axis","7.17952051265",Patterns::Double());
    add_parameter(prm, &alpha,"Flagellar Amplitude","0.761770785745",Patterns::Double());
    add_parameter(prm, &k,"Flagellar wave number","1.31273083546",Patterns::Double());
    add_parameter(prm, &ke,"Reduction parameter","1.31273083546",Patterns::Double());
    add_parameter(prm, &delta_head_flagellum,"Head Flagellum Separation","0.125",Patterns::Double());
    add_parameter(prm, &a,"Flagellum cross section radius","0.1",Patterns::Double());
  }

  template<>
  void FlagellarGeometryHandler<2>::create_initial_flagellum_triangulation (Triangulation<1, 2> &)
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }
  template<>
  void FlagellarGeometryHandler<3>::create_initial_flagellum_triangulation (Triangulation<2, 3> &tria)
  {
    const unsigned int dim = 3;
    tria.clear();
    Triangulation<dim,dim> space_tria;
    Triangulation<dim-1,dim> foo_tria;

    Point<dim> p1,p2,direction;
    p1[0]=delta_head_flagellum;
    p1[1]=a/std::sqrt(2);
    p1[2]=a/std::sqrt(2);

    p2[0]=delta_head_flagellum+Lx;
    p2[1]=-a/std::sqrt(2);
    p2[2]=-a/std::sqrt(2);

    direction[0]=1.;
    CylindricalManifold<dim-1,dim> cilly(0);//(direction,p1);

    GridGenerator::hyper_rectangle(space_tria,p1,p2);

    GridGenerator::extract_boundary_mesh(space_tria,foo_tria);

    foo_tria.set_all_manifold_ids(0);
    foo_tria.set_manifold(0, cilly);
    GridTools::remove_anisotropy(foo_tria,1.,5);

    foo_tria.refine_global();



    // GridGenerator::flatten_triangulation(foo_tria, tria);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::string filename = "foo_tria.vtu";
        std::ofstream wall_ofs;
        wall_ofs.open(filename, std::ofstream::out);
        GridOut goo;
        goo.write_vtu(foo_tria,wall_ofs);

        std::ofstream foo_ofs;
        foo_ofs.open("foo.inp", std::ofstream::out);
        GridOut go;
        go.write_ucd(foo_tria,foo_ofs);



      }
    MPI_Barrier(MPI_COMM_WORLD);
    std::ifstream foo_ifs;
    foo_ifs.open("foo.inp", std::ofstream::out);
    GridIn<dim-1,dim> gi;
    gi.attach_triangulation(tria);
    gi.read_ucd(foo_ifs);

    foo_tria.reset_manifold(0);


  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::parse_parameters (ParameterHandler &prm)
  {
    ParameterAcceptor::parse_parameters(prm);
  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::set_geometry_cache (const DoFHandler<dim-1,dim> &map_dh_in, const IndexSet *map_flagellum_cpu_set_in,const Mapping<dim-1,dim> *mapping_in)
  {
    map_dh = &map_dh_in;
    map_flagellum_cpu_set = map_flagellum_cpu_set_in;
    mapping = &(*mapping_in);
  }

  template<>
  void FlagellarGeometryHandler<2>::compute_reference_euler(Vector<double> &) const
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::compute_reference_euler(Vector<double> &reference_euler) const
  {
    VectorTools::get_position_vector(*map_dh,reference_euler);
    // std::cout<<"MANNAGGIA TROIA PUTTANA VIGLIACCA"<<std::endl;
    std::cout<<Nlambda<<" "<<Lx<<" "<<alpha<<" "<<k<<" "<<ke<<" "<<delta_head_flagellum<<" "<<a<<std::endl;
    // std::vector<Point<dim> > cylindrical_support_points(map_dh->n_dofs());
    // DoFTools::map_dofs_to_support_points<dim-1, dim>(*mapping,
    //         *map_dh, cylindrical_support_points);
    reference_euler.print(std::cout);
    for (types::global_dof_index i=0; i<map_dh->n_dofs()/dim; ++i)
      {
        if (map_flagellum_cpu_set->is_element(i))
          {
            double E, Eprime, D, G, H, M, N, phi, x;
            phi = atan2(reference_euler[i+1*map_dh->n_dofs()/dim],reference_euler[i+2*map_dh->n_dofs()/dim]);
            x=reference_euler[i]-delta_head_flagellum;
            double aa=std::pow(reference_euler[i+2*map_dh->n_dofs()/dim]*reference_euler[i+2*map_dh->n_dofs()/dim]+reference_euler[i+1*map_dh->n_dofs()/dim]*reference_euler[i+1*map_dh->n_dofs()/dim],0.5);
            if (x>0.)
              {
                // if(phi<0){
                //   phi+= 2.*numbers::PI;
                // }

                //if(reference_euler[i+2*map_dh->n_dofs()/dim]<0)
                //{
                //phi += numbers::PI;
                //}
                double a = aa;
                E = 1. - exp(-(ke*x*ke*x));
                double E2 = 1. - exp(-(ke*ke*ke*x*ke*x));
                Eprime = 2. * ke * ke * x * exp(-(ke*x*ke*x));
                double theta = k*x-numbers::PI;
                D=std::pow(1+alpha*alpha*(E*k*E*k+Eprime*Eprime),0.5);
                G=std::pow((E*k*E*k+Eprime*Eprime),0.5);
                H=alpha*a*D/G*sin(phi);
                M=a/G*((E*k*sin(theta)-Eprime*cos(theta))/D*sin(phi)+(Eprime*sin(theta)+E*k*cos(theta))*cos(phi));
                N=a/G*((E*k*sin(theta)-Eprime*cos(theta))*cos(phi)-(Eprime*sin(theta)+E*k*cos(theta))/D*sin(phi));
                std::cout<<x<<" "<<reference_euler[i]<<" "<<-(ke*x*ke*x)<<" "<<D<<" "<<E<<" "<<Eprime<<" "<<H<<" "<<M<<" "<<N<<std::endl;
                reference_euler[i]                       =x+E2*H+delta_head_flagellum;
                reference_euler[i+map_dh->n_dofs()/dim]  =alpha*E*cos(theta)+M;
                reference_euler[i+map_dh->n_dofs()/dim*2]=alpha*E*sin(theta)+N;

              }

          }
      }

  }

  template<>
  void FlagellarGeometryHandler<2>::compute_reference_euler_constant_spiral(Vector<double> &) const
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::compute_reference_euler_constant_spiral(Vector<double> &reference_euler) const
  {
    VectorTools::get_position_vector(*map_dh,reference_euler);
    std::cout<<Nlambda<<" "<<Lx<<" "<<alpha<<" "<<k<<" "<<ke<<" "<<delta_head_flagellum<<" "<<a<<std::endl;
    // std::vector<Point<dim> > cylindrical_support_points(map_dh->n_dofs());
    // DoFTools::map_dofs_to_support_points<dim-1, dim>(*mapping,
    //         *map_dh, cylindrical_support_points);
    // reference_euler.print(std::cout);
    // std::cout<<"MANNAGGIA TROIA PUTTANA VIGLIACCA"<<std::endl;
    map_flagellum_cpu_set->print(std::cout);
    for (types::global_dof_index i=0; i<map_dh->n_dofs()/dim; ++i)
      {
        if (map_flagellum_cpu_set->is_element(i))
          {
            double E, Eprime, D, G, H, M, N, phi, x;
            phi = atan2(reference_euler[i+1*map_dh->n_dofs()/dim],reference_euler[i+2*map_dh->n_dofs()/dim]);
            x=reference_euler[i]-delta_head_flagellum;
            double aa=std::pow(reference_euler[i+2*map_dh->n_dofs()/dim]*reference_euler[i+2*map_dh->n_dofs()/dim]+reference_euler[i+1*map_dh->n_dofs()/dim]*reference_euler[i+1*map_dh->n_dofs()/dim],0.5);
            if (true)
              {
                // if(phi<0){
                //   phi+= 2.*numbers::PI;
                // }

                //if(reference_euler[i+2*map_dh->n_dofs()/dim]<0)
                //{
                //phi += numbers::PI;
                //}

                double E3=1.; //= 1.  - exp(-(k*k*k*k*(x+0.1)*(x+0.1)*(x+0.1)*(x+0.1)));
                double E4=1.; //= 1.  - exp(-(k*k*k*k*(x-Lx-0.1)*(x-Lx-0.1)*(x-Lx-0.1)*(x-Lx-0.1)));
                if (x<0.2)
                  E3 = (x+0.1) / (0.3);
                else if (Lx-x<0.2)
                  E4 = (Lx+0.1-x) / (0.3);

                double a = E3 * E4 * aa;

                E = 1;
                // double E2 = 1. - exp(-(ke*ke*ke*x*ke*x));
                Eprime = 0;
                double theta = k*x-numbers::PI;
                D=std::pow(1+alpha*alpha*k*k,0.5);//std::pow(1+alpha*alpha*(E*k*E*k+Eprime*Eprime),0.5); //1+alpha^2*k^2
                G=k;//std::pow((E*k*E*k+Eprime*Eprime),0.5);//k
                H=alpha*a*D/G*sin(phi);
                M=a/k*((k*sin(theta))/D* sin(phi)+k*cos(theta)*cos(phi));//a/G*((E*k*sin(theta)-Eprime*cos(theta))/D*sin(phi)+(Eprime*sin(theta)+E*k*cos(theta))*cos(phi));//a/k*((k*sin(theta))/D* sin(phi)+k*cos(theta)*cos(phi))
                N=a/k*(k*sin(theta)/D* cos(phi)-k*cos(theta)*sin(phi)/D);//a/G*((E*k*sin(theta)-Eprime*cos(theta))*cos(phi)-(Eprime*sin(theta)+E*k*cos(theta))/D*sin(phi));//a/k*(k*sin(theta)/D* cos(phi)-k*cos(theta)sin(phi)/D)
                std::cout<<x<<" "<<reference_euler[i]<<" "<<-(ke*x*ke*x)<<" "<<D<<" "<<E<<" "<<Eprime<<" "<<H<<" "<<M<<" "<<N<<std::endl;
                reference_euler[i]                       =x+H+delta_head_flagellum;
                reference_euler[i+map_dh->n_dofs()/dim]  =alpha*cos(theta)+M;
                reference_euler[i+map_dh->n_dofs()/dim*2]=alpha*sin(theta)+N;

              }

          }
      }

  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::save_geometry(const std::string path_save, const std::string basename, const std::string extension, const unsigned int n_frames) const
  {
    Assert(false, ExcNotImplemented());

    for (unsigned int i=0; i<n_frames; ++i)
      {
        double theta = numbers::PI * 2. * i / n_frames;
        std::string filename = path_save+basename+Utilities::int_to_string(i)+extension;
        std::ofstream ofs_flag(filename.c_str());
        std::cout<<"Saving the geometry at frame "<<i<<" , theta "<<theta<<std::endl;
        // TO BE COMPLETE ROTATING THE GEOMETRY WITH THETA AND SAVING!!!!


      }
  }
  template<>
  void FlagellarGeometryHandler<2>::compute_euler_at_theta(Vector<double> &, const Vector<double> &, const double ) const
  {
    AssertThrow(false, ExcImpossibleInDim(2));
  }

  template<int dim>
  void FlagellarGeometryHandler<dim>::compute_euler_at_theta(Vector<double> &euler, const Vector<double> &reference_euler, const double theta) const
  {
    FullMatrix<double> R;
    compute_rotation_matrix_given_theta(R,theta);
    R.print_formatted(std::cout);
    reference_euler.print(std::cout);
    for (types::global_dof_index i=0; i<map_dh->n_dofs()/dim; ++i)
      {
        if (map_flagellum_cpu_set->is_element(i))
          {
            Vector<double> position_old(dim), position_new(dim);
            for (unsigned int d=0; d<dim; ++d)
              {
                position_old[d] = reference_euler[i+d*map_dh->n_dofs()/dim];
              }
            R.vmult(position_new, position_old);
            for (unsigned int d=0; d<dim; ++d)
              {
                euler[i+d*map_dh->n_dofs()/dim] = position_new[d];
              }
          }
        else
          {
            for (unsigned int d=0; d<dim; ++d)
              {
                euler[i+d*map_dh->n_dofs()/dim] = reference_euler[i+d*map_dh->n_dofs()/dim];
              }

          }
      }
    euler.print(std::cout);
  }

}

template class BEMStokes::FlagellarGeometryHandler<2>;
template class BEMStokes::FlagellarGeometryHandler<3>;
