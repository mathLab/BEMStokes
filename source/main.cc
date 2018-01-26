#include "bem_stokes.h"

// This is the main function of this program. It is exactly like all previous
// tutorial programs:
int main (int argc, char **argv)
{
  try
    {
      using namespace dealii;
      using namespace BEMStokes;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, numbers::invalid_unsigned_int);

      deallog.depth_console ( -1 );
      unsigned int start_frame = 0;
      unsigned int end_frame = 139;
      unsigned int compose = 0;
      switch (argc)
        {
        case 2:
          start_frame = Utilities::string_to_int(argv[1]);
          break;
        case 3:
          start_frame = Utilities::string_to_int(argv[1]);
          end_frame = Utilities::string_to_int(argv[2]);
          break;
        case 4:
          start_frame = Utilities::string_to_int(argv[1]);
          end_frame = Utilities::string_to_int(argv[2]);
          compose = Utilities::string_to_int(argv[3]);

        }

      std::string pname = "parameters_" + std::to_string(DIMENSION) + ".prm";
      std::string pname2 = "used_parameters_" + std::to_string(DIMENSION) + ".prm";
      BEMProblem<DIMENSION> stokes_problem(MPI_COMM_WORLD);
      ParameterAcceptor::initialize(pname, pname2);
      deallog.depth_console (3);

      if (compose == 1)
        stokes_problem.composer(start_frame, end_frame);
      else
        stokes_problem.run(start_frame, end_frame);
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
