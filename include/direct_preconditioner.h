#include <deal.II/lac/trilinos_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_solver.h>


using namespace dealii;

/// This function is a preconditioner exploiting a direct resolution at a given frame. We assume that such direct resolution acts as a proper preconditioner
/// for the successive frames. If the number of iteration exceeds 100 we provide a new direct resolution.
class DirectPreconditioner : public TrilinosWrappers::PreconditionBase
{
public:
  DirectPreconditioner();

  ~DirectPreconditioner();


  void set_up(SolverControl &s_c, TrilinosWrappers::SolverDirect::AdditionalData ad=TrilinosWrappers::SolverDirect::AdditionalData () );

  /// We initialize the dicrect solver considering a given TrilinosWrappers::SparseMatrix. The Initialize provides the direct solving.
  void initialize(const TrilinosWrappers::SparseMatrix &m);

  using TrilinosWrappers::PreconditionBase::vmult;

  /// We vmult function provides the matrix vector multiplication using the direct risolution of the provided TrilinosWrappers::SparseMatrix m. If == problem system matrix
  /// we can achieve the solution in a single iterations. Otherwise we should retrieve the solution with N<100 (thus with a great computational improvement).
  virtual void vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const;

private:

  /// The dicrect preconditioner is based on the usage of a shared_ptr to a TrilinosWrappers::SolverDirect.
  std::shared_ptr<TrilinosWrappers::SolverDirect> direct_solver;


};
