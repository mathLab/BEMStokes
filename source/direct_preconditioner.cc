#include <direct_preconditioner.h>


DirectPreconditioner::DirectPreconditioner()
{}

DirectPreconditioner::~DirectPreconditioner()
{}

void DirectPreconditioner::set_up(SolverControl &s_c,
                                  TrilinosWrappers::SolverDirect::AdditionalData ad)
{
  direct_solver = std::shared_ptr<TrilinosWrappers::SolverDirect> (new TrilinosWrappers::SolverDirect(s_c, ad));
}
void DirectPreconditioner::initialize(const TrilinosWrappers::SparseMatrix &m)
{
  direct_solver->initialize(m);
}

void DirectPreconditioner::vmult(TrilinosWrappers::MPI::Vector &dst, const TrilinosWrappers::MPI::Vector &src) const
{
  direct_solver->solve(dst, src);
}
