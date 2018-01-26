
DEAL_II_NAMESPACE_OPEN

template <typename number> class VEC;

template<class VEC, class MATRIX>
class CustomOperator
{
public:
  CustomOperator(const VEC &pv, const VEC &dpv, const double norm, const MATRIX &m):
    proj_vec(pv),
    dual_proj_vec(dpv),
    proj_vec_norm(norm),
    matrix(m)
  {}

  void projector(const VEC &src, VEC &dst) const;

  void vmult(VEC dst, const VEC &src) const;

private:
  const VEC &proj_vec;
  const VEC &dual_proj_vec;
  const double proj_vec_norm;
  const MATRIX &matrix;



};

template<class VEC, class MATRIX>
void CustomOperator<VEC, MATRIX>::projector(const VEC &src,  VEC &dst) const
{
  dst.sadd(0.,1.,src);
  dst.sadd(1., -(dual_proj_vec*src)/proj_vec_norm, proj_vec);
  return;

}


template<class VEC, class MATRIX>
void CustomOperator<VEC, MATRIX>::vmult(VEC dst, const VEC &src) const
{
  VEC foo(src.locally_owned_elements(), src.get_mpi_communicator() );
  projector(src, foo);
  matrix.vmult(dst, foo);
  foo = 0.;
  projector(dst, foo);
  dst = foo;

}

DEAL_II_NAMESPACE_CLOSE
