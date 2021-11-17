// -*- C++ -*-
#ifndef ALIEN_HTSIMPL_MTLMATRIX_H
#define ALIEN_HTSIMPL_MTLMATRIX_H
/* Author :
 */

#include <vector>

#include <alien/kernels/hts/HTSPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/

BEGIN_HTSINTERNAL_NAMESPACE

template <typename ValueTi, bool is_mpi> class MatrixInternal;

END_HTSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
template <typename ValueT> class HTSMatrix : public IMatrixImpl
{
 public:
  typedef HTSInternal::MatrixInternal<ValueT, true> MatrixInternal;

 public:
  HTSMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~HTSMatrix();

 public:
  void clear() {}

 public:
  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng, int nrows,
      int const* kcol, int const* cols, int block_size);

  bool setMatrixValues(Arccore::Real const* values);

  bool computeDDMatrix();

  void mult(ValueT const* x, ValueT* y) const;

 public:
  MatrixInternal* internal() { return m_internal.get(); }
  const MatrixInternal* internal() const { return m_internal.get(); }

  //   void update(const Alien::SimpleCSRMatrix<ValueT> & v);
  //   void update(const HTSMatrix & v);

 private:
  std::unique_ptr<MatrixInternal> m_internal;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_NUMERICS_LINEARALGEBRA2_HTSIMPL_MTLMATRIX_H */
