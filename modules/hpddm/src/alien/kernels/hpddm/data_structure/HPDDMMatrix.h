// -*- C++ -*-
#ifndef ALIEN_HPDDMIMPL_HPDDMMATRIX_H
#define ALIEN_HPDDMIMPL_HPDDMMATRIX_H
/* Author :
 */

#include <vector>

#include <alien/kernels/hpddm/HPDDMPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/

BEGIN_HPDDMINTERNAL_NAMESPACE

template <typename ValueT> class MatrixInternal;

END_HPDDMINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
template <typename ValueT> class HPDDMMatrix : public IMatrixImpl
{
 public:
  typedef HPDDMInternal::MatrixInternal<ValueT> MatrixInternal;
  typedef SimpleCSRMatrix<ValueT> CSRMatrixType;
  typedef SimpleCSRVector<ValueT> CSRVectorType;

 public:
  HPDDMMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~HPDDMMatrix() {}

 public:
  void clear() {}

 public:
  void compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& A,
      unsigned short nu, bool schwarz_coarse_correction)
  {
    m_internal.compute(parallel_mng, A, nu, schwarz_coarse_correction);
  }

  void compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& Ad,
      const CSRMatrixType& An, unsigned short nu, bool schwarz_coarse_correction)
  {
    m_internal.compute(parallel_mng, Ad, An, nu, schwarz_coarse_correction);
  }

 public:
  MatrixInternal* internal() { return &m_internal; }
  const MatrixInternal* internal() const { return &m_internal; }

 private:
  MatrixInternal m_internal;
};

/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_HPDDMIMPL_HPDDMMATRIX_H */
