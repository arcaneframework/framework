// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#ifndef ALIEN_TRILINOSIMPL_TRILINOSMATRIX_H
#define ALIEN_TRILINOSIMPL_TRILINOSMATRIX_H

#include <vector>

#include <alien/kernels/trilinos/TrilinosPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/data/ISpace.h>
#include <alien/ref/data/scalar/Vector.h>
#include <alien/ref/handlers/scalar/VectorReader.h>

/*---------------------------------------------------------------------------*/

BEGIN_TRILINOSINTERNAL_NAMESPACE

template <typename ValueT, typename TagT> class MatrixInternal;

template <typename ValueT, typename TagT> class VectorInternal;

END_TRILINOSINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

namespace Alien {

template <typename ValueT, typename TagT> class TrilinosVector;

/*---------------------------------------------------------------------------*/
template <typename ValueT, typename TagT> class ALIEN_TRILINOS_EXPORT  TrilinosMatrix : public IMatrixImpl
{
 public:
  typedef TrilinosInternal::MatrixInternal<ValueT, TagT> MatrixInternal;
  typedef ValueT scalar_type;

 public:
  TrilinosMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~TrilinosMatrix();

 public:
  void clear() {}

 public:
  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
      int local_offset, int global_size, int nrows, int const* kcol, int const* cols,
      int block_size, ValueT const* values);

  bool setMatrixValues(Real const* values);

  void setMatrixCoordinate(Vector const& x, Vector const& y, Vector const& z) ;

  void mult(TrilinosVector<ValueT, TagT> const& x, TrilinosVector<ValueT, TagT>& y) const;

  void mult(ValueT const* x, ValueT* y) const;

  void dump(std::string const& filename) const;

 public:
  MatrixInternal* internal() { return m_internal.get(); }
  const MatrixInternal* internal() const { return m_internal.get(); }

 private:
  std::unique_ptr<MatrixInternal> m_internal;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/

#endif /* ALIEN_NUMERICS_LINEARALGEBRA2_TRILINOSIMPL_TRILINOSMATRIX_H */
