// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <vector>

#include <alien/kernels/composyx/ComposyxPrecomp.h>
#include <alien/core/impl/IMatrixImpl.h>
#include <alien/data/ISpace.h>

/*---------------------------------------------------------------------------*/

BEGIN_COMPOSYXINTERNAL_NAMESPACE

template <typename ValueT> class MatrixInternal;

END_COMPOSYXINTERNAL_NAMESPACE

/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
template <typename ValueT>
class ComposyxMatrix : public IMatrixImpl
{
 public:
  typedef ComposyxInternal::MatrixInternal<ValueT> MatrixInternal;
  typedef SimpleCSRMatrix<ValueT> CSRMatrixType;
  typedef SimpleCSRVector<ValueT> CSRVectorType;

 public:
  ComposyxMatrix(const MultiMatrixImpl* multi_impl);
  virtual ~ComposyxMatrix() {}

 public:
  void clear() {}

  bool isParallel() const ;

 public:
  bool compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& A) ;

 public:
  MatrixInternal* internal() ;
  const MatrixInternal* internal() const ;

 private:
  std::unique_ptr<MatrixInternal> m_internal ;
};

/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/

