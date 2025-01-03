// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#include <list>
#include <vector>
#include <sstream>

#include <alien/kernels/composyx/ComposyxPrecomp.h>

#include <alien/expression/solver/SolverStater.h>
#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>


//#include <alien/kernels/composyx/data_structure/ComposyxInternal.h>
#include <alien/kernels/composyx/data_structure/ComposyxMatrix.h>


#include <alien/kernels/composyx/ComposyxBackEnd.h>

#include <alien/core/impl/MultiMatrixImpl.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
/*---------------------------------------------------------------------------*/

namespace Alien {
/*---------------------------------------------------------------------------*/
template <typename ValueT>
ComposyxMatrix<ValueT>::ComposyxMatrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::composyx>::name())
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();

  if (row_space.size() != col_space.size())
    throw FatalErrorException("Composyx matrix must be square");

  m_internal.reset(new MatrixInternal());
}

template <typename ValueT>
ComposyxMatrix<ValueT>::MatrixInternal*
ComposyxMatrix<ValueT>::internal()
{ return m_internal.get(); }

template <typename ValueT>
ComposyxMatrix<ValueT>::MatrixInternal const*
ComposyxMatrix<ValueT>::internal() const
{ return m_internal.get(); }

template <typename ValueT>
bool ComposyxMatrix<ValueT>::compute(IMessagePassingMng* parallel_mng, const CSRMatrixType& A)
{
  m_internal->init(parallel_mng) ;
  m_internal->compute(A) ;
  return true ;
}
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/

//template class ComposyxMatrix<double>;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
