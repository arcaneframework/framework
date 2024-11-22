
#pragma once

#include <alien/kernels/simple_csr/SimpleCSRVector.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/kernels/composyx/ComposyxPrecomp.h>
//#include <alien/kernels/composyx/data_structure/ComposyxInternal.h>
#include <alien/kernels/composyx/ComposyxBackEnd.h>
#include <alien/core/block/Block.h>
#include "ComposyxVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template <typename ValueT>
ComposyxVector<ValueT>::ComposyxVector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::composyx>::name())
, m_local_offset(0)
{
}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
ComposyxVector<ValueT>::~ComposyxVector()
{
}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
void
ComposyxVector<ValueT>::init(const VectorDistribution& dist, const bool need_allocate)
{
  if (need_allocate)
    allocate();
}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
void
ComposyxVector<ValueT>::allocate()
{
  const VectorDistribution& dist = this->distribution();
  m_local_offset = dist.offset();

  m_internal.reset(new VectorInternal());
}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
bool
ComposyxVector<ValueT>::compute(IMessagePassingMng* parallel_mng, const CSRVectorType& vector)
{
  m_internal->init(parallel_mng) ;
  return m_internal->compute(vector) ;
}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
void
ComposyxVector<ValueT>::getValues(const int nrows, ValueT* values) const
{
  m_internal->getValues((std::size_t)nrows,values) ;
}


/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
template class ComposyxVector<double>;

}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
