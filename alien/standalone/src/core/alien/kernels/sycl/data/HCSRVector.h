// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

#include <alien/core/block/VBlockOffsets.h>
#include <alien/core/impl/IVectorImpl.h>
#include <alien/data/ISpace.h>
#include <alien/kernels/sycl/SYCLBackEnd.h>
#include <alien/kernels/sycl/SYCLPrecomp.h>
#include <iostream>

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
namespace HCSRInternal
{

  template <typename ValueT>
  class VectorInternal;
}

template <typename ValueT>
class HCSRVector : public IVectorImpl
{
 public:
  typedef ValueT  ValueType;
  typedef Integer IndexType;
  typedef HCSRInternal::VectorInternal<ValueType> InternalType ;

  //! Constructeur sans association ? un MultiImpl
  HCSRVector() ;

  //! Constructeur avec association ? un MultiImpl
  HCSRVector(const MultiVectorImpl* multi_impl) ;

  void allocate() ;

  void resize(Integer alloc_size) ;

  void clear() override
  {
     ;
  }

  void init(const VectorDistribution& dist,
            const bool need_allocate) override ;

  const VectorDistribution& distribution() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::distribution();
    else
      return m_own_distribution;
  }

  Arccore::Integer scalarizedLocalSize() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedLocalSize();
    else
      return m_own_distribution.localSize();
  }

  Arccore::Integer scalarizedGlobalSize() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedGlobalSize();
    else
      return m_own_distribution.globalSize();
  }

  Arccore::Integer scalarizedOffset() const override
  {
    if (this->m_multi_impl)
      return IVectorImpl::scalarizedOffset();
    else
      return m_own_distribution.offset();
  }

  Arccore::Integer allocSize() const {
    return m_local_size ;
  }

  Arccore::Integer getAllocSize() const {
    return m_local_size ;
  }

  ValueType const* dataPtr() const ;
  void initDevicePointers(int** rows, ValueType** values) const ;
  void freeDevicePointers(int* rows, ValueType* values) const ;

  void copyValuesTo(ValueType* values) const;

  InternalType* internal() {
    return m_internal.get() ;
  }

  InternalType const* internal() const {
    return m_internal.get() ;
  }

 private:
  Alien::BackEnd::Memory::eType m_mem_kind = Alien::BackEnd::Memory::Device;
  std::unique_ptr<InternalType> m_internal ;
  Integer m_local_size = 0;
  VectorDistribution m_own_distribution;
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
