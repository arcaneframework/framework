// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------

#include <alien/kernels/sycl/SYCLPrecomp.h>

#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include <span>

#include "HCSRVector.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace Alien
{

  namespace HCSRInternal
  {

  /*---------------------------------------------------------------------------*/

  #ifndef USE_SYCL2020
    using namespace cl ;
  #endif

  template <typename ValueT = Real>
  class VectorInternal
  {
   public:
    // clang-format off
    typedef ValueT                           ValueType;
    typedef VectorInternal<ValueType>        ThisType;
    typedef sycl::buffer<ValueType, 1>       ValueBufferType;
    // clang-format on

   public:
    VectorInternal(std::size_t size)
    : m_values(sycl::range<1>(size))
    {}

    virtual ~VectorInternal() {}

    /*
    std::vector<ValueType>& hostValues() {
      return m_h_values ;
    }

    std::vector<ValueType> const& hostValues() const {
      return m_h_values ;
    }*/

    ValueBufferType& values()
    {
      return m_values;
    }


    ValueBufferType& values() const
    {
      return m_values;
    }

    void copyValuesToHost(std::size_t size, ValueT* ptr)
    {
      auto h_values = m_values.template get_host_access();
      for (std::size_t i = 0; i < size; ++i)
        ptr[i] = h_values[i];
    }

    //mutable std::vector<ValueType> m_h_values ;
    mutable ValueBufferType        m_values;
  };

  /*---------------------------------------------------------------------------*/

  } // namespace Alien::SYCLInternal
  template <typename ValueT>
  HCSRVector<ValueT>::HCSRVector()
  : IVectorImpl(nullptr, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {}

  //! Constructeur avec association ? un MultiImpl
  template <typename ValueT>
  HCSRVector<ValueT>::HCSRVector(const MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {}

/*---------------------------------------------------------------------------*/
template <typename ValueT>
void HCSRVector<ValueT>::allocate()
{
  m_internal.reset(new InternalType(m_local_size)) ;
}

template <typename ValueT>
void HCSRVector<ValueT>::resize(Integer alloc_size)
{
  m_local_size = alloc_size ;
  m_internal.reset(new InternalType(m_local_size)) ;
}

/*
template <typename ValueT>
ValueT* HCSRVector<ValueT>::getDataPtr()
{
  if(m_internal.get())
    return m_internal->hostValues().data() ;
  else
    return nullptr ;
}

template <typename ValueT>
ValueT* HCSRVector<ValueT>::data()
{
  if(m_internal.get())
    return m_internal->hostValues().data() ;
  else
    return nullptr ;
}

template <typename ValueT>
ValueT const* HCSRVector<ValueT>::getDataPtr() const
{
  if(m_internal.get())
    return m_internal->hostValues().data() ;
  else
    return nullptr ;
}


template <typename ValueT>
ValueT const* HCSRVector<ValueT>::data() const
{
  if(m_internal.get())
    return m_internal->hostValues().data() ;
  else
    return nullptr ;
}


template <typename ValueT>
ValueT const* HCSRVector<ValueT>::getAddressData() const
{
  if(m_internal.get())
    return m_internal->hostValues().data() ;
  else
    return nullptr ;
}
*/

template <typename ValueT>
void HCSRVector<ValueT>::init(const VectorDistribution& dist,
            const bool need_allocate)
{

  alien_debug([&] { cout() << "Initializing HCSRVector " << this; });
  if (this->m_multi_impl) {
    m_local_size = this->scalarizedLocalSize();
  }
  else {
    // Not associated vector
    m_own_distribution = dist;
    m_local_size = m_own_distribution.localSize();
  }
  if (need_allocate) {
      allocate() ;
  }
}
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
