// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------


#pragma once

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/sycl/data/HCSRVector.h>
#include <alien/kernels/sycl/data/HCSRVectorInternal.h>

#include <alien/kernels/sycl/data/SYCLParallelEngine.h>
#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>

#include <alien/handlers/scalar/sycl/VectorAccessorT.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace SYCL
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/
  template <typename ValueT>
  class VectorAccessorT<ValueT>::Impl
  {
  public :
    typedef typename HCSRVector<ValueT>::InternalType VectorInternalType ;
    typedef typename VectorInternalType::ValueBufferType ValueBufferType ;

    Impl(ValueBufferType& buffer)
    : m_buffer(buffer)
    {}

    //std::span<ValueT> m_values ;
    ValueBufferType& m_buffer ;

    auto accessor(SYCLControlGroupHandler& cgh) {
      return m_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal) ;
    }
  };

  template <typename ValueT>
  VectorAccessorT<ValueT>::VectorAccessorT(IVector& vector, bool update)
  : m_time_stamp(nullptr)
  , m_local_offset(0)
  , m_finalized(false)
  {
    auto& v = vector.impl()->get<BackEnd::tag::hcsr>(update);
    m_local_offset = v.distribution().offset();
    m_impl.reset(new Impl(v.internal()->values())) ;
    m_time_stamp = &v;
  }

  template <typename ValueT>
  typename VectorAccessorT<ValueT>::Impl* VectorAccessorT<ValueT>::impl() {
    assert(m_impl.get()) ;
    return m_impl.get() ;
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void VectorAccessorT<ValueT>::end()
  {
    if (m_finalized)
      return;
    m_finalized = true;
  }

  template <typename ValueT>
  class VectorAccessorT<ValueT>::View
  {
    sycl::handler* m_h = nullptr ;
    sycl::buffer<ValueT,1>* m_b = nullptr ;
    using AccessorType = decltype(m_b->template get_access<sycl::access::mode::read_write>(*m_h));

  public :
    explicit View(AccessorType accessor)
    : m_accessor(accessor)
    {}

    ValueT& operator[](std::size_t index) const {
      return m_accessor[index] ;
    }

  private :
    AccessorType m_accessor ;

  } ;

  template <typename ValueT>
  class VectorAccessorT<ValueT>::ConstView
  {
    sycl::handler* m_h = nullptr ;
    sycl::buffer<ValueT,1>* m_b = nullptr ;
    using AccessorType = decltype(m_b->template get_access<sycl::access::mode::read>(*m_h));

  public :
    explicit ConstView(AccessorType accessor)
    : m_accessor(accessor)
    {}

    ValueT operator[](std::size_t index) const {
      return m_accessor[index] ;
    }

  private :
    AccessorType m_accessor ;
  } ;


  template <typename ValueT>
  class VectorAccessorT<ValueT>::HostView
  {
  public :
    typedef typename HCSRVector<ValueT>::InternalType VectorInternalType ;
    typedef typename VectorInternalType::ValueBufferType ValueBufferType ;

    sycl::buffer<ValueT,1>* m_b = nullptr ;
    using AccessorType = decltype(m_b->get_host_access());

    explicit HostView(AccessorType values)
    : m_values(values)
    {

    }

    ValueType operator[](std::size_t index) const {
      return m_values[index] ;
    }

  private:
    AccessorType m_values ;
  };

  /*---------------------------------------------------------------------------*/
  template <typename ValueT>
  typename VectorAccessorT<ValueT>::View VectorAccessorT<ValueT>::view(SYCLControlGroupHandler& cgh)
  {
    return View(m_impl->m_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal)) ;
  }

  template <typename ValueT>
  typename VectorAccessorT<ValueT>::ConstView VectorAccessorT<ValueT>::constView(SYCLControlGroupHandler& cgh) const
  {
    return VectorAccessorT<ValueT>::ConstView(m_impl->m_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal)) ;
  }

  template <typename ValueT>
  typename VectorAccessorT<ValueT>::HostView VectorAccessorT<ValueT>::hostView() const
  {
    return HostView(m_impl->m_buffer.get_host_access()) ;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace SYCL

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
