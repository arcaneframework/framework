/*
 * Copyright 2020 IFPEN-CEA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiVectorImpl.h>
#include <alien/kernels/sycl/data/HCSRVector.h>
#include <alien/kernels/sycl/data/HCSRVectorInternal.h>

#include <alien/kernels/sycl/data/SYCLParallelEngine.h>
#include <alien/kernels/sycl/data/SYCLParallelEngineImplT.h>

#include <alien/handlers/scalar/sycl/VectorAccessor.h>

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
    typedef HCSRVector<ValueT>::InternalType VectorInternalType ;
    typedef VectorInternalType::ValueBufferType ValueBufferType ;

    //std::span<ValueT> m_values ;
    ValueBufferType& m_buffer ;
  };

  template <typename ValueT>
  VectorAccessorT<ValueT>::VectorAccessorT(IVector& vector, bool update)
  : m_time_stamp(nullptr)
  , m_local_offset(0)
  , m_finalized(false)
  {
    auto& v = vector.impl()->get<BackEnd::tag::hcsr>(update);
    m_local_offset = v.distribution().offset();
    m_impl = new Impl(v.internal()->values()) ;
    m_time_stamp = &v;
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
    typedef HCSRVector<ValueT>::InternalType VectorInternalType ;
    typedef VectorInternalType::ValueBufferType ValueBufferType ;

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
  VectorAccessorT<ValueT>::View VectorAccessorT<ValueT>::view(SYCLControlGroupHandler& cgh)
  {
    return View(m_impl->m_buffer.template get_access<sycl::access::mode::read_write>(cgh.m_internal)) ;
  }

  template <typename ValueT>
  VectorAccessorT<ValueT>::ConstView VectorAccessorT<ValueT>::constView(SYCLControlGroupHandler& cgh) const
  {
    return VectorAccessorT<ValueT>::ConstView(m_impl->m_buffer.template get_access<sycl::access::mode::read>(cgh.m_internal)) ;
  }

  template <typename ValueT>
  VectorAccessorT<ValueT>::HostView VectorAccessorT<ValueT>::hostView() const
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
