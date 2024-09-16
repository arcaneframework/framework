// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
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

#include <cassert>

#include "SYCLVector.h"
#include "SYCLVectorInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

using namespace Arccore;

namespace Alien
{

  //! Constructeur sans association un MultiImpl
  template <typename ValueT>
  SYCLVector<ValueT>::SYCLVector()
  : IVectorImpl(nullptr, AlgebraTraits<BackEnd::tag::sycl>::name())
  {}

  //! Constructeur avec association ? un MultiImpl
  template <typename ValueT>
  SYCLVector<ValueT>::SYCLVector(const MultiVectorImpl* multi_impl)
  : IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::sycl>::name())
  {}

  template <typename ValueT>
  void SYCLVector<ValueT>::allocate()
  {
    //delete m_internal;
    m_h_values.resize(m_local_size);
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::resize(Integer alloc_size) const
  {
    //delete m_internal;
    m_h_values.resize(alloc_size);
    m_internal.reset(new VectorInternal(m_h_values.data(), alloc_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::clear()
  {
    //delete m_internal;
    //m_internal = nullptr;
    m_internal.reset() ;
    std::vector<ValueType>().swap(m_h_values);
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValuesFromHost()
  {
    //delete m_internal;
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::setValues(std::size_t size, ValueType const* ptr)
  {
    //delete m_internal;
    m_h_values.resize(m_local_size);
    std::copy(ptr, ptr + size, m_h_values.begin());
    m_internal.reset(new VectorInternal(m_h_values.data(), m_local_size));
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::copyValuesTo(std::size_t size, ValueType* ptr) const
  {
    if (m_internal.get())
      m_internal->copyValuesToHost(size, ptr);
  }


  template <typename ValueT>
  void SYCLVector<ValueT>::initDevicePointers(int** rows, ValueType** values) const
  {
    if(m_internal.get()==nullptr)
      return ;

    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    auto max_num_treads = env->maxNumThreads() ;

    auto values_ptr = malloc_device<ValueT>(m_local_size, queue);
    auto rows_ptr   = malloc_device<IndexType>(m_local_size, queue);

    queue.submit( [&](sycl::handler& cgh)
                  {
                    auto access_x = m_internal->m_values.template get_access<sycl::access::mode::read>(cgh);
                    std::size_t y_length = m_local_size ;
                    cgh.parallel_for<class init_vector_ptr>(sycl::range<1>{max_num_treads}, [=] (sycl::item<1> itemId)
                                                      {
                                                          auto id = itemId.get_id(0);
                                                          for (auto i = id; i < y_length; i += itemId.get_range()[0])
                                                          {
                                                            values_ptr[i] = access_x[i];
                                                            rows_ptr[i] = i ;
                                                          }
                                                      });
                  });

    queue.wait() ;
    sycl::host_accessor values_h(m_internal->m_values) ;
    for(int i=0;i<m_local_size;++i)
    {
       std::cout<<"SYCL VECTOR VALUES["<<i<<"]"<<values_h[i]<<std::endl ;
    }
    *values = values_ptr;
    *rows   = rows_ptr;
  }

  template <typename ValueT>
  void SYCLVector<ValueT>::freeDevicePointers(int* rows, ValueType* values) const
  {
    auto env = SYCLEnv::instance() ;
    auto& queue = env->internal()->queue() ;
    sycl::free(values,queue) ;
    sycl::free(rows,queue) ;
  }

  /*---------------------------------------------------------------------------*/

  template class ALIEN_EXPORT SYCLVector<Real>;

  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
