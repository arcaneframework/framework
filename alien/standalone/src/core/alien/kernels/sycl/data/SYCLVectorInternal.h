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

#pragma once

#include <alien/kernels/sycl/SYCLPrecomp.h>

#ifdef USE_SYCL2020
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

#include "SYCLEnv.h"
#include "SYCLEnvInternal.h"
/*---------------------------------------------------------------------------*/

namespace Alien::SYCLInternal
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
  typedef std::unique_ptr<ValueBufferType> ValueBufferPtrType;
  // clang-format on

 public:
  VectorInternal(ValueType const* ptr, std::size_t size)
  : m_values(ptr, sycl::range<1>(size))
  {
    m_values.set_final_data(nullptr);
  }

  virtual ~VectorInternal() {}

  ValueBufferType& values()
  {
    return m_values;
  }

  ValueBufferType& values() const
  {
    return m_values;
  }

  ValueBufferType& ghostValues(Integer ghost_size) const
  {
    if (m_ghost_values.get() == nullptr || ghost_size > m_ghost_size) {
      m_ghost_size = ghost_size;
      m_ghost_values.reset(new ValueBufferType(m_ghost_size));
    }
    return *m_ghost_values;
  }

  void copyValuesToHost(std::size_t size, ValueT* ptr)
  {
    auto h_values = m_values.get_host_access();
    for (std::size_t i = 0; i < size; ++i)
      ptr[i] = h_values[i];
  }

  void copy(ValueBufferType& src)
  {
    auto env = SYCLEnv::instance() ;
    env->internal()->queue().submit([&](sycl::handler& cgh)
                                     {
                                       auto access_x = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                                       auto access_src = src.template get_access<sycl::access::mode::read>(cgh);
                                       cgh.copy(access_src,access_x) ;
                                     }) ;
  }

  //VectorInternal<ValueT>* clone() const { return new VectorInternal<ValueT>(*this); }

  // clang-format off
  mutable ValueBufferType    m_values;

  mutable Integer            m_ghost_size = 0 ;
  mutable ValueBufferPtrType m_ghost_values;
  // clang-format on
};

/*---------------------------------------------------------------------------*/

} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/
