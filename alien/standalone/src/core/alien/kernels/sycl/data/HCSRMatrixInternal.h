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

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>
#include "alien/kernels/sycl/data/HCSRMatrix.h"

#include "SYCLEnv.h"
#include "SYCLEnvInternal.h"
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
  class MatrixInternal
  {
   public:
    // clang-format off
    typedef ValueT                           ValueType;
    typedef MatrixInternal<ValueType>        ThisType;
    typedef SimpleCSRInternal::CSRStructInfo ProfileType;
    typedef typename ProfileType::IndexType  IndexType ;
    typedef sycl::buffer<ValueType, 1>       ValueBufferType;
    typedef sycl::buffer<IndexType, 1>       IndexBufferType;
    // clang-format on

   public:
    MatrixInternal(ProfileType* profile)
    : m_profile(profile)
    , m_values(sycl::range<1>(profile->getNnz()+1))
    , m_kcol(profile->kcol(),profile->getNRows()+1)
    , m_cols(profile->cols(),profile->getNnz())
    {
      m_values.set_final_data(nullptr);
    }

    virtual ~MatrixInternal() {}


    ProfileType& getCSRProfile() {
      assert(m_profile) ;
      return *m_profile ;
    }

    ProfileType const& getCSRProfile() const {
      assert(m_profile) ;
      return *m_profile ;
    }

    void setValues(ValueT value)
    {
        auto env = SYCLEnv::instance() ;
        env->internal()->queue().submit([&](sycl::handler& cgh)
                                         {
                                           auto access_x = m_values.template get_access<sycl::access::mode::read_write>(cgh);
                                           cgh.fill(access_x,ValueT()) ;
                                         }) ;
    }

    /*
    ConstArrayView<ValueType> getValues() const
    {
      return ConstArrayView<ValueType>(m_h_values.size(),m_h_values.data());
    }

    ArrayView<ValueType> getValues() {
      return ArrayView<ValueType>(m_h_values.size(),m_h_values.data());
    }
    */

    ValueBufferType& values()
    {
      return m_values;
    }

    ValueBufferType& values() const
    {
      return m_values;
    }

    IndexBufferType& kcol() {
      return m_kcol;
    }

    IndexBufferType const& kcol() const {
      return m_kcol;
    }

    IndexBufferType& cols() {
      return m_cols;
    }

    IndexBufferType const& cols() const {
      return m_cols;
    }


    void copyValuesToHost(std::size_t nnz, ValueT* ptr)
    {
      auto h_values = m_values.get_host_access();
      for (std::size_t i = 0; i < nnz; ++i)
        ptr[i] = h_values[i];
    }
    ProfileType* m_profile = nullptr ;

    mutable ValueBufferType        m_values;
    mutable IndexBufferType        m_kcol;
    mutable IndexBufferType        m_cols;
  };

  /*---------------------------------------------------------------------------*/

  } // namespace Alien::SYCLInternal

  template <typename ValueT>
  HCSRMatrix<ValueT>::HCSRMatrix()
  : IMatrixImpl(nullptr, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {
    m_profile.reset(new ProfileType()) ;
  }

  //! Constructeur avec association ? un MultiImpl
  template <typename ValueT>
  HCSRMatrix<ValueT>::HCSRMatrix(const MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::hcsr>::name())
  , m_local_size(0)
  {
    m_profile.reset(new ProfileType()) ;
  }


  template <typename ValueT>
  HCSRMatrix<ValueT>::~HCSRMatrix()
  {

  }
/*---------------------------------------------------------------------------*/
template <typename ValueT>
void HCSRMatrix<ValueT>::allocate()
{
  assert(m_profile.get()) ;
  m_internal.reset(new InternalType(m_profile.get())) ;
}


/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
