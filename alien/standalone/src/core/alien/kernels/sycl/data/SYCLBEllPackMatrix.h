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
/*
 * BEllPackMatrix.h
 *
 *  Created on: Nov 20, 2021
 *      Author: gratienj
 */

#pragma once

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/data/ISpace.h>

#include <alien/kernels/sycl/SYCLPrecomp.h>

#include <alien/kernels/sycl/data/BEllPackStructInfo.h>
#include <alien/kernels/sycl/data/DistStructInfo.h>

#include <alien/kernels/sycl/SYCLBackEnd.h>

#include <alien/utils/StdTimer.h>

namespace Alien::SYCLInternal
{
template <typename ValueT>
class SYCLBEllPackMatrixMultT;

template <typename ValueT, int BlockSize>
class MatrixInternal;
} // namespace Alien::SYCLInternal

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <int BlockSize, typename IndexT>
class BEllPackStructInfo;

template <typename ValueT>
class SYCLVector;

template <typename ValueT>
class ALIEN_EXPORT SYCLBEllPackMatrix : public IMatrixImpl
{
 public:
  // clang-format off
  static const bool                                         on_host_only = false ;
  typedef ValueT                                            ValueType;
  typedef ValueT                                            value_type ;

  typedef SimpleCSRInternal::DistStructInfo                 DistStructInfo;
  typedef SYCLInternal::MatrixInternal<ValueType,1024>      MatrixInternal1024;


  typedef BEllPackStructInfo<1024,int>                      ProfileInternal1024;
  typedef BEllPackStructInfo<1024,int>                      ProfileType;
  typedef typename ProfileType::IndexType                   IndexType ;

  typedef Alien::StdTimer                                   TimerType ;
  typedef TimerType::Sentry                                 SentryType ;
  // clang-format on

 public:
  /** Constructeur de la classe */
  SYCLBEllPackMatrix()
  : IMatrixImpl(nullptr, AlgebraTraits<BackEnd::tag::sycl>::name())
  , m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
  , m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
  {}

  /** Constructeur de la classe */
  SYCLBEllPackMatrix(const MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::sycl>::name())
  , m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
  , m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
  {}

  /** Destructeur de la classe */
  virtual ~SYCLBEllPackMatrix();

  void setTraceMng(ITraceMng* trace_mng) { m_trace = trace_mng; }

  ProfileType const& getProfile() const
  {
    return *m_profile1024;
  }

  ValueType* getAddressData();
  ValueType* data();

  ValueType const* getAddressData() const;
  ValueType const* data() const;

 public:
  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                  Integer local_offset,
                  Integer global_size,
                  std::size_t nrows,
                  int const* kcol,
                  int const* cols,
                  SimpleCSRInternal::DistStructInfo const& matrix_dist_info);

  SYCLBEllPackMatrix* cloneTo(const MultiMatrixImpl* multi) const;

  bool isParallel() const { return m_is_parallel; }

  Integer getLocalSize() const { return m_local_size; }

  Integer getLocalOffset() const { return m_local_offset; }

  Integer getGlobalSize() const { return m_global_size; }

  Integer getGhostSize() const { return m_ghost_size; }

  Integer getAllocSize() const { return m_local_size + m_ghost_size; }

  bool setMatrixValues(Arccore::Real const* values, bool only_host);

  void notifyChanges();
  void endUpdate();

  void mult(SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;
  void endDistMult(SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;

  void addLMult(ValueType alpha, SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;
  void addUMult(ValueType alpha, SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;

  void multInvDiag(SYCLVector<ValueType>& y) const;
  void computeInvDiag(SYCLVector<ValueType>& y) const;

  const DistStructInfo& getDistStructInfo() const { return m_matrix_dist_info; }

  Alien::SimpleCSRInternal::CommProperty::ePolicyType getSendPolicy() const
  {
    return m_send_policy;
  }

  Alien::SimpleCSRInternal::CommProperty::ePolicyType getRecvPolicy() const
  {
    return m_recv_policy;
  }

  MatrixInternal1024* internal() { return m_matrix1024; }

  MatrixInternal1024 const* internal() const { return m_matrix1024; }

 private:
  class IsLocal
  {
   public:
    IsLocal(const ConstArrayView<Integer> offset, const Integer myrank)
    : m_offset(offset)
    , m_myrank(myrank)
    {}
    bool operator()(Arccore::Integer col) const
    {
      return (col >= m_offset[m_myrank]) && (col < m_offset[m_myrank + 1]);
    }

   private:
    const ConstArrayView<Integer> m_offset;
    const Integer m_myrank;
  };

  // clang-format off
  ProfileInternal1024*                    m_profile1024     = nullptr ;
  MatrixInternal1024*                     m_matrix1024      = nullptr;

  ProfileInternal1024*                    m_ext_profile1024 = nullptr ;

  int                                     m_block_size      = 1024 ;
  std::vector<int>                        m_block_row_offset ;
  std::vector<int>                        m_ext_block_row_offset ;

  bool                                    m_is_parallel  = false;
  IMessagePassingMng*                     m_parallel_mng = nullptr;
  Integer                                 m_nproc        = 1;
  Integer                                 m_myrank       = 0;

  Integer                                 m_local_size   = 0;
  Integer                                 m_local_offset = 0;
  Integer                                 m_global_size  = 0;
  Integer                                 m_ghost_size   = 0;

  SimpleCSRInternal::DistStructInfo            m_matrix_dist_info;
  SimpleCSRInternal::CommProperty::ePolicyType m_send_policy;
  SimpleCSRInternal::CommProperty::ePolicyType m_recv_policy;
  ITraceMng*                                   m_trace = nullptr;
  // clang-format on

  // From unsuccessful try to implement multiplication.
  friend class SYCLInternal::SYCLBEllPackMatrixMultT<ValueType>;

#ifdef ALIEN_USE_PERF_TIMER
 private:
  mutable TimerType m_timer;

 public:
  TimerType& timer() const
  {
    return m_timer;
  }
#endif
};

//extern template class SYCLBEllPackMatrix<double>;
} // namespace Alien
