// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
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

#include <alien/handlers/accelerator/HCSRViewT.h>

#include <alien/kernels/sycl/data/BEllPackStructInfo.h>
#include <alien/kernels/sycl/data/SYCLDistStructInfo.h>

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
  static constexpr bool                                     on_host_only = false ;
  typedef BackEnd::tag::sycl                                TagType ;
  typedef ValueT                                            ValueType;
  typedef ValueT                                            value_type ;

  typedef SYCLInternal::SYCLDistStructInfo                  DistStructInfo;
  typedef SYCLInternal::MatrixInternal<ValueType,1024>      MatrixInternal1024;


  typedef BEllPackStructInfo<1024,int>                      ProfileInternal1024;
  typedef BEllPackStructInfo<1024,int>                      ProfileType;
  typedef typename ProfileType::IndexType                   IndexType ;

  typedef Alien::StdTimer                                   TimerType ;
  typedef TimerType::Sentry                                 SentryType ;


  using HCSRView = HCSRViewT<SYCLBEllPackMatrix<ValueType>>;
  // clang-format on

 public:
  /** Constructeur de la classe */
  SYCLBEllPackMatrix() ;

  /** Constructeur de la classe */
  SYCLBEllPackMatrix(const MultiMatrixImpl* multi_impl) ;

  /** Destructeur de la classe */
  virtual ~SYCLBEllPackMatrix();

  void setTraceMng(ITraceMng* trace_mng) { m_trace = trace_mng; }

  ProfileType const& getProfile() const
  {
    return *m_profile1024;
  }

  HCSRView hcsrView(BackEnd::Memory::eType memory, int nrows, int nnz) const;

  ValueType* getAddressData();
  ValueType* data();

  ValueType const* getAddressData() const;
  ValueType const* data() const;

  IMessagePassingMng* getParallelMng()
  {
    return m_parallel_mng;
  }

 public:
  bool initMatrix(Arccore::MessagePassing::IMessagePassingMng* parallel_mng,
                  Integer local_offset,
                  Integer global_size,
                  std::size_t nrows,
                  int const* kcol,
                  int const* cols,
                  SimpleCSRInternal::DistStructInfo const& matrix_dist_info,
                  int block_size=1);

  SYCLBEllPackMatrix* cloneTo(const MultiMatrixImpl* multi) const;

  bool isParallel() const { return m_is_parallel; }

  Integer getLocalSize() const { return m_local_size; }

  Integer getLocalOffset() const { return m_local_offset; }

  Integer getGlobalSize() const { return m_global_size; }

  Integer getGhostSize() const { return m_ghost_size; }

  Integer getAllocSize() const { return m_local_size + m_ghost_size; }

  Integer blockSize() const
  {
    if (block())
    {
       return block()->size();
    }
    else if (vblock()) {
      return 1 ;
    }
    else {
      return m_own_block_size ;
    }
  }

  void setBlockSize(Integer block_size)
  {
    if(this->m_multi_impl)
      const_cast<MultiMatrixImpl*>(this->m_multi_impl)->setBlockInfos(block_size) ;
    else
      m_own_block_size = block_size;
  }

  bool setMatrixValues(Arccore::Real const* values, bool only_host);

  void copy(SYCLBEllPackMatrix const& matrix) ;

  void notifyChanges();
  void endUpdate();

  void mult(SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;
  void endDistMult(SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;

  void addLMult(ValueType alpha, SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;
  void addUMult(ValueType alpha, SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;

  void multDiag(SYCLVector<ValueType> const& x, SYCLVector<ValueType>& y) const;

  void multDiag(SYCLVector<ValueType>& y) const ;
  void computeDiag(SYCLVector<ValueType>& y) const;

  void multInvDiag(SYCLVector<ValueType>& y) const;
  void computeInvDiag(SYCLVector<ValueType>& y) const;

  void scal(SYCLVector<ValueType> const& diag) ;

  const DistStructInfo& getDistStructInfo() const { return m_matrix_dist_info; }

  Alien::SimpleCSRInternal::CommProperty::ePolicyType getSendPolicy() const
  {
    return m_send_policy;
  }

  Alien::SimpleCSRInternal::CommProperty::ePolicyType getRecvPolicy() const
  {
    return m_recv_policy;
  }

  MatrixInternal1024* internal() { return m_matrix1024.get(); }

  MatrixInternal1024 const* internal() const { return m_matrix1024.get(); }

  void allocateDevicePointers(std::size_t nrows,
                              std::size_t nnz,
                              IndexType** rows,
                              IndexType** ncols,
                              IndexType** cols,
                              ValueType** values) const ;

  void freeDevicePointers(IndexType* rows,
                          IndexType* ncols,
                          IndexType* cols,
                          ValueType* values) const ;

  void copyDevicePointers(std::size_t nrows,
                          std::size_t nnz,
                          IndexType* rows,
                          IndexType* ncols,
                          IndexType* cols,
                          ValueType* values) const ;


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
  std::unique_ptr<ProfileInternal1024>    m_profile1024;
  std::unique_ptr<MatrixInternal1024>     m_matrix1024;

  std::unique_ptr<ProfileInternal1024>    m_ext_profile1024;

  int                                     m_ellpack_size = 1024 ;
  std::vector<int>                        m_block_row_offset ;
  std::vector<int>                        m_ext_block_row_offset ;

  Integer                                 m_own_block_size = 1 ;

  bool                                    m_is_parallel  = false;
  IMessagePassingMng*                     m_parallel_mng = nullptr;
  Integer                                 m_nproc        = 1;
  Integer                                 m_myrank       = 0;

  Integer                                 m_local_size   = 0;
  Integer                                 m_local_offset = 0;
  Integer                                 m_global_size  = 0;
  Integer                                 m_ghost_size   = 0;


  //SimpleCSRInternal::DistStructInfo            m_matrix_dist_info;
  DistStructInfo                               m_matrix_dist_info;
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
