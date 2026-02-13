// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
#pragma once

#include <alien/core/impl/IMatrixImpl.h>
#include <alien/core/impl/MultiMatrixImpl.h>

#include <alien/core/block/Block.h>
#include <alien/core/block/VBlock.h>
#include <alien/core/block/VBlockOffsets.h>
#include <alien/data/ISpace.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/DistStructInfo.h>
#include <alien/kernels/simple_csr/SendRecvOp.h>
#include <alien/kernels/simple_csr/SimpleCSRBackEnd.h>
#include <alien/kernels/simple_csr/SimpleCSRInternal.h>
#include <alien/kernels/simple_csr/SimpleCSRPrecomp.h>

#include <alien/utils/StdTimer.h>
/*---------------------------------------------------------------------------*/

namespace Alien::SimpleCSRInternal
{

template <typename ValueT>
class SimpleCSRMatrixMultT;

}

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
class SimpleCSRMatrix : public IMatrixImpl
{
 public:
  // clang-format off
  static const bool                                    on_host_only = true ;
  typedef BackEnd::tag::simplecsr                      TagType ;
  typedef ValueT                                       ValueType;
  typedef SimpleCSRInternal::CSRStructInfo             CSRStructInfo;
  typedef SimpleCSRInternal::CSRStructInfo             ProfileType;
  typedef SimpleCSRInternal::DistStructInfo            DistStructInfo;
  typedef SimpleCSRInternal::MatrixInternal<ValueType> MatrixInternal;
  typedef typename ProfileType::IndexType              IndexType ;
  typedef Alien::StdTimer                              TimerType ;
  typedef TimerType::Sentry                            SentryType ;
  // clang-format on

 public:
  /** Constructeur de la classe */
  SimpleCSRMatrix()
  : IMatrixImpl(nullptr, AlgebraTraits<BackEnd::tag::simplecsr>::name())
  , m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
  , m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
  {}

  /** Constructeur de la classe */
  SimpleCSRMatrix(const MultiMatrixImpl* multi_impl)
  : IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::simplecsr>::name())
  , m_matrix(multi_impl ? multi_impl->vblock() != nullptr : false)
  , m_send_policy(SimpleCSRInternal::CommProperty::ASynch)
  , m_recv_policy(SimpleCSRInternal::CommProperty::ASynch)
  {}

  /** Destructeur de la classe */
  virtual ~SimpleCSRMatrix()
  {
#ifdef ALIEN_USE_PERF_TIMER
    m_timer.printInfo("SimpleCSR-MATRIX");
#endif
  }

  void setTraceMng(ITraceMng* trace_mng) { m_trace = trace_mng; }

 public:
  void free()
  { /* TODO */
  }
  void freeData()
  { /* TODO */
  }
  void clear() {}

  void allocate()
  {
    if (block()) {
      const Integer size = block()->size();
      m_matrix.getValues().resize((getCSRProfile().getNnz() + 1) * size * size);
    }
    else if (vblock()) {
      m_matrix.getValues().resize(getCSRProfile().getBlockNnz() + 1);
    }
    else {
      m_matrix.getValues().resize(getCSRProfile().getNnz() + 1);
    }
  }

  void scal(ValueType const* values)
  {
    m_matrix.scal(values) ;
  }

  CSRStructInfo& getCSRProfile() { return m_matrix.getCSRProfile(); }

  const CSRStructInfo& getCSRProfile() const { return m_matrix.getCSRProfile(); }

  const CSRStructInfo& getProfile() const { return m_matrix.getCSRProfile(); }

  const DistStructInfo& getDistStructInfo() const { return m_matrix_dist_info; }

  SimpleCSRInternal::CommProperty::ePolicyType getSendPolicy() const
  {
    return m_send_policy;
  }

  SimpleCSRInternal::CommProperty::ePolicyType getRecvPolicy() const
  {
    return m_recv_policy;
  }

  ValueType* getAddressData() { return m_matrix.getDataPtr(); }
  ValueType* data() { return m_matrix.getDataPtr(); }

  ValueType const* getAddressData() const { return m_matrix.getDataPtr(); }
  ValueType const* data() const { return m_matrix.getDataPtr(); }

  MatrixInternal* internal() { return &m_matrix; }

  MatrixInternal const* internal() const { return &m_matrix; }

  bool isParallel() const { return m_is_parallel; }

  Integer getLocalSize() const { return m_local_size; }

  Integer getLocalOffset() const { return m_local_offset; }

  Integer getGlobalSize() const { return m_global_size; }

  Integer getGhostSize() const { return m_ghost_size; }

  Integer getAllocSize() const
  {
    auto total_size = m_local_size + m_ghost_size;
    if (block())
      return total_size * block()->size();
    else if (vblock()) {
      return m_matrix_dist_info.m_block_offsets[total_size];
    }
    else
      return total_size;
  }

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
      m_own_block_size = block_size ;
  }

  IMessagePassingMng* getParallelMng()
  {
    return m_parallel_mng;
  }

  void sequentialStart()
  {
    m_local_offset = 0;
    m_local_size = getCSRProfile().getNRows();
    m_global_size = m_local_size;
    m_myrank = 0;
    m_nproc = 1;
    m_is_parallel = false;
    m_matrix_dist_info.m_local_row_size.resize(m_local_size);
    auto& profile = internal()->getCSRProfile();
    ConstArrayView<Integer> offset = profile.getRowOffset();
    for (Integer i = 0; i < m_local_size; ++i)
      m_matrix_dist_info.m_local_row_size[i] = offset[i + 1] - offset[i];
  }

  void parallelStart(ConstArrayView<Integer> offset, IMessagePassingMng* parallel_mng,
                     bool need_sort_ghost_col = false)
  {
    m_local_size = getCSRProfile().getNRows();
    m_parallel_mng = parallel_mng;
    // m_trace = parallel_mng->traceMng();
    if (m_parallel_mng == NULL) {
      m_local_offset = 0;
      m_global_size = m_local_size;
      m_myrank = 0;
      m_nproc = 1;
      m_is_parallel = false;
    }
    else {
      m_myrank = m_parallel_mng->commRank();
      m_nproc = m_parallel_mng->commSize();
      m_local_offset = offset[m_myrank];
      m_global_size = offset[m_nproc];
      m_is_parallel = (m_nproc > 1);
    }
    if (m_is_parallel) {
      if (need_sort_ghost_col)
        sortGhostCols(offset);
      if (block()) {
        m_matrix_dist_info.compute(
        m_nproc, offset, m_myrank, m_parallel_mng, getCSRProfile(), m_trace);
      }
      else if (vblock()) {
        m_matrix_dist_info.compute(m_nproc, offset, m_myrank, m_parallel_mng,
                                   getCSRProfile(), vblock(), distribution(), m_trace);
      }
      else {
        m_matrix_dist_info.compute(
        m_nproc, offset, m_myrank, m_parallel_mng, getCSRProfile(), m_trace);
      }
      m_ghost_size = m_matrix_dist_info.m_ghost_nrow;
    }
  }

  void sortGhostCols(ConstArrayView<Integer> offset)
  {
    IsLocal isLocal(offset, m_myrank);
    UniqueArray<ValueType>& values = m_matrix.getValues();
    ProfileType& profile = m_matrix.getCSRProfile();
    UniqueArray<Integer>& cols = profile.getCols();
    ConstArrayView<Integer> kcol = profile.getRowOffset();
    Integer next = 0;
    UniqueArray<Integer> gcols;
    UniqueArray<ValueType> gvalues;
    for (Integer irow = 0; irow < m_local_size; ++irow) {
      bool need_sort = false;
      Integer first = next;
      next = kcol[irow + 1];
      Integer row_size = next - first;
      for (Integer k = first; k < next; ++k) {
        if (!isLocal(cols[k])) {
          need_sort = true;
          break;
        }
      }
      if (need_sort) {
        gvalues.resize(row_size);
        gcols.resize(row_size);
        Integer local_count = 0;
        Integer ghost_count = 0;
        for (Integer k = first; k < next; ++k) {
          Integer col = cols[k];
          if (isLocal(col)) {
            cols[first + local_count] = col;
            values[first + local_count] = values[k];
            ++local_count;
          }
          else {
            gcols[ghost_count] = col;
            gvalues[ghost_count] = values[k];
            ++ghost_count;
          }
        }
        for (Integer k = 0; k < ghost_count; ++k) {
          cols[first + local_count] = gcols[k];
          values[first + local_count] = gvalues[k];
          ++local_count;
        }
      }
    }
  }

  /*
    DistStructInfo m_matrix_dist_info;
  */

  virtual SimpleCSRMatrix* cloneTo(const MultiMatrixImpl* multi) const
  {
    SimpleCSRMatrix* matrix = new SimpleCSRMatrix(multi);
    matrix->m_is_parallel = m_is_parallel;
    matrix->m_local_size = m_local_size;
    matrix->m_local_offset = m_local_offset;
    matrix->m_global_size = m_global_size;
    matrix->m_ghost_size = m_ghost_size;
    matrix->m_send_policy = m_send_policy;
    matrix->m_recv_policy = m_recv_policy;
    matrix->m_nproc = m_nproc;
    matrix->m_myrank = m_myrank;
    matrix->m_parallel_mng = m_parallel_mng;
    matrix->m_trace = m_trace;
    matrix->setBlockSize(blockSize()) ;
    matrix->m_matrix.copy(m_matrix);
    matrix->m_matrix_dist_info.copy(m_matrix_dist_info);
    return matrix;
  }

  void copy(SimpleCSRMatrix const& matrix)
  {
    m_is_parallel = matrix.m_is_parallel;
    m_local_size = matrix.m_local_size;
    m_local_offset = matrix.m_local_offset;
    m_global_size = matrix.m_global_size;
    m_ghost_size = matrix.m_ghost_size;
    m_send_policy = matrix.m_send_policy;
    m_recv_policy = matrix.m_recv_policy;
    m_nproc = matrix.m_nproc;
    m_myrank = matrix.m_myrank;
    m_parallel_mng = matrix.m_parallel_mng;
    m_trace = matrix.m_trace;
    if(blockSize()==matrix.blockSize())
      m_matrix.copy(matrix.m_matrix);
    else
    {
      auto nb_blocks = matrix.getCSRProfile().getNnz() + 1 ;
      m_matrix.copy(matrix.m_matrix,blockSize(),matrix.blockSize(),nb_blocks) ;
    }
    m_matrix_dist_info.copy(matrix.m_matrix_dist_info);
  }

  void copyProfile(SimpleCSRMatrix const& matrix)
  {
    m_is_parallel = matrix.m_is_parallel;
    m_local_size = matrix.m_local_size;
    m_local_offset = matrix.m_local_offset;
    m_global_size = matrix.m_global_size;
    m_ghost_size = matrix.m_ghost_size;
    m_send_policy = matrix.m_send_policy;
    m_recv_policy = matrix.m_recv_policy;
    m_nproc = matrix.m_nproc;
    m_myrank = matrix.m_myrank;
    m_parallel_mng = matrix.m_parallel_mng;
    m_trace = matrix.m_trace;
    m_matrix.getCSRProfile().copy(matrix.m_matrix.getCSRProfile());
    m_matrix_dist_info.copy(matrix.m_matrix_dist_info);
    if (vblock()) {
      auto& profile = m_matrix.getCSRProfile();
      const VBlock* block_sizes = vblock();
      auto& block_row_offset = profile.getBlockRowOffset();
      auto& block_cols = profile.getBlockCols();
      auto kcol = profile.kcol();
      auto cols = profile.cols();
      Integer offset = 0;
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        block_row_offset[irow] = offset;
        auto row_blk_size = block_sizes->size(m_local_offset + irow);
        for (auto k = kcol[irow]; k < kcol[irow + 1]; ++k) {
          block_cols[k] = offset;
          auto jcol = cols[k];
          auto col_blk_size = block_sizes->size(jcol);
          offset += row_blk_size * col_blk_size;
        }
      }
      block_row_offset[m_local_size] = offset;
      block_cols[kcol[m_local_size]] = offset;

      const Integer total_size = m_local_size + m_ghost_size;

      m_matrix_dist_info.m_block_sizes.resize(total_size);
      m_matrix_dist_info.m_block_offsets.resize(total_size + 1);

      offset = 0;
      for (Integer i = 0; i < m_local_size; ++i) {
        auto blk_size = block_sizes->size(m_local_offset + i);
        m_matrix_dist_info.m_block_sizes[i] = blk_size;
        m_matrix_dist_info.m_block_offsets[i] = offset;
        offset += blk_size;
      }
      for (Integer i = m_local_size; i < total_size; ++i) {
        auto blk_size = block_sizes->size(m_matrix_dist_info.m_recv_info.m_uids[i - m_local_size]);
        m_matrix_dist_info.m_block_sizes[i] = blk_size;
        m_matrix_dist_info.m_block_offsets[i] = offset;
        offset += blk_size;
      }
      m_matrix_dist_info.m_block_offsets[total_size] = offset;
    }
  }

  void notifyChanges()
  {
    m_matrix.notifyChanges();
  }

  void endUpdate()
  {
    if (m_matrix.needUpdate()) {
      m_matrix.endUpdate();
      this->updateTimestamp();
    }
  }

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

  MatrixInternal m_matrix;
  bool m_is_parallel = 0;
  Integer m_local_size = 0;
  Integer m_local_offset = 0;
  Integer m_global_size = 0;
  Integer m_ghost_size = 0;
  DistStructInfo m_matrix_dist_info;
  SimpleCSRInternal::CommProperty::ePolicyType m_send_policy;
  SimpleCSRInternal::CommProperty::ePolicyType m_recv_policy;
  IMessagePassingMng* m_parallel_mng = nullptr;
  Integer m_own_block_size = 1;
  Integer m_nproc = 1;
  Integer m_myrank = 0;
  ITraceMng* m_trace = nullptr;

  friend class SimpleCSRInternal::SimpleCSRMatrixMultT<ValueType>;

 private:
  mutable TimerType m_timer;

 public:
  TimerType& timer() const
  {
    return m_timer;
  }
};

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
