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

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRInternal.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

/*---------------------------------------------------------------------------*/

#define USE_VMAP

#ifdef USE_VMAP
#include <alien/utils/VMap.h>
#undef KEY_OF
#undef VALUE_OF
#undef NVALUE_OF
#define KEY_OF(i) (i).key()
//#define VALUE_OF(i) (i).value().first
#define VALUE_OF(i) (i).value().first
#define NVALUE_OF(i) (i).value().second
#else /* USE_VMAP */
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <utility>
#include <vector>

#undef KEY_OF
#undef VALUE_OF
#define KEY_OF(i) (i)->first
#define VALUE_OF(i) (i)->second.m_build_position
#endif /* USE_VMAP */

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::StreamMatrixBuilderT(Matrix& matrix, bool init_and_start)
: m_matrix(matrix)
, m_col_ordering(eUndef)
, m_state(eNone)
{
  if (init_and_start) {
    init();
    start();
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::StreamMatrixBuilderT(
BlockMatrix& matrix, bool init_and_start)
: m_matrix(matrix)
, m_col_ordering(eUndef)
, m_state(eNone)
{
  if (init_and_start) {
    init();
    start();
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::StreamMatrixBuilderT(IMatrix& matrix, bool init_and_start)
: m_matrix(matrix)
, m_col_ordering(eUndef)
, m_state(eNone)
{
  if (init_and_start) {
    init();
    start();
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamMatrixBuilderT<ValueT>::~StreamMatrixBuilderT()
{
  _freeInserters();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamMatrixBuilderT<ValueT>::Inserter&
StreamMatrixBuilderT<ValueT>::getNewInserter()
{
  Inserter* inserter = new Inserter(this, m_inserters.size());
  m_inserters.add(inserter);
  return *inserter;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamMatrixBuilderT<ValueT>::Inserter&
StreamMatrixBuilderT<ValueT>::getInserter(Integer id)
{
  Integer size = m_inserters.size();
  if ((id < 0) || (id >= size))
    FatalErrorException(A_FUNCINFO, "Bad Inserter id");
  return *m_inserters[id];
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::finalize()
{
  bool has_error = false;
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter) {
    Filler& filler = **iter;
    if (!filler.isEnd()) {
      if (m_trace)
        m_trace->warning() << "Inserter #" << filler.getId() << " not at final index";
      has_error = true;
    }
  }
  if (has_error)
    throw FatalErrorException(A_FUNCINFO, "All inserters are not at final index");

  m_matrix_impl->updateTimestamp();
  m_matrix.impl()->unlock();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::end()
{
  _freeInserters();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::allocate()
{
  computeProfile();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::_freeInserters()
{
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter)
    delete *iter;
  m_inserters.resize(0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::init()
{
  if (m_state == eInit)
    return;

  m_matrix.impl()->lock();

  m_matrix_impl = &m_matrix.impl()->template get<BackEnd::tag::simplecsr>(true);

  // const ISpace& space = m_matrix_impl->rowSpace();
  //  if (space != m_matrix_impl->colSpace())
  //   throw FatalErrorException(
  //       "stream matrix builder must be used with square matrix");

  const MatrixDistribution& dist = m_matrix_impl->distribution();

  m_parallel_mng = dist.parallelMng();

  m_matrix_impl->free();

  m_local_size = dist.localRowSize();
  m_global_size = dist.globalRowSize();
  m_local_offset = dist.rowOffset();
  const VBlock* vblock = m_matrix.impl()->vblock();
  // XT 15/10/2015 It seems that sizes don't need block information even with fixed size
  // block
  if (vblock)
    throw FatalErrorException(
    A_FUNCINFO, "This builder works only with fixed block size");

  m_col_ordering = eUndef;

  m_state = eInit;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::start()
{
  m_matrix_impl->free();
  m_state = ePrepared;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::fillZero()
{
  m_matrix_impl->internal().getValues().fill(0.);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamMatrixBuilderT<ValueT>::computeProfile()
{
  /**
   * Purpose : build the matrix profile
   * store in inserter the position in the CSR matrix to avoid searching algorithm
   * The CSR matrix has an extra position for ghost row equivalent to /dev/null
   * When filling ghost row position, entries has written in that null position
   */
  // ALIEN_ASSERT((m_state == ePrepared),("Unexpected state: %d vs
  // %d",m_state,ePrepared));

  // InternalSpace::StructInfo const& info = m_matrix.space().structInfo() ;

  // Attributes from classes relocated into this method (never used elsewhere)

  class InsBuildPos
  {
   public:
    Integer m_build_position;
    // list of doublet inserter position an inserter reference
    std::vector<std::pair<BaseInserter*, Integer>> m_inserter_vector;

    InsBuildPos()
    {
      m_build_position = -1;
      m_inserter_vector.reserve(1);
    }
  };
  typedef std::pair<Integer, Integer> BuildPos;
#ifdef USE_VMAP
  typedef VMap<Integer, BuildPos> RowCols;
#else /* USE_VMAP */
  typedef std::map<Integer, InsBuildPos> RowCols;
#endif /* USE_VMAP */
  UniqueArray<RowCols> row_cols;

  if (m_parallel_mng == NULL) {
    m_myrank = 0;
    m_nproc = 1;
  }
  else {
    m_myrank = m_parallel_mng->commRank();
    m_nproc = m_parallel_mng->commSize();
  }

  m_ghost_size = 0;
  m_offset.resize(m_nproc + 1);
  {
    Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                         ConstArrayView<Integer>(1, &m_local_offset), m_offset.subView(0, m_nproc));
  }
  m_offset[m_nproc] = m_global_size;

  // Utilitaire local (en attendant les lambda-functions)
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
  } isLocal(m_offset, m_myrank);

  SimpleCSRInternal::CSRStructInfo& profile = m_matrix_impl->internal().getCSRProfile();
  profile.init(m_local_size);

  m_row_size.resize(m_local_size);
  m_ghost_row_size.resize(m_local_size);
  m_ghost_row_size.fill(0);
  UniqueArray<Integer>& upper_diag_offset = profile.getUpperDiagOffset();
  if (m_order_row_cols_opt) {
    profile.setDiagFirst(false);
    upper_diag_offset.resize(m_local_size);
    m_row_size.fill(0);
    row_cols.resize(m_local_size);
  }
  else {
    profile.setDiagFirst(true);
    m_row_size.fill(1);
    row_cols.resize(m_local_size);
    for (Integer row = 0; row < m_local_size; ++row) {
      row_cols[row][m_local_offset + row].first = 0;
    }
  }

  // LOOP on inserter to compute filling position
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter) {
    BaseInserter* ins = *iter;
    ins->m_data_index.resize(ins->count());
    ins->m_n.add(0); // position de end
    if (m_trace)
      m_trace->info() << "Inserter id=" << ins->getId() << " count=" << ins->count();
    for (Integer i = 0; i < ins->count(); ++i) {
      Integer row = ins->m_row_index[i] - m_local_offset;
      Integer col = ins->m_col_index[i];
      if (row == m_local_size) {
        // case of ghost row
        // save -1, latter will be turn to the null position of the CSR structure
        ins->m_data_index[i] = -1;
        if (m_trace)
          m_trace->info() << "Ghost Row : " << i << " " << ins->m_row_index[i];
      }
      else {
#ifdef USE_VMAP
        std::pair<typename RowCols::iterator, bool> finder = row_cols[row].insert(col);
#else /* USE_VMAP */
        std::pair<typename RowCols::iterator, bool> finder =
        row_cols[row].insert(std::pair<Integer, InsBuildPos>(col, InsBuildPos()));
#endif /* USE_VMAP */
        auto inner_iter = finder.first;
        if (finder.second) {
          Integer k = m_row_size[row];
          Integer gk = m_ghost_row_size[row];
          if (isLocal(col)) {
            VALUE_OF(inner_iter) = k - gk;
            ins->m_data_index[i] = k - gk;
          }
          else {
            // be careful, position will increment with ghost_offset after
            VALUE_OF(inner_iter) = gk;
            ins->m_data_index[i] = gk;
            ++m_ghost_row_size[row];
          }
          ++m_row_size[row];
        }
        else {
          ins->m_data_index[i] = VALUE_OF(inner_iter);
        }
      }
    }
  }

  ArrayView<Integer> row_offsets =
  m_matrix_impl->internal().getCSRProfile().getRowOffset();
  m_matrix_size = 0;
  for (Integer row = 0; row < m_local_size; ++row) {
    row_offsets[row] = m_matrix_size;
    m_matrix_size += m_row_size[row];
  }
  row_offsets[m_local_size] = m_matrix_size;

  UniqueArray<Integer> kcols;
  if (m_order_row_cols_opt)
    kcols.resize(m_matrix_size);

  profile.allocate();
  ArrayView<Integer> cols = profile.getCols();

  m_matrix_impl->allocate();
  m_matrix_impl->internal().getValues().fill(0.);

  Integer icount = 0;
  Integer offset = 0;
  if (m_order_row_cols_opt) {
    for (Integer row = 0; row < m_local_size; ++row) {
      Integer ghost_offset = m_row_size[row] - m_ghost_row_size[row];
      int ordered_idx = 0;
      for (typename RowCols::iterator iter = row_cols[row].begin();
           iter != row_cols[row].end(); ++iter) {
        Integer col_uid = KEY_OF(iter);
        // increment position for ghost cols
        if (!isLocal(col_uid))
          VALUE_OF(iter) += ghost_offset;
        cols[offset + ordered_idx] = col_uid;
        kcols[offset + VALUE_OF(iter)] = offset + ordered_idx;
        if (col_uid == row + m_local_offset) {
          upper_diag_offset[row] = offset + ordered_idx;
        }
        ++ordered_idx;
        ++icount;
      }
      offset += m_row_size[row];
    }
  }
  else {
    for (Integer row = 0; row < m_local_size; ++row) {
      if (m_trace)
        m_trace->info() << "ROW(" << row << ")";
      Integer ghost_offset = m_row_size[row] - m_ghost_row_size[row];
      // int ordered_idx = 0;
      for (typename RowCols::iterator iter = row_cols[row].begin();
           iter != row_cols[row].end(); ++iter) {
        Integer col_uid = KEY_OF(iter);
        // increment position for ghost cols
        if (!isLocal(col_uid))
          VALUE_OF(iter) += ghost_offset;
        cols[offset + VALUE_OF(iter)] = col_uid;
        ++icount;
      }
      offset += m_row_size[row];
    }
  }
  // ALIEN_ASSERT((icount==m_matrix_size),("matrix total size problem")) ;
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter) {
    BaseInserter* ins = *iter;
    for (Integer i = 0; i < ins->count(); ++i) {
      Integer row = ins->m_row_index[i] - m_local_offset;
      if (row == m_local_size) {
        // Ghost row
        ins->m_data_index[i] = m_matrix_size; // equivalent to the null position
      }
      else {
        Integer col = ins->m_col_index[i];
        Integer ghost_offset =
        (isLocal(col) ? 0 : m_row_size[row] - m_ghost_row_size[row]);
        if (m_order_row_cols_opt) {
          Integer build_pos = ins->m_data_index[i] + ghost_offset;
          ins->m_data_index[i] = kcols[row_offsets[row] + build_pos];
        }
        else
          ins->m_data_index[i] += row_offsets[row] + ghost_offset;
      }
    }

    Integer neq = 1;
    if (m_matrix_impl->block())
      neq = m_matrix_impl->block()->size();
    const Integer block_size = neq * neq;
    ins->setMatrixValues(m_matrix_impl->internal().getDataPtr(), block_size);
    ins->m_col_index.dispose();
    ins->m_row_index.dispose();
  }

  if (m_nproc > 1)
    m_matrix_impl->parallelStart(m_offset, m_parallel_mng);
  else
    m_matrix_impl->sequentialStart();
  m_col_ordering = eOwnAndGhost;
  m_state = eStart;

  m_row_size.resize(0);
  m_ghost_row_size.resize(0);
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
