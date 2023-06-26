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

#include <alien/core/block/IBlockBuilder.h>
#include <alien/core/block/VBlock.h>
#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRInternal.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#define USE_VMAP

#ifdef USE_VMAP
#include <alien/utils/VMap.h>
#undef KEY_OF
#undef VALUE_OF
#define KEY_OF(i) (i).key()
#define VALUE_OF(i) (i).value()
#else /* USE_VMAP */
#include <algorithm>
#include <cstdlib>
#include <map>
#include <utility>

#undef KEY_OF
#undef VALUE_OF
#define KEY_OF(i) (i)->first
#define VALUE_OF(i) (i)->second
#endif /* USE_VMAP */

/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamVBlockMatrixBuilderT<ValueT>::StreamVBlockMatrixBuilderT(
VBlockMatrix& matrix, bool init_and_start)
: m_matrix(matrix)
, m_matrix_impl(NULL)
, m_local_size(0)
, m_global_size(0)
, m_local_offset(0)
, m_ghost_size(-1)
, m_block_ghost_size(-1)
, m_matrix_size(0)
, m_block_matrix_size(0)
, m_myrank(-1)
, m_nproc(-1)
, m_col_ordering(eUndef)
, m_parallel_mng(NULL)
, m_trace(NULL)
, m_state(eNone)
{
  if (init_and_start) {
    init();
    start();
  }
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
StreamVBlockMatrixBuilderT<ValueT>::~StreamVBlockMatrixBuilderT()
{
  _freeInserters();
}

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::finalize()
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

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::end()
{
  _freeInserters();
}
/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamVBlockMatrixBuilderT<ValueT>::Inserter&
StreamVBlockMatrixBuilderT<ValueT>::getNewInserter()
{
  Inserter* inserter = new Inserter(this, m_inserters.size());
  m_inserters.add(inserter);
  return *inserter;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
typename StreamVBlockMatrixBuilderT<ValueT>::Inserter&
StreamVBlockMatrixBuilderT<ValueT>::getInserter(Integer id)
{
  if ((id < 0) || (id >= (Integer)m_inserters.size()))
    throw FatalErrorException(A_FUNCINFO, "Bad Inserter id");
  return *m_inserters[id];
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::allocate()
{
  computeProfile();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::_freeInserters()
{
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter)
    delete *iter;
  m_inserters.resize(0);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::init()
{
  if (m_state == eInit)
    return;

  const ISpace& space = m_matrix.rowSpace();
  // if (space != m_matrix.colSpace())
  //  throw FatalErrorException(
  //      "stream matrix builder must be used with square matrix");

  const MatrixDistribution& dist = m_matrix.impl()->distribution();
  const VBlock* vblock = m_matrix.impl()->vblock();

  if (!vblock)
    throw FatalErrorException(
    A_FUNCINFO, "Space is not variable block size - Can't use block builder");

  // m_trace = space.traceMng();
  m_parallel_mng = dist.parallelMng();

  m_matrix.impl()->lock();
  m_matrix_impl = &m_matrix.impl()->template get<BackEnd::tag::simplecsr>(true);
  m_matrix_impl->free();

  m_local_size = dist.localRowSize();
  m_global_size = dist.globalRowSize();
  m_local_offset = dist.rowOffset();
  m_col_ordering = eUndef;

  m_state = eInit;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::start()
{
  m_matrix_impl->free();
  m_state = ePrepared;
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::fillZero()
{
  m_matrix_impl->internal().getValues().fill(0.);
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
const VBlock*
StreamVBlockMatrixBuilderT<ValueT>::vblock() const
{
  return m_matrix.impl()->vblock();
}

/*---------------------------------------------------------------------------*/

template <typename ValueT>
void StreamVBlockMatrixBuilderT<ValueT>::computeProfile()
{
  /**
   * Purpose : build the matrix profile
   * store in inserter the position in the CSR matrix to avoid searching algorithm
   * The CSR matrix has an extra position for ghost row equivalent to /dev/null
   * When filling ghost row position, entries has written in that null position
   */
  // ALIEN_ASSERT((m_state == ePrepared),("Unexpected state: %d vs
  // %d",m_state,ePrepared));

  // InternalSpace::StructInfo const& info = m_matrix.space().structInfo();

  // Attributes from classes relocated into this method (never used elsewhere)

#ifdef USE_VMAP
  typedef VMap<Integer, Integer> RowCols;
#else /* USE_VMAP */
  typedef std::map<Integer, Integer> RowCols;
#endif /* USE_VMAP */
  UniqueArray<RowCols> row_cols;
  UniqueArray<RowCols> block_row_cols;

  if (m_parallel_mng == NULL) {
    m_myrank = 0;
    m_nproc = 1;
  }
  else {
    m_myrank = m_parallel_mng->commRank();
    m_nproc = m_parallel_mng->commSize();
  }

  const VBlock* block_sizes = this->vblock();

  m_ghost_size = 0;
  m_block_ghost_size = 0;
  m_offset.resize(m_nproc + 1);
  Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                       ConstArrayView<Integer>(1, &m_local_offset), m_offset.subView(0, m_nproc));
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
  m_row_size.fill(1);
  m_ghost_row_size.fill(0);
  row_cols.resize(m_local_size);
  block_row_cols.resize(m_local_size);
  for (Integer row = 0; row < m_local_size; ++row) {
    row_cols[row][m_local_offset + row] = 0;
    block_row_cols[row][m_local_offset + row] = 0;
  }

  // Pour les blocs
  m_block_row_size.resize(m_local_size);
  m_block_ghost_row_size.resize(m_local_size);
  m_block_row_size.fill(0);
  m_block_ghost_row_size.fill(0);
  for (Integer row = 0; row < m_local_size; ++row) {
    const Integer id = m_local_offset + row;
    const Integer block_size = block_sizes->size(id);
    m_block_row_size[row] += block_size * block_size;
  }

  UniqueArray<UniqueArray<Integer>> m_block_data_index(m_inserters.size());

  Integer nb_ins = 0;
  // LOOP on inserter to compute filling position
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter) {
    BaseInserter* ins = *iter;
    ins->m_data_index.resize(ins->count());
    m_block_data_index[nb_ins].resize(ins->count());
    ins->m_n.add(0); // position de end
    ins->m_block_size_row.add(0);
    ins->m_block_size_col.add(0);
    if (m_trace)
      m_trace->info() << "Inserter id=" << ins->getId() << " count=" << ins->count();
    for (Integer i = 0; i < ins->count(); ++i) {
      const Integer row = ins->m_row_index[i] - m_local_offset;
      const Integer col = ins->m_col_index[i];
      if (row == m_local_size) {
        // case of ghost row
        // save -1, latter will be turn to the null position of the CSR structure
        ins->m_data_index[i] = -1;
        m_block_data_index[nb_ins][i] = -1;
        // if(m_trace) m_trace->info()<<"Ghost Row : "<<i<<" "<<ins->m_row_index[i];
      }
      else {
#ifdef USE_VMAP
        std::pair<RowCols::iterator, bool> finder = row_cols[row].insert(col);
        std::pair<RowCols::iterator, bool> block_finder = block_row_cols[row].insert(col);
#else
        std::pair<RowCols::iterator, bool> finder =
        row_cols[row].insert(std::pair<Integer, Integer>(col, 0));
#endif
        RowCols::iterator inner_iter = finder.first;
        RowCols::iterator block_iter = block_finder.first;
        if (finder.second) {
          const Integer k = m_row_size[row];
          const Integer gk = m_ghost_row_size[row];
          const Integer block_k = m_block_row_size[row];
          const Integer block_gk = m_block_ghost_row_size[row];
          if (isLocal(col)) {
            VALUE_OF(inner_iter) = k - gk;
            VALUE_OF(block_iter) = block_k - block_gk;
            ins->m_data_index[i] = k - gk;
            m_block_data_index[nb_ins][i] = block_k - block_gk;
          }
          else {
            // be careful, position will increment with ghost_offset after
            VALUE_OF(inner_iter) = gk;
            VALUE_OF(block_iter) = block_gk;
            ins->m_data_index[i] = gk;
            m_block_data_index[nb_ins][i] = block_gk;
            ++m_ghost_row_size[row];
            m_block_ghost_row_size[row] += block_sizes->size(ins->m_row_index[i]) * block_sizes->size(ins->m_col_index[i]);
          }
          ++m_row_size[row];
          m_block_row_size[row] += block_sizes->size(ins->m_row_index[i]) * block_sizes->size(ins->m_col_index[i]);
        }
        else {
          ins->m_data_index[i] = VALUE_OF(inner_iter);
          m_block_data_index[nb_ins][i] = VALUE_OF(block_iter);
        }
      }
    }
    nb_ins++;
  }

  ArrayView<Integer> m_row_offsets =
  m_matrix_impl->internal().getCSRProfile().getRowOffset();
  ArrayView<Integer> m_block_row_offsets =
  m_matrix_impl->internal().getCSRProfile().getBlockRowOffset();
  m_matrix_size = 0;
  m_block_matrix_size = 0;
  for (Integer row = 0; row < m_local_size; ++row) {
    m_row_offsets[row] = m_matrix_size;
    m_matrix_size += m_row_size[row];
    m_block_row_offsets[row] = m_block_matrix_size;
    m_block_matrix_size += m_block_row_size[row];
  }
  m_row_offsets[m_local_size] = m_matrix_size;
  m_block_row_offsets[m_local_size] = m_block_matrix_size;

  profile.allocate();
  ArrayView<Integer> m_cols = profile.getCols();

  Integer icount = 0;
  Integer offset = 0;
  for (Integer row = 0; row < m_local_size; ++row) {
    const Integer ghost_offset = m_row_size[row] - m_ghost_row_size[row];
    for (RowCols::iterator iter = row_cols[row].begin(); iter != row_cols[row].end();
         ++iter) {
      const Integer col_uid = KEY_OF(iter);
      // increment position for ghost cols
      if (!isLocal(col_uid))
        iter.value() += ghost_offset;
      m_cols[offset + VALUE_OF(iter)] = col_uid;
      ++icount;
    }
    offset += m_row_size[row];
  }

  // ALIEN_ASSERT((icount==m_matrix_size),("matrix total size problem"));

  ArrayView<Integer> m_block_cols = profile.getBlockCols();

  Integer previous_block_size = 0, index = 0;
  for (Integer irow = 0; irow < m_local_size; ++irow) { // Attention, c'est local !!!!
    for (Integer j = m_row_offsets[irow]; j < m_row_offsets[irow + 1]; ++j) {
      const Integer col = m_cols[j];
      if (index == 0) {
        m_block_cols[index] = 0;
      }
      else {
        m_block_cols[index] = m_block_cols[index - 1] + previous_block_size;
      }
      previous_block_size =
      block_sizes->size(col) * block_sizes->size(irow + m_local_offset);
      index++;
    }
  }

  m_matrix_impl->allocate();
  m_matrix_impl->internal().getValues().fill(0.);

  nb_ins = 0;
  for (auto iter = m_inserters.begin(); iter != m_inserters.end(); ++iter) {
    BaseInserter* ins = *iter;
    for (Integer i = 0; i < ins->count(); ++i) {
      const Integer row = ins->m_row_index[i] - m_local_offset;
      if (row == m_local_size) {
        // Ghost row
        ins->m_data_index[i] = m_block_matrix_size; // equivalent to the null position
      }
      else {
        const Integer col = ins->m_col_index[i];
        const Integer ghost_offset =
        (isLocal(col) ? 0 : m_block_row_size[row] - m_block_ghost_row_size[row]);
        ins->m_data_index[i] =
        m_block_data_index[nb_ins][i] + m_block_row_offsets[row] + ghost_offset;
      }
    }
    nb_ins++;
    ins->setMatrixValues(m_matrix_impl->internal().getDataPtr());
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
  m_block_row_size.resize(0);
  m_block_ghost_row_size.resize(0);
}

/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
