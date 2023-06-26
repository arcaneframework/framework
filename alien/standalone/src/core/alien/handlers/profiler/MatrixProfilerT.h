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

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRInternal.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/utils/ArrayUtils.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  //! Scalar matrix builder
  template <typename ValueT>
  MatrixProfilerT<ValueT>::MatrixProfilerT(IMatrix& matrix)
  : m_matrix(matrix)
  {
    m_matrix_impl = &m_matrix.impl()->template get<BackEnd::tag::simplecsr>(false);

    const ISpace& space = m_matrix_impl->rowSpace();

    if (space != m_matrix_impl->colSpace())
      m_square_matrix = false;

    const MatrixDistribution& dist = m_matrix_impl->distribution();
    m_parallel_mng = dist.parallelMng();

    if (m_parallel_mng == nullptr) {
      m_nproc = 1;
    }
    else {
      m_nproc = m_parallel_mng->commSize();
    }

    m_local_size = dist.localRowSize();
    m_global_size = dist.globalRowSize();
    m_local_offset = dist.rowOffset();
    if (!m_square_matrix) {
      m_col_local_size = dist.localColSize();
      m_col_global_size = dist.globalColSize();
      m_col_local_offset = dist.colOffset();
    }

    m_def_matrix.resize(m_local_size);
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  MatrixProfilerT<ValueT>::~MatrixProfilerT()
  {
    if (!m_allocated) {
      allocate();
    }
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void MatrixProfilerT<ValueT>::addMatrixEntry(Integer iIndex, Integer jIndex)
  {
    const Integer local_row = iIndex - m_local_offset;
    // ALIEN_ASSERT((local_row >= 0 and local_row < m_local_size),("Cannot manage not
    // prepared row"));

    std::vector<Integer>& row_def = m_def_matrix[local_row];
    const Integer row_def_size = static_cast<Integer>(row_def.size());
    if (row_def_size == 0)
      row_def.push_back(jIndex);
    else {
      Integer pos = ArrayScan::dichotomicPositionScan(
      jIndex, ConstArrayView<Integer>(row_def_size, &row_def[0]));
      if (pos >= row_def_size)
        row_def.push_back(jIndex);
      else if (row_def[pos] != jIndex) {
        row_def.insert(row_def.begin() + pos, jIndex);
      }
    }
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void MatrixProfilerT<ValueT>::allocate()
  {
    if (m_allocated)
      return;
    _startTimer();
    computeProfile();
    m_matrix_impl->updateTimestamp();
    m_allocated = true;
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  template <typename ValueT>
  void MatrixProfilerT<ValueT>::computeProfile()
  {
    UniqueArray<Integer> m_offset;
    m_offset.resize(m_nproc + 1);
    if (m_parallel_mng) {
      Arccore::MessagePassing::mpAllGather(m_parallel_mng,
                                           ConstArrayView<Integer>(1, &m_local_offset), m_offset.subView(0, m_nproc));
    }
    m_offset[m_nproc] = m_global_size;

    SimpleCSRInternal::CSRStructInfo& profile = m_matrix_impl->internal().getCSRProfile();
    profile.init(m_local_size);

    ArrayView<Integer> row_offsets = profile.getRowOffset();
    Integer offset = 0;
    for (Integer i = 0; i < m_local_size; ++i) {
      row_offsets[i] = offset;
      offset += static_cast<Integer>(m_def_matrix[i].size());
    }
    row_offsets[m_local_size] = offset;

    profile.allocate();
    ArrayView<Integer> cols = profile.getCols();

    for (Integer i = 0, pos = 0; i < m_local_size; ++i) {
      const VectorDefinition& vdef = m_def_matrix[i];
      for (VectorDefinition::const_iterator iterJ = vdef.begin(); iterJ != vdef.end();
           ++iterJ)
        cols[pos++] = *iterJ;
    }

    if (m_matrix_impl->vblock()) {
      const VBlock* block_sizes = m_matrix_impl->vblock();
      auto& block_row_offset = profile.getBlockRowOffset();
      auto& block_cols = profile.getBlockCols();
      auto kcol = profile.kcol();
      auto cols = profile.cols();
      for (Integer irow = 0; irow < m_local_size; ++irow) {
        block_row_offset[irow] = offset;
        auto row_blk_size = block_sizes->size(m_local_offset + irow);
        for (auto k = kcol[irow]; k < kcol[irow + 1]; ++k) {
          auto jcol = cols[k];
          auto col_blk_size = block_sizes->size(jcol);
          offset += row_blk_size * col_blk_size;
          block_cols[k] = offset;
        }
      }
      block_row_offset[m_local_size] = offset;
      block_cols[kcol[m_local_size]] = offset;
    }

    m_matrix_impl->allocate();
    ArrayView<ValueT> values = m_matrix_impl->internal().getValues();
    values.fill(0);

    if (m_nproc > 1)
      m_matrix_impl->parallelStart(m_offset, m_parallel_mng, true);
    else
      m_matrix_impl->sequentialStart();

    profile.getColOrdering() = SimpleCSRInternal::CSRStructInfo::eFull;

    profile.setTimestamp(m_matrix_impl->timestamp() + 1);
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
