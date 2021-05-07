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

#include "ProfiledFixedBlockMatrixBuilder.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/utils/ArrayUtils.h>

//#define CHECKPROFILE_ON_FILLING

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Common
{

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  ProfiledFixedBlockMatrixBuilder::ProfiledFixedBlockMatrixBuilder(
  IMatrix& matrix, const ResetFlag reset_values)
  : m_matrix(matrix)
  , m_finalized(false)
  {
    m_matrix.impl()->lock();
    m_matrix_impl = &m_matrix.impl()->get<BackEnd::tag::simplecsr>(true);

    const ISpace& space = m_matrix.rowSpace();
    // if (space != m_matrix.colSpace())
    //  throw FatalErrorException(
    //      "profiled matrix builder must be used with square matrix");

    const MatrixDistribution& dist = m_matrix_impl->distribution();

    const auto* block = m_matrix_impl->block();

    if (block == nullptr) {
      throw FatalErrorException("matrix must be block");
    }

    m_block_size = block->size();
    m_local_size = dist.localRowSize();
    m_local_offset = dist.rowOffset();
    m_next_offset = m_local_offset + m_local_size;

    SimpleCSRInternal::CSRStructInfo const& profile =
    m_matrix_impl->internal().getCSRProfile();
    m_row_starts = profile.getRowOffset();
    m_cols = profile.getCols();
    m_local_row_size = m_matrix_impl->getDistStructInfo().m_local_row_size;
    m_values = m_matrix_impl->internal().getValues();

    if (reset_values == ProfiledBlockMatrixBuilderOptions::eResetValues)
      m_values.fill(0.);

    if (profile.getColOrdering() != SimpleCSRInternal::CSRStructInfo::eFull)
      throw FatalErrorException(
      A_FUNCINFO, "Cannot build system without full column ordering");
  }

  /*---------------------------------------------------------------------------*/

  ProfiledFixedBlockMatrixBuilder::~ProfiledFixedBlockMatrixBuilder()
  {
    if (!m_finalized)
      finalize();
    m_matrix.impl()->unlock();
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledFixedBlockMatrixBuilder::addData(
  const Integer iIndex, const Integer jIndex, const ConstArray2View<Real> value)
  {
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    ALIEN_ASSERT((value.dim1Size() == m_block_size), ("Block size error"));
    ALIEN_ASSERT((value.dim2Size() == m_block_size), ("Block size error"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException(A_FUNCINFO, "Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer row_size = m_row_starts[local_row + 1] - row_start;
    const Integer ncols = m_local_row_size[local_row];
    Integer col = -1;
    if (isLocal(jIndex))
      col = ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
    else {
      const Integer ngcols = row_size - ncols;
      col = ncols + ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start + ncols, ngcols));
    }
#ifdef CHECKPROFILE_ON_FILLING
    if (col == -1)
      throw FatalErrorException(A_FUNCINFO, "Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer brow_start = row_start * m_block_size * m_block_size;
    const Integer bcol = col * m_block_size * m_block_size;
    Array2View<Real> block(&m_values[brow_start + bcol], m_block_size, m_block_size);
    for (Integer i = 0; i < m_block_size; ++i)
      for (Integer j = 0; j < m_block_size; ++j)
        block[i][j] += value[i][j];
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledFixedBlockMatrixBuilder::addData(const Integer iIndex,
                                                const Integer jIndex, const Real factor, const ConstArray2View<Real> value)
  {
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    ALIEN_ASSERT((value.dim1Size() == m_block_size), ("Block size error"));
    ALIEN_ASSERT((value.dim2Size() == m_block_size), ("Block size error"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException(A_FUNCINFO, "Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer row_size = m_row_starts[local_row + 1] - row_start;
    const Integer ncols = m_local_row_size[local_row];
    Integer col = -1;
    if (isLocal(jIndex))
      col = ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
    else {
      const Integer ngcols = row_size - ncols;
      col = ncols + ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start + ncols, ngcols));
    }
#ifdef CHECKPROFILE_ON_FILLING
    if (col == -1)
      throw FatalErrorException(A_FUNCINFO, "Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer brow_start = row_start * m_block_size * m_block_size;
    const Integer bcol = col * m_block_size * m_block_size;
    Array2View<Real> block(&m_values[brow_start + bcol], m_block_size, m_block_size);
    for (Integer i = 0; i < m_block_size; ++i)
      for (Integer j = 0; j < m_block_size; ++j)
        block[i][j] += factor * value[i][j];
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledFixedBlockMatrixBuilder::setData(
  const Integer iIndex, const Integer jIndex, ConstArray2View<Real> value)
  {
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    ALIEN_ASSERT((value.dim1Size() == m_block_size), ("Block size error"));
    ALIEN_ASSERT((value.dim2Size() == m_block_size), ("Block size error"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException(A_FUNCINFO, "Cannot set data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer row_size = m_row_starts[local_row + 1] - row_start;
    const Integer ncols = m_local_row_size[local_row];
    Integer col = -1;
    if (isLocal(jIndex))
      col = ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
    else {
      const Integer ngcols = row_size - ncols;
      col = ncols + ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start + ncols, ngcols));
    }
#ifdef CHECKPROFILE_ON_FILLING
    if (col == -1)
      throw FatalErrorException(A_FUNCINFO, "Cannot set data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer brow_start = row_start * m_block_size * m_block_size;
    const Integer bcol = col * m_block_size * m_block_size;
    Array2View<Real> block(&m_values[brow_start + bcol], m_block_size, m_block_size);
    for (Integer i = 0; i < m_block_size; ++i)
      for (Integer j = 0; j < m_block_size; ++j)
        block[i][j] = value[i][j];
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledFixedBlockMatrixBuilder::finalize() { m_finalized = true; }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
