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

#include "BaseProfiledMatrixBuilder.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>

#include <alien/utils/ArrayUtils.h>

// #define CHECKPROFILE_ON_FILLING

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

  ProfiledMatrixBuilder::ProfiledMatrixBuilder(
  IMatrix& matrix, const ResetFlag reset_values)
  : m_matrix(matrix)
  , m_finalized(false)
  {
    m_matrix.impl()->lock();
    m_matrix_impl = &m_matrix.impl()->get<BackEnd::tag::simplecsr>(true);

    const MatrixDistribution& dist = m_matrix_impl->distribution();

    m_local_size = dist.localRowSize();
    m_local_offset = dist.rowOffset();
    m_next_offset = m_local_offset + m_local_size;

    SimpleCSRInternal::CSRStructInfo const& profile =
    m_matrix_impl->internal().getCSRProfile();
    m_row_starts = profile.getRowOffset();
    m_local_row_size = m_matrix_impl->getDistStructInfo().m_local_row_size;
    m_cols = profile.getCols();
    m_values = m_matrix_impl->internal().getValues();

    if (reset_values == ProfiledMatrixOptions::eResetValues)
      m_values.fill(0.);

    if (profile.getColOrdering() != SimpleCSRInternal::CSRStructInfo::eFull)
      throw FatalErrorException(
      A_FUNCINFO, "Cannot build system without full column ordering");
  }

  /*---------------------------------------------------------------------------*/

  ProfiledMatrixBuilder::~ProfiledMatrixBuilder()
  {
    if (!m_finalized) {
      finalize();
    }
  }

  /*---------------------------------------------------------------------------*/
  Real ProfiledMatrixBuilder::getData(const Integer iIndex, const Integer jIndex) const
  {
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer ncols = m_local_row_size[local_row];
    if (isLocal(jIndex)) {
      const Integer col =
      ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      return m_values[row_start + col];
    }
    else {
      const Integer ngcols =
      m_row_starts[iIndex - m_local_offset + 1] - row_start - ncols;
      const Integer col = ArrayScan::dichotomicScan(
      jIndex, m_cols.subConstView(row_start + ncols, ngcols));
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      return m_values[row_start + ncols + col];
    }
  }

  void ProfiledMatrixBuilder::addData(
  const Integer iIndex, const Integer jIndex, const Real value)
  {
    _startTimer();
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer ncols = m_local_row_size[local_row];
    if (isLocal(jIndex)) {
      const Integer col =
      ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      m_values[row_start + col] += value;
    }
    else {
      const Integer ngcols =
      m_row_starts[iIndex - m_local_offset + 1] - row_start - ncols;
      const Integer col = ArrayScan::dichotomicScan(
      jIndex, m_cols.subConstView(row_start + ncols, ngcols));
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      m_values[row_start + ncols + col] += value;
    }
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledMatrixBuilder::addData(const Integer iIndex, const Real factor,
                                      ConstArrayView<Integer> jIndexes, ConstArrayView<Real> jValues)
  {
    _startTimer();
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    ALIEN_ASSERT((jIndexes.size() == jValues.size()),
                 ("Inconsistent sizes: %d vs %d", jIndexes.size(), jValues.size()));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot add data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer row_size = m_row_starts[local_row + 1] - row_start;
    const Integer ncols = m_local_row_size[local_row];
    const Integer n = jIndexes.size();
    for (Integer j = 0; j < n; ++j) {

      Integer jIndex = jIndexes[j];
      Integer col = -1;
      if (isLocal(jIndex)) {
        col = ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
      }
      else {
        const Integer ngcols = row_size - ncols;
        col = ncols + ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start + ncols, ngcols));
      }
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot add data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      m_values[row_start + col] += factor * jValues[j];
    }
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledMatrixBuilder::setData(
  const Integer iIndex, const Integer jIndex, const Real value)
  {
    _startTimer();
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot set data on undefined row");
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
      throw FatalErrorException("Cannot set data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
    m_values[row_start + col] = value;
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledMatrixBuilder::setData(const Integer iIndex, const Real factor,
                                      ConstArrayView<Integer> jIndexes, ConstArrayView<Real> jValues)
  {
    _startTimer();
    ALIEN_ASSERT((!m_finalized), ("Finalized matrix cannot be modified"));
    ALIEN_ASSERT((jIndexes.size() == jValues.size()),
                 ("Inconsistent sizes: %d vs %d", jIndexes.size(), jValues.size()));
    const Integer local_row = iIndex - m_local_offset;
#ifdef CHECKPROFILE_ON_FILLING
    if (local_row < 0 or local_row >= m_local_size)
      throw FatalErrorException("Cannot set data on undefined row");
#endif /* CHECKPROFILE_ON_FILLING */
    const Integer row_start = m_row_starts[local_row];
    const Integer row_size = m_row_starts[local_row + 1] - row_start;
    const Integer ncols = m_local_row_size[local_row];
    const Integer n = jIndexes.size();
    for (Integer j = 0; j < n; ++j) {
      Integer jIndex = jIndexes[j];
      Integer col = -1;
      if (isLocal(jIndex))
        col = ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start, ncols));
      else {
        const Integer ngcols = row_size - ncols;
        col = ncols + ArrayScan::dichotomicScan(jIndex, m_cols.subConstView(row_start + ncols, ngcols));
      }
#ifdef CHECKPROFILE_ON_FILLING
      if (col == -1)
        throw FatalErrorException("Cannot set data on undefined column");
#endif /* CHECKPROFILE_ON_FILLING */
      m_values[row_start + col] = factor * jValues[j];
    }
    _stopTimer();
  }

  /*---------------------------------------------------------------------------*/

  void ProfiledMatrixBuilder::finalize()
  {
    if (m_finalized)
      return;
    m_matrix.impl()->unlock();
    m_finalized = true;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Common

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
