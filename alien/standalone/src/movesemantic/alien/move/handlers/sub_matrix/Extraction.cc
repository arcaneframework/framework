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

#include "Extraction.h"

#include <algorithm>

#include <alien/kernels/simple_csr/CSRStructInfo.h>
#include <alien/kernels/simple_csr/SimpleCSRMatrix.h>
#include <alien/move/handlers/scalar/DirectMatrixBuilder.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien::Move
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MatrixData
SubMatrix::Extract(const IMatrix& matrix, const ExtractionIndices& indices)
{
  if (indices.rowRange() != -1 || indices.colRange() != -1)
    return SubMatrix::extractRange(matrix, indices);
  else
    return SubMatrix::extractIndices(matrix, indices);
}

/*---------------------------------------------------------------------------*/

MatrixData
SubMatrix::extractRange(const IMatrix& matrix, const ExtractionIndices& indices)
{

  const SimpleCSRMatrix<Real>* matrix_impl =
  &matrix.impl()->get<BackEnd::tag::simplecsr>();

  const auto& dist = matrix_impl->distribution();

  const Integer startingRow = indices.rowStart();
  const Integer rowRange = indices.rowRange();
  const Integer startingCol = indices.colStart() != -1 ? indices.colStart() : 0;
  const Integer colRange =
  indices.colRange() != -1 ? indices.colRange() : dist.globalRowSize();

  auto* parallel_mng = dist.parallelMng();

  const Integer offset = dist.rowOffset();
  const Integer matrixStart = offset;
  const Integer matrixEnd = dist.localRowSize() + offset;

  const Integer subMatrixStart = startingRow;
  const Integer subMatrixEnd = startingRow + rowRange;

  const Integer subMatrixLocalSize = std::max(
  std::min(subMatrixEnd, matrixEnd) - std::max(subMatrixStart, matrixStart), 0);
  Space subMatrixRowSpace(rowRange);
  Space subMatrixColSpace(colRange);
  MatrixDistribution subMatrixDistribution(
  rowRange, subMatrixLocalSize, rowRange, parallel_mng);
  MatrixData subMatrix(subMatrixDistribution);
  DirectMatrixBuilder builder(std::move(subMatrix), DirectMatrixOptions::eResetValues,
                              DirectMatrixOptions::eUnSymmetric);
  const Integer nnzMatrix = matrix_impl->internal().getValues().size();
  const Integer nrowsMatrix = matrix_impl->distribution().localRowSize();
  const Integer averageEntriesByRow = nnzMatrix / nrowsMatrix;
  builder.reserve(averageEntriesByRow);
  builder.allocate();

  const SimpleCSRInternal::CSRStructInfo& matrixProfile = matrix_impl->getCSRProfile();
  // UniqueArray<Integer>& rowsOffset = matrixProfile.getRowOffset();
  ConstArrayView<Integer> rowsOffset = matrixProfile.getRowOffset();
  // UniqueArray<Integer>& cols = matrixProfile.getCols();
  ConstArrayView<Integer> cols = matrixProfile.getCols();
  // UniqueArray<Real>& values = matrix_impl->internal().getValues();
  ConstArrayView<Real> values = matrix_impl->internal().getValues();

  ALIEN_ASSERT(
  (startingRow >= 0 && startingRow < matrix_impl->distribution().globalRowSize()),
  ("Error, submatrix and matrix dimensions are incompatibles"));
  ALIEN_ASSERT((startingRow + rowRange <= matrix_impl->distribution().globalRowSize()),
               ("Error, submatrix and matrix dimensions are incompatibles"));

  for (Integer i = 0; i < nrowsMatrix; ++i) {
    const Integer local_row = i;
    const Integer global_row = i + offset;
    if (global_row < startingRow || global_row >= startingRow + rowRange)
      continue;

    for (Integer j = rowsOffset[local_row]; j < rowsOffset[local_row + 1]; ++j) {
      const Integer global_col = cols[j];
      if (global_col < startingCol || global_col >= startingCol + colRange)
        continue;

      builder.setData(indices.toLocalRow(global_row), indices.toLocalCol(global_col),
                      values[global_col]);
    }
  }

  builder.finalize();
  return builder.release();
}

/*---------------------------------------------------------------------------*/

MatrixData
SubMatrix::extractIndices(const IMatrix& matrix ALIEN_UNUSED_PARAM,
                          const ExtractionIndices& indices ALIEN_UNUSED_PARAM)
{
  return MatrixData();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien::Move

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
