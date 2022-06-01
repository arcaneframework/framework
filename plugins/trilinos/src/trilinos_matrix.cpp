/*
 * Copyright 2022 IFPEN-CEA
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

#include "trilinos_matrix.h"

#include <alien/core/impl/MultiMatrixImpl.h>
#include <alien/data/ISpace.h>
#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

namespace Alien::Trilinos
{
Matrix::Matrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::trilinos>::name())
, mtx(nullptr)
{
  // Checks that the matrix is square
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();
  if (row_space.size() != col_space.size())
    throw Arccore::FatalErrorException("Matrix must be square");

  // communicator
  using Teuchos::Comm;
  using Teuchos::MpiComm;
  using Teuchos::RCP;
  MPI_Comm yourComm = MPI_COMM_WORLD;
  t_comm = RCP<const Comm<int>>(new MpiComm<int>(yourComm)); // Récupérer le communicateur Arcane ?
}

void Matrix::setProfile(int numLocalRows, int numGlobalRows, const Arccore::UniqueArray<int>& rowSizes)
{

  using Teuchos::RCP;
  using Teuchos::rcp;

  // map
  RCP<const map_type> rowMap = rcp(new map_type(numGlobalRows, numLocalRows, 0, t_comm));

  // matrix
  Teuchos::Array<size_t> entriesPerRow(numLocalRows);
  for (size_t i = 0; i < numLocalRows; i++)
    entriesPerRow[i] = rowSizes[i];

  mtx = rcp(new crs_matrix_type(rowMap, entriesPerRow()));
}

void Matrix::assemble()
{
  mtx->fillComplete();
}

void Matrix::setRowValues(int row, Arccore::ConstArrayView<int> columns, Arccore::ConstArrayView<double> values)
{
  auto ncols = columns.size();

  if (ncols != values.size()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "sizes are not equal");
  }

  Teuchos::Array<SC> vals(ncols);
  Teuchos::Array<GO> cols(ncols);

  for (size_t i = 0; i < ncols; i++) {
    cols[i] = columns[i];
    vals[i] = values[i];
  }

  mtx->insertGlobalValues(row, ncols, values.data(), cols.data()); // insertLocal possible but needs colmap
}

} // namespace Alien::Trilinos