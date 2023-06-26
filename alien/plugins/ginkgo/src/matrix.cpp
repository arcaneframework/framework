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

#include "matrix.h"

#include <alien/core/impl/MultiMatrixImpl.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>
namespace Alien::Ginkgo
{

Matrix::Matrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::ginkgo>::name())
, gko::matrix::Csr<double, int>(
  Ginkgo_executor::exec_map.at(Ginkgo_executor::target_machine)(), // "at" throws if not valid
  gko::dim<2>(multi_impl->rowSpace().size(), multi_impl->colSpace().size()))
, data(gko::dim<2>{ (multi_impl->rowSpace().size(), multi_impl->colSpace().size()) })
{
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();

  // Checks that the matrix is square
  if (row_space.size() != col_space.size())
    throw Arccore::FatalErrorException("Matrix must be square");
}

Matrix::~Matrix() {}

void Matrix::assemble()
{
  if ((this->rowSpace().size() == data.get_size()[0]) && (this->colSpace().size() == data.get_size()[1])) {
    this->read(data);
  }
  else
    throw Arccore::FatalErrorException("Matrix size does not match data size");
}

void Matrix::setRowValues(int row, Arccore::ConstArrayView<int> cols, Arccore::ConstArrayView<double> values)
{
  auto ncols = cols.size();
  if (ncols != values.size()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "sizes are not equal");
  }

  for (auto icol = 0; icol < ncols; ++icol) {
    data.add_value(row, cols[icol], values[icol]);
  }
}

} // namespace Alien::Ginkgo
