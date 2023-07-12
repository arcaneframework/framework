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

#include <alien/petsc/backend.h>
#include <alien/core/impl/MultiMatrixImpl.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include "petsc_instance.h"

namespace Alien::PETSc
{
Matrix::Matrix(const MultiMatrixImpl* multi_impl)
: IMatrixImpl(multi_impl, AlgebraTraits<BackEnd::tag::petsc>::name())
, m_mat(nullptr)
{
  petsc_init_if_needed();
  const auto& row_space = multi_impl->rowSpace();
  const auto& col_space = multi_impl->colSpace();
  if (row_space.size() != col_space.size())
    throw Arccore::FatalErrorException("Petsc matrix must be square"); // est ce le cas pour petsc ?
}

Matrix::~Matrix()
{
  if (m_mat)
    MatDestroy(&m_mat);
}

void Matrix::setProfile(
int ilower, int iupper, int jlower, int jupper,
[[maybe_unused]] Arccore::ConstArrayView<int> row_sizes)
{
  if (m_mat) {
    MatDestroy(&m_mat);
  }

  auto* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(distribution().parallelMng());
  m_comm = pm ? (*pm->getMPIComm()) : MPI_COMM_WORLD;

  auto ierr = MatCreate(m_comm, &m_mat);
  ierr |= MatSetSizes(m_mat, iupper - ilower + 1, jupper - jlower + 1,
                      PETSC_DETERMINE, PETSC_DETERMINE);
  ierr |= MatSetType(m_mat, MATMPIAIJ);
  // Allocate Matrix of twice the size, as we cannot know the local and the remote columns.
  ierr |= MatMPIAIJSetPreallocation(m_mat, 0, row_sizes.data(), 0, row_sizes.data());
  ierr |= MatAssemblyBegin(m_mat, MAT_FINAL_ASSEMBLY);
  ierr |= MatSetUp(m_mat);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Initialisation failed");
  }
}

void Matrix::assemble()
{
  auto ierr = MatAssemblyEnd(m_mat, MAT_FINAL_ASSEMBLY);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc assembling failed");
  }
}

void Matrix::setRowValues(int row, Arccore::ConstArrayView<int> cols, Arccore::ConstArrayView<double> values)
{
  auto ncols = cols.size();

  if (ncols != values.size()) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "sizes are not equal");
  }

  auto ierr = MatSetValues(m_mat, 1, &row, ncols, cols.data(), values.data(), INSERT_VALUES);

  if (ierr) {
    auto msg = Arccore::String::format("Cannot set PETSc Matrix Values for row {0}", row);
    throw Arccore::FatalErrorException(A_FUNCINFO, msg);
  }
}

} // namespace Alien::PETSc
