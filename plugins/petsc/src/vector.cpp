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

#include "vector.h"

#include <alien/petsc/backend.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include "petsc_instance.h"

#include <petscvec.h>

namespace Alien::PETSc
{
Vector::Vector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::petsc>::name())
, m_vec(nullptr)
{
  petsc_init_if_needed();

  auto block_size = 1;
  const auto* block = this->block();
  if (block)
    block_size *= block->size();
  else if (this->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

  const auto localOffset = distribution().offset();
  const auto localSize = distribution().localSize();
  const auto ilower = localOffset * block_size;
  const auto iupper = ilower + localSize * block_size - 1;

  setProfile(ilower, iupper);
}

Vector::~Vector()
{
  if (m_vec)
    VecDestroy(&m_vec);
}

void Vector::setProfile(int ilower, int iupper)
{
  if (m_vec)
    VecDestroy(&m_vec);

  auto* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(distribution().parallelMng());
  m_comm = pm ? (*pm->getMPIComm()) : MPI_COMM_WORLD;

  // -- B Vector --
  auto ierr = VecCreate(m_comm, &m_vec);
  ierr |= VecSetType(m_vec, VECMPI);
  ierr |= VecSetSizes(m_vec, iupper - ilower + 1, PETSC_DECIDE);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Vector Initialisation failed");
  }

  m_rows.resize(iupper - ilower + 1);
  for (int i = 0; i < m_rows.size(); ++i)
    m_rows[i] = ilower + i;
}

void Vector::setValues(Arccore::ConstArrayView<double> values)
{
  auto ierr = VecSetValues(m_vec, m_rows.size(), m_rows.data(), values.data(), INSERT_VALUES);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Vector set values failed");
  }
}

void Vector::getValues(Arccore::ArrayView<double> values) const
{
  auto ierr = VecGetValues(m_vec, m_rows.size(), m_rows.data(), values.data());

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Vector get values failed");
  }
}

void Vector::assemble()
{
  auto ierr = VecAssemblyBegin(m_vec);
  ierr |= VecAssemblyEnd(m_vec);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "PETSc Vector assembling failed");
  }
}
} // namespace Alien::PETSc
