/*
 * Copyright 2020-2021 IFPEN-CEA
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

#include "hypre_vector.h"
#include "hypre_instance.h"

#include <alien/hypre/backend.h>

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

#include <HYPRE.h>
// For hypre_*Alloc
#include <_hypre_utilities.h>

#include <alien/core/impl/MultiVectorImpl.h>

#ifdef HAVE_HYPRE_BIGINT
using HypreId = HYPRE_BigInt;
#else
using HypreId = HYPRE_Int;
#endif

namespace Alien::Hypre
{
Vector::Vector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::hypre>::name())
{
  auto distribution = multi_impl->distribution();
  auto const* pm = dynamic_cast<Arccore::MessagePassing::Mpi::MpiMessagePassingMng*>(distribution.parallelMng());
  m_comm = pm ? (*pm->getMPIComm()) : MPI_COMM_WORLD;

  hypre_init_if_needed(m_comm);
  auto block_size = 1;

  if (const auto* block = this->block(); block)
    block_size *= block->size();
  else if (this->vblock())
    throw Arccore::FatalErrorException(A_FUNCINFO, "Not implemented yet");

  const auto localOffset = distribution.offset();
  const auto localSize = distribution.localSize();
  const auto row_lower = localOffset * block_size;
  const auto row_upper = row_lower + localSize * block_size - 1;

  setProfile(row_lower, row_upper);
}

Vector::~Vector()
{
  if (m_hypre)
    HYPRE_IJVectorDestroy(m_hypre);
}

void Vector::setProfile(int ilower, int iupper)
{
  if (m_hypre)
    HYPRE_IJVectorDestroy(m_hypre);

  // -- B Vector --
  auto ierr = HYPRE_IJVectorCreate(m_comm, ilower, iupper, &m_hypre);
  ierr |= HYPRE_IJVectorSetObjectType(m_hypre, HYPRE_PARCSR);
  ierr |= HYPRE_IJVectorInitialize(m_hypre);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre Initialisation failed");
  }

  m_rows.resize(iupper - ilower + 1);
  for (int i = 0; i < m_rows.size(); ++i)
    m_rows[i] = ilower + i;
}

void Vector::setValues(Arccore::ConstArrayView<double> values)
{
  const HypreId* rows = nullptr;
  const HYPRE_Real* data = nullptr;

#ifdef ALIEN_HYPRE_DEVICE
  HYPRE_MemoryLocation memory_location;
  HYPRE_GetMemoryLocation(&memory_location);
  if (memory_location != HYPRE_MEMORY_HOST) {
    HypreId* d_rows = hypre_CTAlloc(HypreId, m_rows.size(), memory_location);
    HYPRE_Real* d_values = hypre_CTAlloc(HYPRE_Real, values.size(), memory_location);

    hypre_TMemcpy(d_rows, m_rows.data(), HypreId, m_rows.size(), memory_location, HYPRE_MEMORY_HOST);
    hypre_TMemcpy(d_values, values.data(), HYPRE_Real, values.size(), memory_location, HYPRE_MEMORY_HOST);
    rows = d_rows;
    data = d_values;
  }
  else
#endif // ALIEN_HYPRE_DEVICE
  {
    rows = m_rows.data();
    data = values.data();
  }

  auto ierr = HYPRE_IJVectorSetValues(m_hypre, m_rows.size(), rows, data);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre set values failed");
  }

#ifdef ALIEN_HYPRE_DEVICE
  if (memory_location != HYPRE_MEMORY_HOST) {
    hypre_TFree(rows, memory_location);
    hypre_TFree(data, memory_location);
  }
#endif // ALIEN_HYPRE_DEVICE
}

void Vector::getValues(Arccore::ArrayView<double> values) const
{
  const HypreId* rows = nullptr;
  HYPRE_Real* data = nullptr;

#ifdef ALIEN_HYPRE_DEVICE
  HYPRE_MemoryLocation memory_location;
  HYPRE_GetMemoryLocation(&memory_location);
  if (memory_location != HYPRE_MEMORY_HOST) {
    HypreId* d_rows = hypre_CTAlloc(HypreId, m_rows.size(), memory_location);
    HYPRE_Real* d_values = hypre_CTAlloc(HYPRE_Real, values.size(), memory_location);

    hypre_TMemcpy(d_rows, m_rows.data(), HypreId, m_rows.size(), memory_location, HYPRE_MEMORY_HOST);
    rows = d_rows;
    data = d_values;
  }
  else
#endif // ALIEN_HYPRE_DEVICE
  {
    rows = m_rows.data();
    data = values.data();
  }
  auto ierr = HYPRE_IJVectorGetValues(m_hypre, m_rows.size(), rows, data);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre get values failed");
  }

#ifdef ALIEN_HYPRE_DEVICE
  if (memory_location != HYPRE_MEMORY_HOST) {
    hypre_TMemcpy(values.data(), data, HYPRE_Real, values.size(), HYPRE_MEMORY_HOST, memory_location);
    hypre_TFree(rows, memory_location);
    hypre_TFree(data, memory_location);
  }
#endif // ALIEN_HYPRE_DEVICE
}

void Vector::assemble()
{
  auto ierr = HYPRE_IJVectorAssemble(m_hypre);

  if (ierr) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "Hypre assembling failed");
  }
}
} // namespace Alien::Hypre