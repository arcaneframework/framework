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

#include "trilinos_vector.h"

#include <arccore/message_passing_mpi/MpiMessagePassingMng.h>

namespace Alien::Trilinos
{
Vector::Vector(const MultiVectorImpl* multi_impl)
: IVectorImpl(multi_impl, AlgebraTraits<BackEnd::tag::trilinos>::name())
, vec(nullptr)
, t_comm(new Teuchos::MpiComm<int>(MPI_COMM_WORLD))
{
  // allocate by calling setProfile
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
  const int globalSize = distribution().globalSize();

  setProfile(ilower, iupper, globalSize, localSize);
}

void Vector::setProfile(int ilower, int iupper, int numGlobalElts, int numLocalElts)
{
  //if already exists, dealloc
  if (vec)
    vec.release();

  // map
  Teuchos::RCP<const map_type> map = rcp(new map_type(numGlobalElts, numLocalElts, 0, t_comm));
  vec = rcp(new MV(map, 1, true)); /* map, numvec, init 0)*/
}

void Vector::setValues(Arccore::ConstArrayView<double> values)
{
  auto ncols = values.size();

  // Locally, with Tpetra vector methods
  for (size_t i = 0; i < ncols; i++) {
    vec->replaceLocalValue(i, 0, values[i]); /*lclRow, colIdx, value*/
  }
}

void Vector::getValues(Arccore::ArrayView<double> values) const
{
  // get trilinos data
  auto trilinos_vec = vec->getDataNonConst(0);

  // check sizes
  auto csr_cols = values.size();
  auto trilinos_cols = trilinos_vec.size();
  if (csr_cols != trilinos_cols) {
    throw Arccore::FatalErrorException(A_FUNCINFO, "sizes are not equal");
  }

  // update alien data with trilinos data
  for (size_t i = 0; i < csr_cols; i++) {
    values[i] = trilinos_vec[i];
  }
}

} // namespace Alien::Trilinos