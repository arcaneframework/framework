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

/*!
 * \file ComputeBlockOffsets.h
 * \brief ComputeBlockOffsets.h
 */

#ifndef ALIEN_CORE_BLOCK_COMPUTEBLOCKOFFSETS_H
#define ALIEN_CORE_BLOCK_COMPUTEBLOCKOFFSETS_H

#include <vector>

#include <alien/core/block/Block.h>
#include <alien/utils/Precomp.h>

#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

namespace IFPEN
{
  class BlockVector;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compute block offsets for an uniform block vector
 * \tparam T The type of the block offsets
 * \param[in] dist The vector distribution
 * \param[in] block The current block
 * \param[in,out] offsets The offset vector
 */
template <typename T = std::vector<std::size_t>>
void computeBlockOffsets(const VectorDistribution& dist, const Block& block, T& offsets)
{
  IMessagePassingMng* parallel_mng = dist.parallelMng();
  const Integer block_size = block.size();
  if (parallel_mng && parallel_mng->commSize() > 1) {
    const Integer nproc = parallel_mng->commSize();
    offsets.resize(nproc + 1);
    for (Integer i = 0; i < nproc; ++i)
      offsets[i] = dist.offset(i) * block_size;
    offsets[nproc] = dist.globalSize() * block_size;
  }
  else {
    offsets.resize(2);
    offsets[0] = 0;
    offsets[1] = dist.globalSize() * block_size;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Compute block offsets for an uniform block matrix
 * \tparam T The type of the block offsets
 * \param[in] dist The matrix distribution
 * \param[in] block The current block
 * \param[in,out] offsets The offset vector
 */
template <typename T = std::vector<std::size_t>>
void computeBlockOffsets(const MatrixDistribution& dist, const Block& block, T& offsets)
{
  IMessagePassingMng* parallel_mng = dist.parallelMng();
  const Integer block_size = block.size();
  if (parallel_mng && parallel_mng->commSize() > 1) {
    const Integer nproc = parallel_mng->commSize();
    offsets.resize(nproc + 1);
    for (Integer i = 0; i < nproc; ++i)
      offsets[i] = dist.rowOffset(i) * block_size;
    offsets[nproc] = dist.globalRowSize() * block_size;
  }
  else {
    offsets.resize(2);
    offsets[0] = 0;
    offsets[1] = dist.globalRowSize() * block_size;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif /* ALIEN_CORE_BLOCK_COMPUTEBLOCKOFFSETS_H */
