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
 * \file VBlockSizes.cc
 * \brief VBlockSizes.cc
 */

#include "VBlockSizes.h"

#include <arccore/message_passing/Messages.h>
#include <cstdlib>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;
using namespace Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockSizes::VBlockSizes(const VBlock& blocks, const VectorDistribution& dist)
{
  const Integer localSize = dist.localSize();
  const Integer localOffset = dist.offset();
  this->compute(blocks.blockSizes(), localSize, localOffset, dist.parallelMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockSizes::VBlockSizes(const VBlock& blocks, const MatrixDistribution& dist)
{
  const Integer localSize = dist.localRowSize();
  const Integer localOffset = dist.rowOffset();
  this->compute(blocks.blockSizes(), localSize, localOffset, dist.parallelMng());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VBlockSizes::compute(const VBlock::ValuePerBlock& all_blocks_sizes, Integer localSize,
                          Integer localOffset, IMessagePassingMng* parallel_mng)
{
  m_local_scalarized_size = 0;
  for (Integer i = 0; i < localSize; ++i) {
    const Integer block_size = all_blocks_sizes.find(i + localOffset).value();
    m_local_scalarized_size += block_size;
  }

  m_global_scalarized_size = 0;
  m_scalarized_offset = 0;
  const bool is_parallel = (parallel_mng != NULL) && (parallel_mng->commSize() > 1);
  if (is_parallel) {
    m_global_scalarized_size = Arccore::MessagePassing::mpAllReduce(
    parallel_mng, Arccore::MessagePassing::ReduceSum, m_local_scalarized_size);
    Arccore::UniqueArray<Arccore::Integer> local_sizes(parallel_mng->commSize());
    Arccore::MessagePassing::mpAllGather(parallel_mng,
                                         Alien::ArrayView<Integer>(1, &m_local_scalarized_size), local_sizes);
    for (Integer i = 0; i < parallel_mng->commRank(); ++i)
      m_scalarized_offset += local_sizes[i];
  }
  else {
    m_global_scalarized_size = m_local_scalarized_size;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockSizes::localSize() const
{
  return m_local_scalarized_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockSizes::globalSize() const
{
  return m_global_scalarized_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockSizes::offset() const
{
  return m_scalarized_offset;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
