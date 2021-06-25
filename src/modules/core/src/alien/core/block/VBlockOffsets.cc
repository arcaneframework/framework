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
 * \file VBlockOffsets.cc
 * \brief VBlockOffsets.cc
 */

#include "VBlockOffsets.h"

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal structure for variable block offset computation tool
 */
struct VBlockImpl::Internal
{
  // SD : clang error here, explicit ctor (VMap) in implicit ctor
  Internal(const VBlock& blocks)
  : m_blocks(blocks)
  {}

  //! Variable blocks information
  const VBlock& m_blocks;
  //! All offsets array
  VBlock::ValuePerBlock m_all_offsets;
  //! Local sizes array
  UniqueArray<Integer> m_local_sizes;
  //! Local sizes offset
  UniqueArray<Integer> m_local_offsets;

  /*!
   * \brief Compute offsets for variable block size algebraic element
   * \param[in] blocks Variable blocks size information
   * \returns Variable block size computed offsets
   */
  static VBlockImpl::Internal* newVariableSizeBlock(const VBlock& blocks);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockImpl::Internal*
VBlockImpl::Internal::newVariableSizeBlock(const VBlock& blocks)
{
  return new VBlockImpl::Internal{ blocks };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockImpl::VBlockImpl(const VBlockImpl& block)
: m_internal(block.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockImpl::VBlockImpl(const VBlock& blocks, const VectorDistribution& dist)
: m_internal(Internal::newVariableSizeBlock(blocks))
{
  const Integer localSize = dist.localSize();
  const Integer localOffset = dist.offset();
  this->compute(localSize, localOffset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlockImpl::VBlockImpl(const VBlock& blocks, const MatrixDistribution& dist)
: m_internal(Internal::newVariableSizeBlock(blocks))
{
  const Integer localSize = dist.localRowSize();
  const Integer localOffset = dist.rowOffset();
  this->compute(localSize, localOffset);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VBlockImpl::compute(Integer localSize, Integer localOffset)
{
  m_internal->m_local_sizes.resize(localSize);
  m_internal->m_local_offsets.resize(localSize + 1);

  Integer sum = 0;
  const VBlock::ValuePerBlock& all_blocks_sizes = m_internal->m_blocks.blockSizes();
  for (Integer i = 0; i < localSize; ++i) {
    const Integer block_size = all_blocks_sizes.find(i + localOffset).value();
    m_internal->m_local_sizes[i] = block_size;
    m_internal->m_local_offsets[i] = sum;
    m_internal->m_all_offsets[i + localOffset] = sum;
    sum += block_size;
  }
  m_internal->m_local_offsets[localSize] = sum;

  sum = 0;
  for (VBlock::ValuePerBlock::const_iterator it = all_blocks_sizes.begin();
       it != all_blocks_sizes.end(); ++it) {
    const Integer globalIndex = it.key();
    const Integer block_size = it.value();
    if (m_internal->m_all_offsets.find(globalIndex) == m_internal->m_all_offsets.end())
      m_internal->m_all_offsets[globalIndex] = sum;
    sum += block_size;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockImpl::sizeFromLocalIndex(Integer index) const
{
  return m_internal->m_local_sizes[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockImpl::offset(Integer index) const
{
  auto it = m_internal->m_all_offsets.find(index);

  if (it == m_internal->m_all_offsets.end())
    throw FatalErrorException(A_FUNCINFO, "index is not registered");

  return it.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlockImpl::offsetFromLocalIndex(Integer index) const
{
  return m_internal->m_local_offsets[index];
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TOCHECK : to remove or not
/*
Integer
VBlockImpl::
localSize() const
{
  return m_internal->m_local_scalarized_size;
}
*/

/*---------------------------------------------------------------------------*/

// TOCHECK : to remove or not
/*
Integer
VBlockImpl::
globalSize() const
{
 return m_internal->m_global_scalarized_size;
}
*/

/*---------------------------------------------------------------------------*/

// TOCHECK : to remove or not
/*
Integer
VBlockImpl::
offset() const
{
return m_internal->m_scalarized_offset;
}
*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
VBlockImpl::sizeOfLocalIndex() const
{
  return m_internal->m_local_sizes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Integer>
VBlockImpl::offsetOfLocalIndex() const
{
  return m_internal->m_local_offsets;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<VBlockImpl>
VBlockImpl::clone() const
{
  return std::make_shared<VBlockImpl>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
