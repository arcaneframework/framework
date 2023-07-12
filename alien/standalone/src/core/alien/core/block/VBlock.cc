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
 * \file VBlock.cc
 * \brief VBlock.cc
 */

#include "VBlock.h"

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
 * \brief Actual implementation of variable blocks size
 */
struct VBlock::Internal
{
  //! Size for all blocks
  ValuePerBlock m_all_sizes;
  //! Maximum size amongst all blocks
  Integer m_max_block_size;

  /*!
   * \brief Creates a new variable block size element
   * \param[in] all_sizes Size information for all blocks
   * \param[in] max_block_size The maximum size amongst all blocks
   * \returns A new variable block size object
   */
  static VBlock::Internal* newVariableSizeBlock(
  ValuePerBlock&& all_sizes, Integer max_block_size);

  /*!
   * \brief Creates a new variable block size element
   * \param[in] all_sizes Size information for all blocks
   * \param[in] max_block_size The maximum size amongst all blocks
   * \returns A new variable block size object
   */
  static VBlock::Internal* newVariableSizeBlock(
  const ValuePerBlock& all_sizes, Integer max_block_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlock::Internal*
VBlock::Internal::newVariableSizeBlock(ValuePerBlock&& all_sizes, Integer max_block_size)
{
  return new VBlock::Internal{ std::move(all_sizes), max_block_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlock::Internal*
VBlock::Internal::newVariableSizeBlock(
const ValuePerBlock& all_sizes, Integer max_block_size)
{
  return new VBlock::Internal{ all_sizes, max_block_size };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlock::VBlock(const VBlock& block)
: m_internal(block.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlock::VBlock(ValuePerBlock&& all_blocks_sizes)
{
  Integer max_block_size = 0;
  for (auto it = all_blocks_sizes.begin(); it != all_blocks_sizes.end(); ++it) {
    const Integer block_size = it.value();
    max_block_size = (max_block_size > block_size) ? max_block_size : block_size;
  }
  m_internal.reset(
  Internal::newVariableSizeBlock(std::move(all_blocks_sizes), max_block_size));

  // TOCHECK: Debug info to remove or not ?
  /*
    if(parallel_mng->commRank()==0)
    {
    std::cout << "Blocks in VBlock:\n";
    std::cout << "m_local_scalarized_size: " << m_internal->m_local_scalarized_size <<
    "\n"; std::cout << "m_global_scalarized_size: " <<
    m_internal->m_global_scalarized_size << "\n"; std::cout << "m_scalarized_offset: " <<
    m_internal->m_scalarized_offset << "\n"; std::cout << "m_max_block_size: " <<
    m_internal->m_max_block_size << "\n"; std::cout << "m_local_sizes.size(): " <<
    m_internal->m_local_sizes.size() << "\n"; std::cout << "m_local_offsets.size(): " <<
    m_internal->m_local_offsets.size() << "\n"; std::cout << "m_all_sizes.size(): " <<
    m_internal->m_all_sizes.size() << "\n"; std::cout << "m_all_offsets.size(): " <<
    m_internal->m_all_offsets.size() << "\n"; std::cout << "m_local_sizes: \n";
    for(Integer i=0;i<m_internal->m_local_sizes.size();++i)
    std::cout << m_internal->m_local_sizes[i] << "\n";
    std::cout << "m_local_offsets: \n";
    for(Integer i=0;i<m_internal->m_local_offsets.size();++i)
    std::cout << m_internal->m_local_offsets[i] << "\n";
    std::cout << "m_all_sizes: \n";
    for(ValuePerBlock::const_iterator it = m_internal->m_all_sizes.begin(); it !=
    m_internal->m_all_sizes.end(); ++it)
    {
    std::cout << "index: " << it.key() << " value: " << it.value() << "\n";
    }
    std::cout << "m_all_offsets: \n";
    for(ValuePerBlock::const_iterator it = m_internal->m_all_offsets.begin(); it !=
    m_internal->m_all_offsets.end(); ++it)
    {
    std::cout << "index: " << it.key() << " value: " << it.value() << "\n";
    }
    }
  */
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VBlock::VBlock(const ValuePerBlock& all_blocks_sizes)
{
  Integer max_block_size = 0;
  for (ValuePerBlock::const_iterator it = all_blocks_sizes.begin();
       it != all_blocks_sizes.end(); ++it) {
    const Integer block_size = it.value();
    max_block_size = (max_block_size > block_size) ? max_block_size : block_size;
  }
  m_internal.reset(
  Internal::newVariableSizeBlock(std::move(all_blocks_sizes), max_block_size));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlock::size(Integer index) const
{
  auto it = m_internal->m_all_sizes.find(index);

  if (it == m_internal->m_all_sizes.end())
    throw FatalErrorException(A_FUNCINFO, "index is not registered");
  return it.value();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
VBlock::maxBlockSize() const
{
  return m_internal->m_max_block_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<VBlock>
VBlock::clone() const
{
  return std::make_shared<VBlock>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

const VBlock::ValuePerBlock&
VBlock::blockSizes() const
{
  return m_internal->m_all_sizes;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
