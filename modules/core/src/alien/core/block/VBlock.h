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
 * \file VBlock.h
 * \brief VBlock.h
 */

#pragma once

#include <alien/distribution/MatrixDistribution.h>
#include <alien/utils/Precomp.h>
#include <alien/utils/VMap.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup block
 * \brief Variable size block elements for block matrices
 *
 * Defines blocks parameters that may be different for each block
 */
class ALIEN_EXPORT VBlock final
{
 public:
  //! Type of the size of each block
  typedef VMap<Arccore::Integer, Arccore::Integer> ValuePerBlock;

 public:
  /*!
   * \brief Rvalue constructor
   * \param[in] all_blocks_sizes Block size for all blocks
   */
  VBlock(ValuePerBlock&& all_blocks_sizes);

  /*!
   * \brief Ref constructor
   * \param[in] all_blocks_sizes Block size for all blocks
   */
  VBlock(const ValuePerBlock& all_blocks_sizes);

  //! Free resources
  ~VBlock() {}

  /*!
   * \brief Copy constructor
   * \param[in] block The block to copy
   */
  VBlock(const VBlock& block);

 private:
  VBlock(VBlock&& block) = delete;
  VBlock& operator=(const VBlock& block) = delete;

 public:
  /*!
   * \brief Get the size of a block
   * \param[in] index The index of the block the size is requested
   * \returns The size of the block
   */
  Arccore::Integer size(Arccore::Integer index) const;

  /*!
   * \brief Get the max size of all block size
   * \returns The max block size
   */
  Arccore::Integer maxBlockSize() const;

  /*!
   * \brief Get the size of all blocks
   * \returns All blocks size information
   */
  const ValuePerBlock& blockSizes() const;

  /*!
   * \brief Copy this object
   * \returns A copy of this object
   */
  std::shared_ptr<VBlock> clone() const;

 private:
  struct Internal;
  //! Actual implementation of variable blocks size
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
