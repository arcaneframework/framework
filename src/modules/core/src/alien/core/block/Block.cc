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
 * \file Block.cc
 * \brief Block.cc
 */

#include "Block.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Internal structure for square or rectangular blocks
 */
struct Block::Internal
{
  //! Block size in "X" direction
  Integer m_block_size_x;
  //! Block size in "Y" direction
  Integer m_block_size_y;

  /*!
   * \brief Creates a new block
   * \param[in] sizeX Block size in the "X" direction
   * \param[in] sizeY Block size in the "Y" direction
   * \returns Block of size sizeX * sizeY
   */
  static Block::Internal* newFixedSizeBlock(Integer sizeX, Integer sizeY);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Block::Internal*
Block::Internal::newFixedSizeBlock(Integer sizeX, Integer sizeY)
{
  return new Block::Internal{ sizeX, sizeY };
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Block::Block(const Block& block)
: m_internal(block.m_internal)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Block::Block(Integer block_size)
: m_internal(Internal::newFixedSizeBlock(block_size, block_size))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Block::Block(Integer block_sizeX, Integer block_sizeY)
: m_internal(Internal::newFixedSizeBlock(block_sizeX, block_sizeY))
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
Block::size() const
{
  return m_internal->m_block_size_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
Block::sizeX() const
{
  return m_internal->m_block_size_x;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Integer
Block::sizeY() const
{
  return m_internal->m_block_size_y;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

std::shared_ptr<Block>
Block::clone() const
{
  return std::make_shared<Block>(*this);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
