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
 * \file VBlockSizes.h
 * \brief VBlockSizes.h
 */

#pragma once

#include <alien/core/block/VBlock.h>
#include <alien/distribution/MatrixDistribution.h>
#include <alien/distribution/VectorDistribution.h>
#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup block
 * \brief Compute the actual size of variable block size algebraic elements
 *
 * This class will refer to scalarized sizes, which means the actual size of the algebraic
 * elements, as if it was not a block element.
 * For exemple, a vector with two block 2x1 entries has a size of two and a scalarized
 * size of four (2 elements for each block).
 *
 */
class ALIEN_EXPORT VBlockSizes final
{
 public:
  /*!
   * \brief Constructor for vectors variable blocks
   * \param[in] block The variable blocks
   * \param[in] dist The vector distribution
   */
  VBlockSizes(const VBlock& block, const VectorDistribution& dist);

  /*!
   * \brief Constructor for matrices variable blocks
   * \param[in] block The variable blocks
   * \param[in] dist The matrix distribution
   */
  VBlockSizes(const VBlock& block, const MatrixDistribution& dist);

  //! Free resources
  ~VBlockSizes() {}

 private:
  VBlockSizes(VBlockSizes&& block) = delete;
  VBlockSizes& operator=(const VBlockSizes& block) = delete;

 public:
  /*!
   * \brief Get the "scalarized" local size
   * \returns The actual local size
   */
  Arccore::Integer localSize() const;

  /*!
   * \brief Get the "scalarized" global size
   * \returns The actual global size
   */
  Arccore::Integer globalSize() const;

  /*!
   * \brief Get the "scalarized" offset
   * \returns The actual offset
   */
  Arccore::Integer offset() const;

  /*!
   * \brief Copy this object
   * \returns A copy of this object
   */
  // FIXME: not implemented yet !
  std::shared_ptr<VBlockSizes> clone() const;

 private:
  /*!
   * \brief Compute the actual sizes and offset of variable block size algebraic elements
   * \param[in] all_block_size All variable block size elements
   * \param[in] local_size Local size (where one block count as one)
   * \param[in] local_offset Local offset (where one block count as one)
   * \param[in] parallel_mng The parallel manager
   */
  void compute(const VBlock::ValuePerBlock& all_blocks_sizes, Arccore::Integer local_size,
               Arccore::Integer local_offset,
               Arccore::MessagePassing::IMessagePassingMng* parallel_mng);

  //! Scalarized local size
  Arccore::Integer m_local_scalarized_size;
  //! Scalarized global size
  Arccore::Integer m_global_scalarized_size;
  //! Scalarized offset
  Arccore::Integer m_scalarized_offset;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
