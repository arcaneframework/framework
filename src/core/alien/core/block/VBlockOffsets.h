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
 * \file VBlockOffsets.h
 * \brief VBlockOffsets.h
 */

#pragma once

#include <alien/distribution/MatrixDistribution.h>

#include <alien/core/block/VBlock.h>
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
 * \brief Compute block offsets for variable block elements
 */
class ALIEN_EXPORT VBlockImpl final
{
 public:
  /*!
   * \brief Constructor for vectors variable blocks
   * \param[in] block The variable blocks
   * \param[in] dist The vector distribution
   */
  VBlockImpl(const VBlock& block, const VectorDistribution& dist);

  /*!
   * \brief Constructor for matrices variable blocks
   * \param[in] block The variable blocks
   * \param[in] dist The matrix distribution
   */
  VBlockImpl(const VBlock& block, const MatrixDistribution& dist);

  //! Free resources
  ~VBlockImpl() {}

  /*!
   * \brief Copy constructor
   * \param[in] block The variable block offset computation tool
   */
  VBlockImpl(const VBlockImpl& block);

 private:
  VBlockImpl(VBlockImpl&& block) = delete;
  VBlockImpl& operator=(const VBlockImpl& block) = delete;

 public:
  /*!
   * \brief Get the block size from a local index
   * \param[in] index The local index
   * \returns The block size for the index
   */
  Arccore::Integer sizeFromLocalIndex(Arccore::Integer index) const;

  /*!
   * \brief Get the offset from a local index
   * \param[in] index The local index
   * \returns The offset for the index
   */
  Arccore::Integer offsetFromLocalIndex(Arccore::Integer index) const;

  /*!
   * \brief Get the offset from a global or local index
   * \param[in] index The global or local index
   * \returns The offset for the index
   */
  Arccore::Integer offset(Arccore::Integer index) const;

  /*!
   * \brief Get the block sizes for all local blocks
   * \returns The array with all local block sizes
   */
  Arccore::ConstArrayView<Arccore::Integer> sizeOfLocalIndex() const;

  /*!
   * \brief Get the offsets for all local blocks
   * \returns The array with all offsets for all local blocks
   */
  Arccore::ConstArrayView<Arccore::Integer> offsetOfLocalIndex() const;

  /*!
   * \brief Copy this object
   * \returns A copy of this object
   */
  std::shared_ptr<VBlockImpl> clone() const;

 private:
  /*!
   * \brief Compute offsets for variable block size elements
   * \param[in] local_size Local size of the algebraic element
   * \param[in] local_offset Local offset
   */
  void compute(Arccore::Integer local_size, Arccore::Integer local_offset);

  struct Internal;
  //! Actual implementation of the variable block offsets computation tool
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
