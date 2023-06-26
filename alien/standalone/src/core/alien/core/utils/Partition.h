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
 * \file Partition.h
 * \brief Partition.h
 */

#pragma once

#include <arccore/base/String.h>

#include <alien/data/Space.h>
#include <alien/distribution/MatrixDistribution.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Creates tags in the matrix
 *
 * Decorates the matrix by associating tags to entries
 */
class ALIEN_EXPORT Partition
{
 public:
  /*!
   * \brief Constructor
   * \param[in] space The space of the matrix
   * \param[in] distribution The distribution of the matrix
   */
  Partition(const ISpace& space, const MatrixDistribution& distribution);

  //! Free resources
  ~Partition() = default;

  /*!
   * \brief Create the tags of the matrix
   * \param[in] tags The array of tags
   */
  void create(const Arccore::UniqueArray<Arccore::String>& tags);

  /*!
   * \brief Get the number of untagged parts of the matrix
   * \returns The number of untagged parts
   */
  Arccore::Integer nbTaggedParts() const { return m_tagged_parts.size(); }

  /*!
   * \brief Get the i-th tag
   * \param[in] i The requested tag
   * \returns The tag
   */
  Arccore::String tag(Arccore::Integer i) const { return m_tags[i]; }

  /*!
   * \brief Get indices of a specific matching tag
   * \param[in] i The requested tag
   * \returns Local ids corresponding to the tag
   */
  const Arccore::UniqueArray<Arccore::Integer>& taggedPart(Arccore::Integer i) const
  {
    return m_tagged_parts[i];
  }

  /*!
   * \brief Whether or not the matrix has untagged part
   * \returns Whether or not the matrix has untagged part
   */
  bool hasUntaggedPart() const { return !m_untagged_part.empty(); }

  /*!
   * \brief Get untagged indices
   * \returns Local ids of the untagged indices
   */
  const Arccore::UniqueArray<Arccore::Integer>& untaggedPart() const
  {
    return m_untagged_part;
  }

 private:
  //! The matrix space
  const ISpace& m_space;
  //! The matrix distribution
  const MatrixDistribution& m_distribution;
  //! The array of tags
  Arccore::UniqueArray<Arccore::String> m_tags;
  //! The arrays of local ids for each tag
  Arccore::UniqueArray<Arccore::UniqueArray<Arccore::Integer>> m_tagged_parts;
  // The array of local ids for untagged entries
  Arccore::UniqueArray<Arccore::Integer> m_untagged_part;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
