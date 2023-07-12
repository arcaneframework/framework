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
 * \file Parameters.h
 * \brief Parameters.h
 */

#pragma once

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Global indexer
 */
struct GlobalIndexer
{
  /*!
   * \brief Get local id from global id
   * \param[in] id The global id
   * \param[in] offset The offset
   * \returns The local id
   */
  static Arccore::Integer index(Arccore::Integer id, Arccore::Integer offset)
  {
    return id - offset;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Local indexer
 */
struct LocalIndexer
{
  /*!
   * \brief Get local id
   * \param[in] id The local id
   * \param[in] offset The offset
   * \returns The local id
   */
  static Arccore::Integer index(
  Arccore::Integer id, Arccore::Integer offset ALIEN_UNUSED_PARAM)
  {
    return id;
  }
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \ingroup utils
 * \brief Indexer
 * \tparam IndexerParameter The type of the indexer
 */
template <typename IndexerParemeter = GlobalIndexer>
struct Parameters
{
  using Indexer = IndexerParemeter;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
