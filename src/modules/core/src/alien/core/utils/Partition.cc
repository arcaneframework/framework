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
 * \file Partition.cc
 * \brief Partition.cc
 */

#include "Partition.h"

#include <arccore/base/FatalErrorException.h>
#include <arccore/base/TraceInfo.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

using namespace Arccore;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Partition::Partition(const ISpace& space, const MatrixDistribution& distribution)
: m_space(space)
, m_distribution(distribution)
{}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void Partition::create(const UniqueArray<String>& tags)
{
  m_tags.clear();
  m_tags.copy(tags);

  m_tagged_parts.clear();
  m_tagged_parts.resize(m_tags.size());

  Integer offset = m_distribution.rowOffset();
  UniqueArray<bool> tagged(m_distribution.localRowSize(), false);
  Integer size = m_tags.size();
  for (Integer i = 0; i < size; ++i) {
    auto tag = m_tags[i];

    // tableau d'indices globaux
    const auto& indices = m_space.field(tag);
    for (auto gid : indices) {
      Integer lid = gid - offset;
      if (tagged[lid])
        throw FatalErrorException(A_FUNCINFO, "index defined in multiple field");
      tagged[lid] = true;
    }
    m_tagged_parts[i] = indices;
  }
  // non taggés
  Integer nb_untagged = 0;
  for (auto i : tagged) {
    if (not i)
      nb_untagged++;
  }
  m_untagged_part.resize(nb_untagged);
  Integer index = 0;
  Integer taggedsize = tagged.size();
  for (Integer i = 0; i < taggedsize; ++i) {
    if (not tagged[i]) {
      m_untagged_part[index] = i + offset;
      index++;
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
