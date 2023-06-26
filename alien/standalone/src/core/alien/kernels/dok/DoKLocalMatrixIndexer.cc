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

/*
 * DoKLocalMatrixIndexer.cpp
 *
 *  Created on: 25 juil. 2016
 *      Author: chevalic
 */

#include "DoKLocalMatrixIndexer.h"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "DoKReverseIndexer.h"

namespace Alien
{
using namespace Arccore;

void DoKLocalMatrixIndexer::associate(Integer i, Integer j, Offset offset)
{
  m_data[Key(i, j)] = offset;
}

std::optional<ILocalMatrixIndexer::Offset>
DoKLocalMatrixIndexer::find(Integer i, Integer j)
{
  try {
    return m_data.at(DoKLocalMatrixIndexer::Key(i, j));
  }
  catch (std::out_of_range&) {
    return std::nullopt;
  }
}

DoKLocalMatrixIndexer::Offset
DoKLocalMatrixIndexer::create(
Integer i, Integer j, DoKLocalMatrixIndexer::Offset& tentative_offset)
{
  auto to_insert = std::make_pair<Key, Offset>(Key(i, j), tentative_offset++);
  auto [index, is_inserted] = m_data.insert(to_insert);
  if (!is_inserted) {
    tentative_offset--;
  }
  return index->second;
}

ILocalMatrixIndexer*
DoKLocalMatrixIndexer::clone() const
{
  return new DoKLocalMatrixIndexer(*this);
}

namespace
{
  template <class Map>
  class KeyCompare
  {
   public:
    typedef typename Map::iterator Iterator;
    typedef typename Map::key_type Key;

   public:
    bool operator()(const Iterator& a, const Iterator& b)
    {
      Key vala = a->first;
      Key valb = b->first;
      return ((vala.first < valb.first) || ((vala.first == valb.first) && (vala.second < valb.second)));
    }
  };
} // namespace

IReverseIndexer*
DoKLocalMatrixIndexer::sort(Arccore::Array<DoKLocalMatrixIndexer::Renumbering>& perm)
{
  std::vector<HashTable::iterator> src(m_data.size());

  auto curr = src.begin();
  for (auto iter = m_data.begin(); iter != m_data.end(); ++iter, ++curr) {
    *curr = iter;
  }

  KeyCompare<HashTable> compare;
  std::sort(src.begin(), src.end(), compare);

  auto* indexer = new DoKReverseIndexer();
  auto size = static_cast<Arccore::Integer>(m_data.size());
  perm.resize(size);
  for (auto curs = 0; curs < size; ++curs) {
    perm[curs] = Renumbering(src[curs]->second, curs);
    indexer->record(curs, src[curs]->first);
    m_data[Key(src[curs]->first.first, src[curs]->first.second)] = curs;
  }

  return indexer;
}

} // namespace Alien
