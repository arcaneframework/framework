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

#pragma once

#include <algorithm>
#include <unordered_map>

#include <alien/utils/Precomp.h>

#include <alien/kernels/dok/ILocalMatrixIndexer.h>

namespace Alien
{

class IReverseIndexer;

//! Local matrix indexer using HashMap
class ALIEN_EXPORT DoKLocalMatrixIndexer : public ILocalMatrixIndexer
{
 public:
  DoKLocalMatrixIndexer() = default;
  virtual ~DoKLocalMatrixIndexer() = default;

  DoKLocalMatrixIndexer(const DoKLocalMatrixIndexer& src) = default;
  DoKLocalMatrixIndexer(DoKLocalMatrixIndexer&& src) = default;

  DoKLocalMatrixIndexer& operator=(const DoKLocalMatrixIndexer& src) = default;
  DoKLocalMatrixIndexer& operator=(DoKLocalMatrixIndexer&& src) = default;

  void associate(Integer i, Integer j, Offset offset) override;
  std::optional<Offset> find(Integer i, Integer j) override;
  Offset create(Integer i, Integer j, Offset& tentative_offset) override;

  IReverseIndexer* sort(Arccore::Array<Renumbering>& perm) override;

  ILocalMatrixIndexer* clone() const override;

 private:
  class HashKey
  {
   public:
    HashKey() = default;
    ~HashKey() = default;

    size_t operator()(const Key& k) const
    {
      size_t seed = 42;
      std::hash<Integer> h;
      // Magic numbers from boost::hash_combine
      seed ^= h(k.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= h(k.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };
  typedef std::unordered_map<Key, Offset, HashKey> HashTable;
  HashTable m_data;
};

} // namespace Alien
