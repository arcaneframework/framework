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
#include <alien/kernels/dok/ILocalMatrixIndexer.h>
#include <alien/utils/Precomp.h>
#include <unordered_map>

namespace Alien {

class IReverseIndexer;

class ALIEN_EXPORT DoKLocalMatrixIndexer : public ILocalMatrixIndexer
{
 public:
  DoKLocalMatrixIndexer() {}
  virtual ~DoKLocalMatrixIndexer() {}

  DoKLocalMatrixIndexer(const DoKLocalMatrixIndexer& src) = default;
#ifndef WIN32
  DoKLocalMatrixIndexer(DoKLocalMatrixIndexer&& src) = default;
#endif

  DoKLocalMatrixIndexer& operator=(const DoKLocalMatrixIndexer& src) = default;
#ifndef WIN32
  DoKLocalMatrixIndexer& operator=(DoKLocalMatrixIndexer&& src) = default;
#endif

  void associate(Integer i, Integer j, Offset offset) override;
  Offset find(Integer i, Integer j) override;
  Offset create(Integer i, Integer j, Offset& tentative_offset) override;

  IReverseIndexer* sort(ArrayView<Renumbering> perm) override;

  ILocalMatrixIndexer* clone() const override;

 private:
  class HashKey
  {
   public:
    HashKey() {}
    ~HashKey() {}

    size_t operator()(const Key& k) const
    {
      size_t seed = 42;
      std::hash<Integer> h;
      seed ^= h(k.first) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      seed ^= h(k.second) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      return seed;
    }
  };
  typedef std::unordered_map<Key, Offset, HashKey> HashTable;
  HashTable m_data;
};

} // namespace Alien
