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

#include <alien/utils/Precomp.h>

#include <algorithm>
#include <unordered_map>
#include <utility>

namespace Alien {

class IReverseIndexer;

class ILocalMatrixIndexer
{
 public:
  typedef Integer Offset;
  typedef std::pair<Offset, Offset> Renumbering;
  typedef std::pair<Int32, Int32> Key;

 public:
  virtual ~ILocalMatrixIndexer() {}

  virtual void associate(Integer i, Integer j, Offset offset) = 0;
  virtual Offset find(Integer i, Integer j) = 0;
  virtual Offset create(Integer i, Integer j, Offset& tentative_offset) = 0;

  virtual IReverseIndexer* sort(ArrayView<Renumbering> perm) = 0;

  virtual ILocalMatrixIndexer* clone() const = 0;
};

} // namespace Alien
