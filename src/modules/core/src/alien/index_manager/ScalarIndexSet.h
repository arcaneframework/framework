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

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class IAbstractFamily;
class IndexManager;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*! Index handler.
 *
 * This class is just a proxy, so its copy is cheap and its implementation
 * can vary.
 */
class ALIEN_EXPORT ScalarIndexSet
{
 public:
  ScalarIndexSet();

  ScalarIndexSet(const ScalarIndexSet& en);

  ScalarIndexSet(ScalarIndexSet&& en) noexcept;

  ScalarIndexSet(const String& name, Integer creationIndex, const IndexManager* manager,
                 Integer kind);

  ScalarIndexSet& operator=(const ScalarIndexSet& en);

  ScalarIndexSet& operator=(ScalarIndexSet&& en) noexcept;

  bool operator==(const ScalarIndexSet& en) const;

  //! Indices of owned, for this entry.
  ConstArrayView<Integer> getOwnIndexes() const;

  //! Indices of owned then ghosts, for this entry.
  ConstArrayView<Integer> getAllIndexes() const;

  ConstArrayView<Integer> getOwnLocalIds() const;

  ConstArrayView<Integer> getAllLocalIds() const;

  String getName() const;

  //! Item where the entry is defined.
  Integer getKind() const;

  //! Creation label
  Integer getUid() const;

  const IAbstractFamily& getFamily() const;

  //! Associated index manager.
  const IndexManager* manager() const;

 private:
  struct Internal;
  std::shared_ptr<Internal> m_internal;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
