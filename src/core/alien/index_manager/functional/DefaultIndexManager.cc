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

#include "DefaultIndexManager.h"

#include <list>

#include "DefaultAbstractFamily.h"
#include "alien/utils/Precomp.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultIndexManager::DefaultIndexManager(
IMessagePassingMng* parallel_mng, std::initializer_list<ConstArrayView<Int64>> uids)
: m_index_manager(parallel_mng)
{
  m_index_manager.setVerboseMode(true);

  for (auto& uid : uids) {
    DefaultAbstractFamily family(uid, parallel_mng);
    auto kind = m_index_sets.size();
    auto name = Alien::format("Eq_{0}", kind);
    add(m_index_sets,
        m_index_manager.buildScalarIndexSet(name, family, kind, IndexManager::Clone));
  }

  m_index_manager.prepare();
}
/*---------------------------------------------------------------------------*/

UniqueArray<Integer>
DefaultIndexManager::operator[](Integer label) const
{
  return m_index_manager.getIndexes(m_index_sets[label]);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
