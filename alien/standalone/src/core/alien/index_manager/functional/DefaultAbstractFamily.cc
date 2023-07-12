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

#include "DefaultAbstractFamily.h"

#include <algorithm>

#include <alien/utils/Precomp.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

DefaultAbstractFamily::DefaultAbstractFamily(const DefaultAbstractFamily& family)
: m_parallel_mng(family.m_parallel_mng)
, m_unique_ids(family.m_unique_ids)
, m_owners(family.m_owners)
{}

/*---------------------------------------------------------------------------*/

DefaultAbstractFamily::DefaultAbstractFamily(ConstArrayView<Int64> uniqueIds,
                                             ConstArrayView<Integer> owners,
                                             IMessagePassingMng* parallel_mng)
: m_parallel_mng(parallel_mng)
{
  copy(m_unique_ids, uniqueIds);
  copy(m_owners, owners);
}

/*---------------------------------------------------------------------------*/

DefaultAbstractFamily::DefaultAbstractFamily(ConstArrayView<Int64> uniqueIds,
                                             IMessagePassingMng* parallel_mng)
: m_parallel_mng(parallel_mng)
{
  const Integer commSize = m_parallel_mng->commSize();
  const Integer commRank = m_parallel_mng->commRank();
  const Integer localSize = uniqueIds.size();
  UniqueArray<Integer> sizes(commSize);
  Arccore::MessagePassing::mpAllGather(
  m_parallel_mng, ConstArrayView<Integer>(1, &localSize), sizes);
  UniqueArray<Integer> starts(commSize + 1);
  starts[0] = 0;
  for (Integer i = 0; i < commSize; ++i)
    starts[i + 1] = starts[i] + sizes[i];

  UniqueArray<Int64> allUniqueIds;
  Arccore::MessagePassing::mpAllGatherVariable(m_parallel_mng, uniqueIds, allUniqueIds);
  m_unique_ids.reserve(allUniqueIds.size());
  m_owners.reserve(allUniqueIds.size());

  // remise en tête des uids locaux associé à commRank
  addRange(m_unique_ids, subConstView(allUniqueIds, starts[commRank], sizes[commRank]));
  addRange(m_owners, commRank, sizes[commRank]);
  for (Integer iRank = 0; iRank < commSize; ++iRank) {
    if (iRank != commRank) {
      addRange(m_unique_ids, subConstView(allUniqueIds, starts[iRank], sizes[iRank]));
      addRange(m_owners, iRank, sizes[iRank]);
    }
  }

#ifndef NDEBUG
  ALIEN_ASSERT((m_unique_ids.size() == allUniqueIds.size()), ("Inconsistant sizes"));
  ALIEN_ASSERT((m_owners.size() == allUniqueIds.size()), ("Inconsistant sizes"));
  for (Integer i = 0; i < localSize; ++i) {
    ALIEN_ASSERT((m_unique_ids[i] == uniqueIds[i]), ("Bad local numbering"));
    ALIEN_ASSERT((m_owners[i] == commRank), ("Bad local owner"));
  }

  // Check duplicated uids
  std::sort(allUniqueIds.begin(), allUniqueIds.end());
  for (Integer i = 1; i < allUniqueIds.size(); ++i)
    ALIEN_ASSERT((allUniqueIds[i - 1] != allUniqueIds[i]), ("Duplicated uid"));
#endif /* NDEBUG */
}

/*---------------------------------------------------------------------------*/

void DefaultAbstractFamily::uniqueIdToLocalId(ArrayView<Int32> localIds,
                                              ConstArrayView<Int64> uniqueIds) const
{
  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    Integer localId = -1;
    for (Integer j = 0; j < m_unique_ids.size(); ++j)
      if (uniqueIds[i] == m_unique_ids[j]) {
        localId = j;
        break;
      }
    if (localId == -1)
      throw Alien::FatalErrorException(A_FUNCINFO, "UniqueId not found");
    localIds[i] = localId;
  }
}

/*---------------------------------------------------------------------------*/

IAbstractFamily::Item
DefaultAbstractFamily::item(Int32 localId) const
{
  return IAbstractFamily::Item(m_unique_ids[localId], m_owners[localId]);
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Integer>
DefaultAbstractFamily::owners(ConstArrayView<Int32> localIds) const
{
  const Integer size = localIds.size();
  SharedArray<Integer> result(size);
  for (Integer i = 0; i < size; ++i) {
    result[i] = m_owners[localIds[i]];
  }
  return SafeConstArrayView<Integer>(result);
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Int64>
DefaultAbstractFamily::uids(ConstArrayView<Int32> localIds) const
{
  const Integer size = localIds.size();
  SharedArray<Int64> result(size);
  for (Integer i = 0; i < size; ++i) {
    result[i] = m_unique_ids[localIds[i]];
  }
  return SafeConstArrayView<Int64>(result);
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Int32>
DefaultAbstractFamily::allLocalIds() const
{
  SharedArray<Int32> local_ids(m_unique_ids.size());
  for (Integer i = 0; i < m_unique_ids.size(); ++i)
    local_ids[i] = i;
  return SafeConstArrayView<Int32>(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
