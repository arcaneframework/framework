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

#include <algorithm>
#include <unordered_map>

#include <alien/utils/Precomp.h>

#include "AbstractItemFamily.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

AbstractItemFamily::AbstractItemFamily(const AbstractItemFamily& family)
: m_parallel_mng(family.m_parallel_mng)
, m_trace_mng(family.m_trace_mng)
, m_unique_ids(family.m_unique_ids)
, m_owners(family.m_owners)
, m_uid2lid(family.m_uid2lid)
{}

/*---------------------------------------------------------------------------*/
AbstractItemFamily::AbstractItemFamily(ConstArrayView<Int64> uniqueIds,
                                       ConstArrayView<Integer> owners,
                                       IMessagePassingMng* parallel_mng,
                                       ITraceMng* trace_mng)
: m_parallel_mng(parallel_mng)
, m_trace_mng(trace_mng)
{
  copy(m_unique_ids, uniqueIds);
  copy(m_owners, owners);

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }
}

AbstractItemFamily::AbstractItemFamily(ConstArrayView<Int64> uniqueIds,
                                       ConstArrayView<Int64> ghost_uniqueIds,
                                       ConstArrayView<Integer> ghost_owners,
                                       IMessagePassingMng* parallel_mng,
                                       ITraceMng* trace_mng)
: m_parallel_mng(parallel_mng)
, m_trace_mng(trace_mng)
{
  const Integer commRank = m_parallel_mng->commRank();
  const Integer localSize = uniqueIds.size();
  const Integer ghostSize = ghost_uniqueIds.size();

  m_unique_ids.reserve(localSize + ghostSize);
  m_owners.reserve(localSize + ghostSize);

  // remise en tête des uids locaux associé à commRank
  addRange(m_unique_ids, uniqueIds);
  addRange(m_owners, commRank, localSize);
  for (Integer ighost = 0; ighost < ghostSize; ++ighost) {
    addRange(m_unique_ids, ghost_uniqueIds);
    addRange(m_owners, ghost_owners);
  }

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }

  for (Integer i = 0; i < ghost_uniqueIds.size(); ++i) {
    m_uid2lid[ghost_uniqueIds[i]] = localSize + i;
  }
}

/*---------------------------------------------------------------------------*/

AbstractItemFamily::AbstractItemFamily(ConstArrayView<Int64> uniqueIds,
                                       IMessagePassingMng* parallel_mng, [[maybe_unused]] ITraceMng* trace_mng)
{

  m_unique_ids.copy(uniqueIds);
  m_owners.fill(parallel_mng->commRank());

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }
}

/*---------------------------------------------------------------------------*/

void AbstractItemFamily::uniqueIdToLocalId(
ArrayView<Int32> localIds, ConstArrayView<Int64> uniqueIds) const
{

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    auto iter = m_uid2lid.find(uniqueIds[i]);
    if (iter == m_uid2lid.end()) {
      throw Alien::FatalErrorException(A_FUNCINFO, "UniqueId not found");
    }
    else
      localIds[i] = iter->second;
  }
}

/*---------------------------------------------------------------------------*/

IAbstractFamily::Item
AbstractItemFamily::item(Int32 localId) const
{
  return IAbstractFamily::Item(m_unique_ids[localId], m_owners[localId]);
}

/*---------------------------------------------------------------------------*/

SafeConstArrayView<Integer>
AbstractItemFamily::owners(ConstArrayView<Int32> localIds) const
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
AbstractItemFamily::uids(ConstArrayView<Int32> localIds) const
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
AbstractItemFamily::allLocalIds() const
{
  SharedArray<Int32> local_ids(m_unique_ids.size());
  for (Integer i = 0; i < m_unique_ids.size(); ++i)
    local_ids[i] = i;
  return SafeConstArrayView<Int32>(local_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
AbstractFamily::AbstractFamily(const AbstractFamily& family)
: m_parallel_mng(family.m_parallel_mng)
, m_trace_mng(family.m_trace_mng)
, m_unique_ids(family.m_unique_ids)
, m_owners(family.m_owners)
, m_uid2lid(family.m_uid2lid)
{}

/*---------------------------------------------------------------------------*/
AbstractFamily::AbstractFamily(ConstArrayView<Int64> uniqueIds,
                               ConstArrayView<Integer> owners,
                               IMessagePassingMng* parallel_mng,
                               ITraceMng* trace_mng)
: m_parallel_mng(parallel_mng)
, m_trace_mng(trace_mng)
{
  copy(m_unique_ids, uniqueIds);
  copy(m_owners, owners);

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }
}

AbstractFamily::AbstractFamily(ConstArrayView<Int64> uniqueIds,
                               ConstArrayView<Int64> ghost_uniqueIds,
                               ConstArrayView<Integer> ghost_owners,
                               IMessagePassingMng* parallel_mng,
                               ITraceMng* trace_mng)
: m_parallel_mng(parallel_mng)
, m_trace_mng(trace_mng)
{
  const Integer commRank = m_parallel_mng->commRank();
  const Integer localSize = uniqueIds.size();
  const Integer ghostSize = ghost_uniqueIds.size();

  m_unique_ids.reserve(localSize + ghostSize);
  m_owners.reserve(localSize + ghostSize);

  // remise en tête des uids locaux associé à commRank
  addRange(m_unique_ids, uniqueIds);
  addRange(m_owners, commRank, localSize);
  for (Integer ighost = 0; ighost < ghostSize; ++ighost) {
    addRange(m_unique_ids, ghost_uniqueIds);
    addRange(m_owners, ghost_owners);
  }

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }

  for (Integer i = 0; i < ghost_uniqueIds.size(); ++i) {
    m_uid2lid[ghost_uniqueIds[i]] = localSize + i;
  }
}

/*---------------------------------------------------------------------------*/

AbstractFamily::AbstractFamily(ConstArrayView<Int64> uniqueIds,
                               IMessagePassingMng* parallel_mng,
                               ITraceMng* trace_mng)
: m_parallel_mng(parallel_mng)
, m_trace_mng(trace_mng)
{
  copy(m_unique_ids, uniqueIds);
  m_owners.resize(uniqueIds.size());
  m_owners.fill(parallel_mng->commRank());

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    m_uid2lid[uniqueIds[i]] = i;
  }
}
/*---------------------------------------------------------------------------*/

void AbstractFamily::uniqueIdToLocalId(
ArrayView<Int32> localIds, ConstArrayView<Int64> uniqueIds) const
{

  for (Integer i = 0; i < uniqueIds.size(); ++i) {
    auto iter = m_uid2lid.find(uniqueIds[i]);
    if (iter == m_uid2lid.end()) {
      localIds[i] = -1;
      throw Alien::FatalErrorException(A_FUNCINFO, "UniqueId not found");
    }
    else
      localIds[i] = iter->second;
  }
}

/*---------------------------------------------------------------------------*/

IIndexManager::IAbstractFamily::Item
AbstractFamily::item(Int32 localId) const
{
  return IIndexManager::IAbstractFamily::Item(m_unique_ids[localId], m_owners[localId]);
}

/*---------------------------------------------------------------------------*/

Arccore::SharedArray<Arccore::Integer>
AbstractFamily::owners(ConstArrayView<Int32> localIds) const
{
  const Integer size = localIds.size();
  SharedArray<Integer> result(size);
  for (Integer i = 0; i < size; ++i) {
    result[i] = m_owners[localIds[i]];
  }
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::SharedArray<Arccore::Int64>
AbstractFamily::uids(ConstArrayView<Int32> localIds) const
{
  const Integer size = localIds.size();
  SharedArray<Int64> result(size);
  for (Integer i = 0; i < size; ++i) {
    result[i] = m_unique_ids[localIds[i]];
  }
  return result;
}

/*---------------------------------------------------------------------------*/

Arccore::SharedArray<Arccore::Int32>
AbstractFamily::allLocalIds() const
{
  SharedArray<Int32> local_ids(m_unique_ids.size());
  for (Integer i = 0; i < m_unique_ids.size(); ++i)
    local_ids[i] = i;
  return local_ids;
}

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
