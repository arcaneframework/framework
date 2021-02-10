#include "alien/arcane_tools/indexManager/SimpleAbstractFamily.h"

#include <algorithm>

#include <arcane/utils/FatalErrorException.h>

#include <arcane/IParallelMng.h>
#include <arccore/message_passing/IMessagePassingMng.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  SimpleAbstractFamily::SimpleAbstractFamily(const Arccore::Int64ConstArrayView uniqueIds,
      const Arccore::IntegerConstArrayView owners, IIndexManager* manager)
  : m_manager(manager)
  , m_unique_ids(uniqueIds)
  , m_owners(owners)
  {
    ;
  }

  /*---------------------------------------------------------------------------*/

  SimpleAbstractFamily::SimpleAbstractFamily(
      const Arccore::Int64ConstArrayView uniqueIds, IIndexManager* manager)
  : m_manager(manager)
  {
    Alien::IMessagePassingMng* parallelMng = m_manager->parallelMng();
    const Arccore::Integer commSize = parallelMng->commSize();
    const Arccore::Integer commRank = parallelMng->commRank();
    const Arccore::Integer localSize = uniqueIds.size();
    Arccore::UniqueArray<Arccore::Integer> sizes(commSize);
    Arccore::MessagePassing::mpAllGather(parallelMng,Arccore::IntegerConstArrayView(1, &localSize), sizes);
    Arccore::UniqueArray<Arccore::Integer> starts(commSize + 1);
    starts[0] = 0;
    for (Arccore::Integer i = 0; i < commSize; ++i)
      starts[i + 1] = starts[i] + sizes[i];

    Arccore::UniqueArray<Arccore::Int64> allUniqueIds;
    Arccore::MessagePassing::mpAllGatherVariable(parallelMng,uniqueIds, allUniqueIds);

    m_unique_ids.reserve(allUniqueIds.size());
    m_owners.reserve(allUniqueIds.size());

    // remise en tête des uids locaux associé à commRank
    m_unique_ids.addRange(allUniqueIds.subView(starts[commRank], sizes[commRank]));
    m_owners.addRange(commRank, sizes[commRank]);
    for (Arccore::Integer iRank = 0; iRank < commSize; ++iRank) {
      if (iRank != commRank) {
        m_unique_ids.addRange(allUniqueIds.subView(starts[iRank], sizes[iRank]));
        m_owners.addRange(iRank, sizes[iRank]);
      }
    }

#ifndef NDEBUG
    ARCANE_ASSERT((m_unique_ids.size() == allUniqueIds.size()), ("Inconsistant sizes"));
    ARCANE_ASSERT((m_owners.size() == allUniqueIds.size()), ("Inconsistant sizes"));
    for (Arccore::Integer i = 0; i < localSize; ++i) {
      ARCANE_ASSERT((m_unique_ids[i] == uniqueIds[i]), ("Bad local numbering"));
      ARCANE_ASSERT((m_owners[i] == commRank), ("Bad local owner"));
    }

    // Check duplicated uids
    std::sort(allUniqueIds.begin(), allUniqueIds.end());
    for (Arccore::Integer i = 1; i < allUniqueIds.size(); ++i)
      ARCANE_ASSERT((allUniqueIds[i - 1] != allUniqueIds[i]), ("Duplicated uid"));
#endif /* NDEBUG */
  }

  /*---------------------------------------------------------------------------*/

  SimpleAbstractFamily::~SimpleAbstractFamily() { m_manager->keepAlive(this); }

  /*---------------------------------------------------------------------------*/

  void SimpleAbstractFamily::uniqueIdToLocalId(
      Arccore::Int32ArrayView localIds, Arccore::Int64ConstArrayView uniqueIds) const
  {
    for (Arccore::Integer i = 0; i < uniqueIds.size(); ++i) {
      Arccore::Integer localId = -1;
      for (Arccore::Integer j = 0; j < m_unique_ids.size(); ++j)
        if (uniqueIds[i] == m_unique_ids[j]) {
          localId = j;
          break;
        }
      if (localId == -1)
        throw Arcane::FatalErrorException(A_FUNCINFO, "UniqueId not found");
      localIds[i] = localId;
    }
  }

  /*---------------------------------------------------------------------------*/

  IIndexManager::IAbstractFamily::Item SimpleAbstractFamily::item(
      Arccore::Int32 localId) const
  {
    return IAbstractFamily::Item(m_unique_ids[localId], m_owners[localId]);
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Integer> SimpleAbstractFamily::owners(
      Arccore::Int32ConstArrayView localIds) const
  {
    const Arccore::Integer size = localIds.size();
    Arccore::SharedArray<Arccore::Integer> result(size);
    for (Arccore::Integer i = 0; i < size; ++i) {
      result[i] = m_owners[localIds[i]];
    }
    return result;
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Int64> SimpleAbstractFamily::uids(
      Arccore::Int32ConstArrayView localIds) const
  {
    const Arccore::Integer size = localIds.size();
    Arccore::SharedArray<Arccore::Int64> result(size);
    for (Arccore::Integer i = 0; i < size; ++i) {
      result[i] = m_unique_ids[localIds[i]];
    }
    return result;
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Int32> SimpleAbstractFamily::allLocalIds() const
  {
    Arccore::SharedArray<Arccore::Int32> local_ids(m_unique_ids.size());
    for (Arccore::Integer i = 0; i < m_unique_ids.size(); ++i)
      local_ids[i] = i;
    return local_ids;
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

} // namespace Alien

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
