#include "alien/arcane_tools/indexManager/AnyItemAbstractFamily.h"
#include "alien/utils/ArrayUtils.h"

#include <algorithm>

#include <arcane/utils/FatalErrorException.h>
#include <arcane/IItemFamily.h>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Alien {

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace ArcaneTools {

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  // Inside module tools.
  namespace {

    const Arccore::Integer GROUP_BIT_SIZE = 8;
    const Arccore::Integer GROUP_BIT_OFFSET = 55; // 64b - 1b (sign) - 8b (groups)
    const Arccore::Int64 GROUP_MASK = ((Arccore::Int64(1) << GROUP_BIT_SIZE) - 1)
        << GROUP_BIT_OFFSET;
    const Arccore::Int64 ITEM_UID_MASK = (Arccore::Int64(1) << GROUP_BIT_OFFSET) - 1;

#ifndef _MSC_VER
    constexpr
#endif // _MSC_VER
        Arccore::Int64
        makeUID(Arccore::Integer igrp, Arccore::Int64 item_unique_id)
    {
      return ((Arccore::Int64(igrp) << GROUP_BIT_OFFSET) | item_unique_id);
    }

#ifndef _MSC_VER
    constexpr
#endif // _MSC_VER
        Arccore::Integer
        getIgroup(Arccore::Int64 uid)
    {
      return static_cast<Arccore::Integer>(uid >> GROUP_BIT_OFFSET);
    }

#ifndef _MSC_VER
    constexpr
#endif
        Arccore::Int64
        getItemUID(Arccore::Int64 uid)
    {
      return (uid & ITEM_UID_MASK);
    }
  }

  /*---------------------------------------------------------------------------*/
  /*---------------------------------------------------------------------------*/

  AnyItemAbstractFamily::AnyItemAbstractFamily(
      const Arcane::AnyItem::Family& family, IIndexManager* manager)
  : m_family(family)
  , m_manager(manager)
  {
    Arccore::Int64 max_unique_id = 0;
    m_lower_bounds.resize(m_family.groupSize() + 1);
    for (Arccore::Integer igrp = 0; igrp < m_family.groupSize(); ++igrp) {
      Arcane::ItemGroup group = m_family.group(igrp);
      m_lower_bounds[igrp] = m_family.firstLocalId(group);
      ENUMERATE_ITEM (iitem, group) {
        max_unique_id = std::max(max_unique_id, iitem->uniqueId().asInt64());
      }
    }
    Arccore::Integer max_local_id = m_family.maxLocalId();
    if ((max_local_id & ~ITEM_UID_MASK) != 0)
      throw Arcane::FatalErrorException(
          A_FUNCINFO, "Too large item unique ids in AnyItem::Family");
    m_lower_bounds[m_family.groupSize()] = max_local_id;
    if (m_family.groupSize() > (1 << GROUP_BIT_SIZE))
      throw Arcane::FatalErrorException(A_FUNCINFO, "Too many groups in AnyItem::Family");
  }

  /*---------------------------------------------------------------------------*/

  AnyItemAbstractFamily::~AnyItemAbstractFamily() { m_manager->keepAlive(this); }

  /*---------------------------------------------------------------------------*/

  IIndexManager::IAbstractFamily* AnyItemAbstractFamily::clone() const
  {
    return new AnyItemAbstractFamily(m_family, m_manager);
  }

  /*---------------------------------------------------------------------------*/

  Arccore::Integer AnyItemAbstractFamily::maxLocalId() const
  {
    return m_family.maxLocalId();
  }

  /*---------------------------------------------------------------------------*/

  void AnyItemAbstractFamily::uniqueIdToLocalId(
      Arccore::Int32ArrayView localIds, Arccore::Int64ConstArrayView uniqueIds) const
  {
    ARCANE_ASSERT(
        (localIds.size() == uniqueIds.size()), ("Incompatible array argument sizes"));
    const Arccore::Integer size = uniqueIds.size();

    // AnyItem uid -> igrp + item unique id
    Arccore::UniqueArray<Arccore::Int32> group_ids(size); // tableau mémorisant dans de
                                                          // quel groupe est issu l'item à
                                                          // la même position
    Arccore::UniqueArray<Arccore::UniqueArray<Arccore::Int64>> unique_id_by_groups(
        m_family.groupSize());
    for (Arccore::Integer i = 0; i < size; ++i) { // Classe les uids par leur groupe
                                                  // sous-jacent avant d'appeler les
                                                  // méthodes
      // par IItemFamily
      Arccore::Int64 uniqueId = uniqueIds[i];
      Arccore::Integer igrp = getIgroup(uniqueId);
      Arccore::Int64 item_uid = getItemUID(uniqueId);
      group_ids[i] = igrp;
      unique_id_by_groups[igrp].add(item_uid);
    }

    // item unique id -> item local id
    Arccore::UniqueArray<Arccore::UniqueArray<Arccore::Int32>> local_id_by_groups(
        m_family.groupSize());
    for (Arccore::Integer igrp = 0; igrp < m_family.groupSize(); ++igrp) {
      local_id_by_groups[igrp].resize(unique_id_by_groups[igrp].size());
      if (not local_id_by_groups[igrp].empty()) {
        Arcane::ItemGroup group = m_family.group(igrp);
        Arcane::IItemFamily* family = group.itemFamily();
        family->itemsUniqueIdToLocalId(
            local_id_by_groups[igrp], unique_id_by_groups[igrp], true); // do_fatal = true
      }
    }
    unique_id_by_groups.dispose();

    // item local id -> AnyItem local id
    Arccore::UniqueArray<Arccore::Integer> current_indexes(m_family.groupSize(), 0);
    for (Arccore::Integer i = 0; i < size; ++i) {
      const Arccore::Integer group_id = group_ids[i];
      Arcane::ItemGroup group = m_family.group(group_id);
      const Arccore::Int32 item_lid =
          local_id_by_groups[group_id][current_indexes[group_id]++];
      const Arccore::Integer local_index = (*(group.localIdToIndex()))[item_lid];
      localIds[i] = m_lower_bounds[group_id] + local_index;
    }
  }

  /*---------------------------------------------------------------------------*/

  IIndexManager::IAbstractFamily::Item AnyItemAbstractFamily::item(
      Arccore::Int32 localId) const
  {
    ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
    Arccore::Integer igrp = Alien::ArrayScan::dichotomicIntervalScan(
        localId, m_lower_bounds.size(), m_lower_bounds.unguardedBasePointer());
    ARCANE_ASSERT((igrp < m_family.groupSize()), ("Invalid group index"));
    Arcane::ItemVectorView view = m_family.group(igrp).view();
    Arcane::Item item = view[localId - m_lower_bounds[igrp]];
    return IAbstractFamily::Item(makeUID(igrp, item.uniqueId()), item.owner());
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Integer> AnyItemAbstractFamily::owners(
      Arccore::Int32ConstArrayView localIds) const
  {
    const Arccore::Integer size = localIds.size();
    Arccore::SharedArray<Arccore::Integer> result(size);
    for (Arccore::Integer i = 0; i < size; ++i) {
      Arccore::Int32 localId = localIds[i];
      ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
      Arccore::Integer igrp = Alien::ArrayScan::dichotomicIntervalScan(
          localId, m_lower_bounds.size(), m_lower_bounds.unguardedBasePointer());
      ARCANE_ASSERT((igrp < m_family.groupSize()), ("Invalid group index"));
      Arcane::ItemVectorView view = m_family.group(igrp).view();
      Arcane::Item item = view[localId - m_lower_bounds[igrp]];
      result[i] = item.owner();
    }
    return result;
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Int64> AnyItemAbstractFamily::uids(
      Arccore::Int32ConstArrayView localIds) const
  {
    const Arccore::Integer size = localIds.size();
    Arccore::SharedArray<Arccore::Int64> result(size);
    for (Arccore::Integer i = 0; i < size; ++i) {
      Arccore::Int32 localId = localIds[i];
      ARCANE_ASSERT((localId < m_family.maxLocalId()), ("Bad Abstract Family localId"));
      Arccore::Integer igrp = Alien::ArrayScan::dichotomicIntervalScan(
          localId, m_lower_bounds.size(), m_lower_bounds.unguardedBasePointer());
      ARCANE_ASSERT((igrp < m_family.groupSize()), ("Invalid group index"));
      Arcane::ItemVectorView view = m_family.group(igrp).view();
      Arcane::Item item = view[localId - m_lower_bounds[igrp]];
      result[i] = makeUID(igrp, item.uniqueId());
    }
    return result;
  }

  /*---------------------------------------------------------------------------*/

  Arccore::SharedArray<Arccore::Int32> AnyItemAbstractFamily::allLocalIds() const
  {
    const Arccore::Integer size = m_family.maxLocalId();
    Arccore::SharedArray<Arccore::Int32> local_ids(size);
    for (Arccore::Integer i = 0; i < size; ++i)
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
