// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2026 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* VariableSynchronizerComputeList.cc                          (C) 2000-2024 */
/*                                                                           */
/* Calculation of the list of entities to synchronize.                       */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/internal/VariableSynchronizerComputeList.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/OStringStream.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IItemFamily.h"
#include "arcane/core/ItemPrinter.h"

#include "arcane/impl/DataSynchronizeInfo.h"
#include "arcane/impl/internal/VariableSynchronizer.h"

#include <algorithm>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSynchronizerComputeList::
VariableSynchronizerComputeList(VariableSynchronizer* var_sync)
: TraceAccessor(var_sync->traceMng())
, m_synchronizer(var_sync)
, m_parallel_mng(var_sync->m_parallel_mng)
, m_item_group(var_sync->m_item_group)
, m_is_verbose(var_sync->m_is_verbose)
{
  if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_DEBUG_VARIABLESYNCHRONIZERCOMPUTELIST", true))
    m_is_debug = (v.value() != 0);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*!
 * \brief Creation of the synchronization element list.
 *
 * To generate synchronization information, it is assumed that the owner()
 * of each entity is correct.
 *
 * Based on the communication file, constructs the structures for
 * synchronization. It is a list of elements, each element consisting
 * of the processor rank with which communication must occur and the localId()
 * of the mesh entities that must be sent and received.
 *
 * If the group associated with this instance is allItems(), it is checked
 * that all entities in the family are either domain-local or ghosts. If
 * an entity is not in this case, it will not be synchronized and parallelism
 * consistency will not be ensured: this is a fatal error.
 *
 * Operation on any group (other than allItems) is primarily dependent on
 * changeLocalIds() being implemented on all groups.
 */
void VariableSynchronizerComputeList::
compute()
{
  const bool is_debug = m_is_debug;
  IItemFamily* item_family = m_item_group.itemFamily();
  Int32 my_rank = m_parallel_mng->commRank();
  Int32 nb_rank = m_parallel_mng->commSize();

  m_is_verbose = traceMng()->verbosityLevel() >= 4;

  UniqueArray<SharedArray<Int32>> boundary_items(nb_rank);

  info() << "Compute synchronize informations group=" << m_item_group.name()
         << " family=" << item_family->fullName()
         << " group size=" << m_item_group.size()
         << " is_verbose=" << m_is_verbose;

  {
    Integer nb_error = 0;
    Int64UniqueArray bad_items_uid;
    ENUMERATE_ITEM (i, m_item_group) {
      Item item = *i;
      impl::ItemBase item_internal = item.itemBase();
      Int32 owner = item_internal.owner();
      if (owner == my_rank)
        continue;
      Int64 uid = item_internal.uniqueId().asInt64();
      if (owner == A_NULL_RANK || owner >= nb_rank) {
        ++nb_error;
        if (nb_error < 10)
          bad_items_uid.add(uid);
        continue;
      }
      if (is_debug) {
        info() << "Add entity uid=" << uid
               << " lid=" << item_internal.localId() << " to the subdomain " << owner;
      }
      boundary_items[owner].add(item_internal.localId());
    }
    if (nb_error != 0) {
      for (Int64 uid : bad_items_uid) {
        info() << "ERROR: The entity uid=" << uid
               << " group=" << m_item_group.name() << " doesn't belong to "
               << "any subdomain or belong to an invalid subdomain";
      }
      ARCANE_FATAL("Error while creating synchronization information nb_error={0}", nb_error);
    }
  }

  _createList(boundary_items);

  if (is_debug)
    _printSyncList();

  info() << "End compute synchronize information group=" << m_item_group.name()
         << " Date=" << platform::getCurrentDateTime();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerComputeList::
_createList(UniqueArray<SharedArray<Int32>>& boundary_items)
{
  const bool is_debug = m_is_debug;
  DataSynchronizeInfo* sync_info = m_synchronizer->m_sync_info.get();

  sync_info->clear();

  IItemFamily* item_family = m_item_group.itemFamily();
  IParallelMng* pm = m_parallel_mng;
  Int32 my_rank = pm->commRank();
  Int32 nb_rank = pm->commSize();
  info(4) << "VariableSynchronizer::createList() begin for group=" << m_item_group.name();

  Real time_begin = platform::getRealTime();
  Real time_before_all_gather = 0.0;
  Real time_after_all_gather = 0.0;
  Real time_before_sendrecv = 0.0;
  Real time_after_sendrecv = 0.0;
  Real time_after_sendrecv_wait = 0.0;

  // Table of neighborhood known by ghost items.
  // This is not necessarily the final list for sync_info->communicatingRanks() in the case
  // of non-symmetric ghost/shared relationship between processors (if one of them is 0)
  // The complementary processing appears after the "Communication Reciprocity" section
  Int32UniqueArray communicating_ghost_ranks;
  for (Integer i = 0; i < nb_rank; ++i) {
    if (boundary_items[i].empty())
      continue;
    communicating_ghost_ranks.add(i);
    if (is_debug) {
      ItemInfoListView items_internal(item_family);
      for (Integer z = 0, zs = boundary_items[i].size(); z < zs; ++z) {
        Item item = items_internal[boundary_items[i][z]];
        info() << "Item uid=" << item.uniqueId() << ",lid=" << item.localId();
      }
    }
  }

  Integer nb_comm_rank = communicating_ghost_ranks.size();

  Int32UniqueArray nb_ghost(nb_rank);
  Int32UniqueArray nb_share(nb_rank);
  nb_ghost.fill(0);

  // Maximum number of connected subdomains. Used to dimension
  // the arrays for the allGather()
  Integer max_comm_rank = pm->reduce(Parallel::ReduceMax, nb_comm_rank);
  debug() << "communicating sub domains my=" << nb_comm_rank
          << " max=" << max_comm_rank;

  // List of ghost mesh groups.
  UniqueArray<SharedArray<Int32>> ghost_group_list(boundary_items.size());

  // Retrieves the lists of ghost entities.
  for (Integer i = 0; i < nb_comm_rank; ++i) {
    Int32 current_rank = communicating_ghost_ranks[i];
    SharedArray<Int32>& ghost_grp = boundary_items[current_rank];
    ghost_group_list[i] = ghost_grp;
    nb_ghost[current_rank] = ghost_grp.size();
  }

  UniqueArray<ShareRankInfo> share_rank_info;
  UniqueArray<GhostRankInfo> ghost_rank_info;
  {
    Integer gather_size = 1 + (max_comm_rank * 2);
    Int32UniqueArray global_ghost_info(gather_size * nb_rank);
    {
      // Each subdomain constructs an array indicating for
      // each group of ghost elements, the concerned processor and
      // the number of elements in this group.
      // This array will then be gathered across all subdomains
      // (by an allGather()) so that each subdomain can iterate through it
      // and then know who owns its shared meshes.
      Int32UniqueArray local_ghost_info(gather_size);
      Integer pos = 0;
      local_ghost_info[pos++] = nb_comm_rank; // Indicates the number of elements in the array
      debug() << "Send local info " << nb_comm_rank;
      for (Integer index = 0, s = communicating_ghost_ranks.size(); index < s; ++index) {
        local_ghost_info[pos++] = communicating_ghost_ranks[index];
        local_ghost_info[pos++] = ghost_group_list[index].size();
        debug() << "Send local info i=" << index << " target=" << communicating_ghost_ranks[index]
                << " nb=" << ghost_group_list[index].size();
      }
      time_before_all_gather = platform::getRealTime();
      if (m_is_verbose) {
        info() << "AllGather size() " << local_ghost_info.size()
               << ' ' << global_ghost_info.size()
               << " begin_date=" << platform::getCurrentDateTime();
      }
      pm->allGather(local_ghost_info, global_ghost_info);
      time_after_all_gather = platform::getRealTime();
      if (m_is_verbose) {
        info() << "AllGather end_date=" << platform::getCurrentDateTime();
      }
    }
    {
      for (Integer index = 0, s = nb_rank; index < s; ++index) {
        Integer pos = gather_size * index;
        Integer sub_size = global_ghost_info[pos++];
        for (Integer sub_index = 0; sub_index < sub_size; ++sub_index) {
          Integer proc_id = global_ghost_info[pos++];
          Integer nb_elem = global_ghost_info[pos++];
          if (proc_id == my_rank) {
            if (is_debug) {
              info() << "Get for share group " << index << ' ' << nb_elem;
            }
            share_rank_info.add(ShareRankInfo(index, nb_elem));
          }
        }
      }
    }

    {
      // Created the infos concerning the ghost meshes
      Integer nb_send = communicating_ghost_ranks.size();
      ghost_rank_info.resize(nb_send);
      for (Integer i = 0; i < nb_send; ++i) {
        SharedArray<Int32> gr = ghost_group_list[i];
        ghost_rank_info[i].setInfos(communicating_ghost_ranks[i], gr);
      }
    }
  }
  //pm->barrier();
  ItemInfoListView items_internal(item_family);
  {
    {
      // Communication Reciprocity
      // For synchronizations to work correctly, it is necessary that
      // 'share_rank_info' and 'ghost_rank_info' have
      // the same number of elements. If this is not the case, it means
      // that a processor 'n' owns shared meshes with proc 'm'
      // without this being reciprocal. If this is the case, we add
      // a reference to this subdomain in 'share_rank_info'
      // with no elements to send.
      Integer nb_recv = share_rank_info.size();
      Integer nb_send = ghost_rank_info.size();

      if (is_debug) {
        info() << "Infos before auto add: send " << nb_send << " recv " << nb_recv;
        for (Integer i = 0; i < ghost_rank_info.size(); ++i) {
          const GhostRankInfo& asdi = ghost_rank_info[i];
          info() << "Ghost: " << i << asdi.nbItem() << ' ' << asdi.rank();
        }
        for (Integer i = 0; i < share_rank_info.size(); ++i) {
          const ShareRankInfo& asdi = share_rank_info[i];
          info() << "Shared: " << i << ' ' << asdi.nbItem() << ' ' << asdi.rank();
        }
      }

      for (Integer i = 0; i < nb_send; ++i) {
        Integer proc_id = ghost_rank_info[i].rank();
        Integer z = 0;
        for (; z < nb_recv; ++z)
          if (share_rank_info[z].rank() == proc_id)
            break;
        debug(Trace::Highest) << "CHECKS " << proc_id << ' ' << z << ' ' << nb_recv;
        if (z == nb_recv) {
          debug() << "Add communication with the subdomain " << proc_id;
          share_rank_info.add(ShareRankInfo(proc_id));
        }
      }

      for (Integer i = 0; i < nb_recv; ++i) {
        Integer proc_id = share_rank_info[i].rank();
        Integer z = 0;
        for (; z < nb_send; ++z)
          if (ghost_rank_info[z].rank() == proc_id)
            break;
        debug(Trace::Highest) << "CHECKR " << proc_id << ' ' << z << ' ' << nb_send;
        if (z == nb_send) {
          debug() << "Add communication with subdomain " << proc_id;
          ghost_rank_info.add(GhostRankInfo(proc_id));
        }
      }

      if (ghost_rank_info.size() != share_rank_info.size()) {
        ARCANE_FATAL("Problem with the number of subdomain shared ({0}) and ghosts ({1})",
                     share_rank_info.size(), ghost_rank_info.size());
      }
      // Sort the array by increasing subdomain number.
      std::sort(std::begin(share_rank_info), std::end(share_rank_info));
      std::sort(std::begin(ghost_rank_info), std::end(ghost_rank_info));
    }

    // OK, now we know the list of subdomains that possess
    // the shared meshes of this subdomain and conversely, the list of
    // subdomains interested in the own meshes of this subdomain.
    // All that remains is to send and receive the corresponding information.
    // To do this, and to avoid deadlocks, we first send the info
    // for the subdomains whose number is less than this subdomain.
    Integer nb_comm_proc = ghost_rank_info.size();
    info(4) << "Number of communicating processors: " << nb_comm_proc;
    UniqueArray<Parallel::Request> requests;
    time_before_sendrecv = platform::getRealTime();
    {
      //Integer nb_recv = share_rank_info.size();

      // Sort the array by increasing subdomain number.
      for (Integer i = 0; i < ghost_rank_info.size(); ++i) {
        const GhostRankInfo& asdi = ghost_rank_info[i];
        debug() << "Ghost: " << i << " " << asdi.nbItem() << ' ' << asdi.rank();
      }
      for (Integer i = 0; i < share_rank_info.size(); ++i) {
        const ShareRankInfo& asdi = share_rank_info[i];
        debug() << "Shared: " << i << " " << asdi.nbItem() << ' ' << asdi.rank();
      }
      Integer current_send_index = 0;
      Integer current_recv_index = 0;
      for (Integer i = 0; i < nb_comm_proc * 2; ++i) {
        Integer send_proc = nb_rank;
        Integer recv_proc = nb_rank;
        if (current_send_index != nb_comm_proc)
          send_proc = ghost_rank_info[current_send_index].rank();
        if (current_recv_index != nb_comm_proc)
          recv_proc = ghost_rank_info[current_recv_index].rank();
        bool do_send = true;
        if (send_proc == recv_proc) {
          if (send_proc < my_rank)
            do_send = true;
          else
            do_send = false;
        }
        else if (send_proc < recv_proc)
          do_send = true;
        else
          do_send = false;
        if (do_send) {
          GhostRankInfo& asdi = ghost_rank_info[current_send_index];
          asdi.resize();
          Int64ArrayView uids = asdi.uniqueIds();
          Int32ConstArrayView asdi_local_ids = asdi.localIds();
          //Integer zindex = 0;
          Integer nb_local = asdi_local_ids.size();
          for (Integer z = 0, zs = nb_local; z < zs; ++z) {
            //for( ItemGroup::const_iter z(asdi.group()); z.hasNext(); ++z, ++zindex ){
            const Item& elem = items_internal[asdi_local_ids[z]];
            uids[z] = elem.uniqueId().asInt64();
          }
          if (is_debug) {
            info() << "Number of elements that will be sent to the subdomain " << send_proc
                   << " " << nb_local << " elements";
            for (Integer z = 0; z < nb_local; ++z) {
              info() << "Unique id " << uids[z];
            }
          }
          debug() << "Send proc " << send_proc;
          if (!uids.empty())
            requests.add(pm->send(uids, send_proc, false));
          ++current_send_index;
        }
        else {
          ShareRankInfo& asdi = share_rank_info[current_recv_index];
          asdi.resize();
          Int64ArrayView items_unique_id = asdi.uniqueIds();
          debug() << "Recv proc " << recv_proc;
          //TODO use non-blocking.
          if (!items_unique_id.empty())
            pm->recv(items_unique_id, recv_proc);
          //String group_name(share_name);
          //group_name += recv_proc;

          SharedArray<Int32> items_local_id(items_unique_id.size()); //! Ids of group entities
          item_family->itemsUniqueIdToLocalId(items_local_id, items_unique_id);
          SharedArray<Int32> share_group = items_local_id;
          debug() << "Creating shared entities for the subdomain " << recv_proc
                  << " with " << items_local_id.size() << " entities";
          //ItemGroup share_group = mesh->itemFamily(item_kind)->createGroup(group_name,items_local_id,true);
          //share_group.setLocalToSubDomain(true);
          asdi.setLocalIds(share_group);
          if (is_debug) {
            for (Integer z = 0, zs = share_group.size(); z < zs; ++z) {
              const Item& item = items_internal[share_group[z]];
              info() << "Item uid=" << item.uniqueId() << ",lid=" << item.localId();
            }
          }
          ++current_recv_index;
        }
      }
      time_after_sendrecv = platform::getRealTime();
      if (m_is_verbose) {
        info() << "Wait requests n=" << requests.size()
               << " begin_date=" << platform::getCurrentDateTime();
      }
      pm->waitAllRequests(requests);
      time_after_sendrecv_wait = platform::getRealTime();
      if (m_is_verbose) {
        info() << "Wait requests end_date=" << platform::getCurrentDateTime();
      }
    }
  }
  _checkValid(ghost_rank_info, share_rank_info);
  sync_info->recompute();

  // Verify that all ghosts have been found
  for (Integer i = 0, n = sync_info->size(); i < n; ++i) {
    Int32 target_rank = sync_info->targetRank(i);
    if (sync_info->receiveInfo().nbItem(i) != boundary_items[target_rank].size())
      ARCANE_FATAL("Inconsistent ghost count");
  }

  info() << "VariableSynchronize:: end compute list group=" << m_item_group.name()
         << " t_init=" << Trace::Precision(4, time_before_all_gather - time_begin, true)
         << " t_allgather=" << Trace::Precision(4, time_after_all_gather - time_before_all_gather, true)
         << " t_sendrecv=" << Trace::Precision(4, time_after_sendrecv_wait - time_before_sendrecv, true)
         << " t_wait=" << Trace::Precision(4, time_after_sendrecv_wait - time_after_sendrecv, true);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerComputeList::
_checkValid(ArrayView<GhostRankInfo> ghost_rank_info,
            ArrayView<ShareRankInfo> share_rank_info)
{
  const bool is_debug = m_is_debug;
  Integer nb_comm_proc = ghost_rank_info.size();
  Integer nb_error = 0;
  bool has_error = false;
  const Integer max_error = 10; // Max number of errors displayed.
  Int32 my_rank = m_parallel_mng->commRank();
  IItemFamily* item_family = m_item_group.itemFamily();
  ItemInfoListView items_internal(item_family);

  // Array used to mark elements that are either
  // owned by the subdomain or ghost.
  // Normally, if the data is consistent, this should mark
  // all elements.
  // NOTE: this is only useful if \a itemGroup() equals allItems()
  UniqueArray<bool> marked_elem(item_family->maxLocalId());
  marked_elem.fill(false);
  // Mark elements owned by the subdomain
  ENUMERATE_ITEM (i_item, m_item_group) {
    Item item = *i_item;
    if (item.isOwn()) {
      marked_elem[item.localId()] = true;
      if (is_debug) {
        info() << "Own Item " << ItemPrinter(item);
      }
    }
  }

  for (Integer i_comm = 0; i_comm < nb_comm_proc; ++i_comm) {
    GhostRankInfo& ghost_info = ghost_rank_info[i_comm];
    ShareRankInfo& share_info = share_rank_info[i_comm];
    if (ghost_info.rank() != share_info.rank()) {
      ARCANE_FATAL("Inconsistency between the subdomain numbers ghost_rank={0} share_rank={1}",
                   ghost_info.rank(), share_info.rank());
    }
    Integer current_proc = ghost_info.rank();
    Int32ConstArrayView ghost_grp = ghost_info.localIds();
    Int32ConstArrayView share_grp = share_info.localIds();

    if (share_grp.empty() && ghost_grp.empty()) {
      error() << "Shared and ghosts groups null for the subdomain " << current_proc;
      has_error = true;
      continue;
    }
    if (current_proc == my_rank) {
      error() << "Error in the communication pattern: "
              << "the processor can't communicate with itself";
      has_error = true;
      continue;
    }

    // Mark elements of the shared group
    for (Integer z = 0, zs = ghost_grp.size(); z < zs; ++z) {
      const Item& elem = items_internal[ghost_grp[z]];
      bool is_marked = marked_elem[elem.localId()];
      if (is_marked) {
        // The element should not already be marked.
        if (nb_error < max_error)
          error() << "The entity " << ItemPrinter(elem) << " belongs to another ghost group "
                  << "or is owned by the subdomain.";
        ++nb_error;
        continue;
      }
      marked_elem[elem.localId()] = true;
    }

    m_synchronizer->m_sync_info->add(VariableSyncInfo(share_grp, ghost_grp, current_proc));
  }

  // Verify that all elements are marked
  ENUMERATE_ITEM (i, m_item_group) {
    Item item = *i;
    if (!marked_elem[item.localId()]) {
      if (nb_error < max_error) {
        error() << "The entity " << ItemPrinter(item)
                << " doesn't belong to the subdomain or any ghost group.";
      }
      ++nb_error;
    }
  }

  // If there is an error, we stop.
  if (nb_error != 0) {
    has_error = true;
    if (nb_error >= max_error)
      error() << nb_error << " total elements are incorrectly dealt with";
  }
  if (has_error)
    ARCANE_FATAL("Error while creating the exchange structures of the family={0}",
                 item_family->fullName());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSynchronizerComputeList::
_printSyncList()
{
  DataSynchronizeInfo* sync_info = m_synchronizer->m_sync_info.get();
  Integer nb_comm = sync_info->size();
  info() << "SYNC LIST FOR GROUP : " << m_item_group.fullName() << " N=" << nb_comm;
  OStringStream ostr;
  IItemFamily* item_family = m_item_group.itemFamily();
  ItemInfoListView items_internal(item_family);
  for (Integer i = 0; i < nb_comm; ++i) {
    Int32 target_rank = sync_info->targetRank(i);
    ostr() << " TARGET=" << target_rank << '\n';
    Int32ConstArrayView share_ids = sync_info->sendInfo().localIds(i);
    ostr() << "\t\tSHARE(lid,uid) n=" << share_ids.size() << " :";
    for (Integer z = 0, zs = share_ids.size(); z < zs; ++z) {
      Item item = items_internal[share_ids[z]];
      ostr() << " (" << item.localId() << "," << item.uniqueId() << ")";
    }
    ostr() << "\n";
    Int32ConstArrayView ghost_ids = sync_info->receiveInfo().localIds(i);
    ostr() << "\t\tGHOST(lid,uid) n=" << ghost_ids.size() << " :";
    for (Integer z = 0, zs = ghost_ids.size(); z < zs; ++z) {
      Item item = items_internal[ghost_ids[z]];
      ostr() << " (" << item.localId() << "," << item.uniqueId() << ")";
    }
    ostr() << "\n";
  }
  info() << ostr.str();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
