// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* DataSynchronizeInfo.cc                                      (C) 2000-2023 */
/*                                                                           */
/* Informations pour synchroniser les données.                               */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/impl/DataSynchronizeInfo.h"

#include "arcane/utils/FatalErrorException.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/IMemoryRessourceMng.h"
#include "arcane/utils/MemoryView.h"
#include "arcane/utils/SmallArray.h"
#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/VariableCollection.h"
#include "arcane/core/ParallelMngUtils.h"
#include "arcane/core/IParallelExchanger.h"
#include "arcane/core/ISerializeMessage.h"
#include "arcane/core/ISerializer.h"
#include "arcane/core/IParallelMng.h"
#include "arcane/core/IData.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/internal/IDataInternal.h"

#include "arcane/accelerator/core/Runner.h"

#include "arcane/impl/IBufferCopier.h"
#include "arcane/impl/IDataSynchronizeBuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

// TODO: plutôt que d'utiliser la mémoire managée, il est préférable d'avoir
// une copie sur le device des IDs. Cela permettra d'éviter des transferts
// potentiels si on mélange synchronisation de variables sur accélérateurs et
// sur CPU.

VariableSyncInfo::
VariableSyncInfo()
: m_share_ids(platform::getDefaultDataAllocator())
, m_ghost_ids(platform::getDefaultDataAllocator())
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSyncInfo::
VariableSyncInfo(Int32ConstArrayView share_ids, Int32ConstArrayView ghost_ids,
                 Int32 rank)
: VariableSyncInfo()
{
  m_target_rank = rank;
  m_share_ids.copy(share_ids);
  m_ghost_ids.copy(ghost_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

VariableSyncInfo::
VariableSyncInfo(const VariableSyncInfo& rhs)
: VariableSyncInfo()
{
  // NOTE: pour l'instant (avril 2023) il faut un constructeur de recopie
  // explicite pour spécifier l'allocateur
  m_target_rank = rhs.m_target_rank;
  m_share_ids.copy(rhs.m_share_ids);
  m_ghost_ids.copy(rhs.m_ghost_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
_changeIds(Array<Int32>& ids, Int32ConstArrayView old_to_new_ids)
{
  UniqueArray<Int32> orig_ids(ids);
  ids.clear();

  for (Integer z = 0, zs = orig_ids.size(); z < zs; ++z) {
    Int32 old_id = orig_ids[z];
    Int32 new_id = old_to_new_ids[old_id];
    if (new_id != NULL_ITEM_LOCAL_ID)
      ids.add(new_id);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void VariableSyncInfo::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  _changeIds(m_share_ids, old_to_new_ids);
  _changeIds(m_ghost_ids, old_to_new_ids);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataSynchronizeInfo::
recompute()
{
  Integer nb_message = this->size();

  DataSynchronizeBufferInfoList& receive_info = _receiveInfo();
  DataSynchronizeBufferInfoList& send_info = _sendInfo();

  receive_info.m_displacements_base.resize(nb_message);
  send_info.m_displacements_base.resize(nb_message);

  receive_info.m_total_nb_item = 0;
  send_info.m_total_nb_item = 0;

  {
    Integer ghost_displacement = 0;
    Integer share_displacement = 0;
    Int32 index = 0;
    for (const VariableSyncInfo& vsi : m_ranks_info) {
      Int32 ghost_size = vsi.nbGhost();
      receive_info.m_displacements_base[index] = ghost_displacement;
      ghost_displacement += ghost_size;
      Int32 share_size = vsi.nbShare();
      send_info.m_displacements_base[index] = share_displacement;
      share_displacement += share_size;
      ++index;
    }
    receive_info.m_total_nb_item = ghost_displacement;
    send_info.m_total_nb_item = share_displacement;
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void DataSynchronizeInfo::
changeLocalIds(Int32ConstArrayView old_to_new_ids)
{
  for (VariableSyncInfo& vsi : m_ranks_info) {
    vsi.changeLocalIds(old_to_new_ids);
  }
  recompute();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstArrayView<Int32> DataSynchronizeBufferInfoList::
localIds(Int32 index) const
{
  const VariableSyncInfo& s = m_sync_info->m_ranks_info[index];
  return (m_is_share) ? s.shareIds() : s.ghostIds();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int32 DataSynchronizeBufferInfoList::
nbItem(Int32 index) const
{
  const VariableSyncInfo& s = m_sync_info->m_ranks_info[index];
  return (m_is_share) ? s.nbShare() : s.nbGhost();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
