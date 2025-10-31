// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.cc                     (C) 2000-2024 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ValueConvert.h"
#include "arcane/utils/FatalErrorException.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/internal/IParallelMngInternal.h"
#include "arcane/core/materials/IMeshMaterialMng.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"
#include "arcane/core/materials/internal/IMeshMaterialMngInternal.h"

#include "arcane/materials/MeshMaterialVariable.h"
#include "arcane/materials/IMeshMaterialSynchronizeBuffer.h"
#include "arcane/materials/IMeshMaterialVariableSynchronizer.h"

#include "arcane/accelerator/core/RunQueue.h"

#include "arccore/message_passing/Messages.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{
namespace MP = Arccore::MessagePassing;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariableSynchronizerList::SyncInfo
{
 public:

  UniqueArray<Parallel::Request> requests;
  Ref<IMeshMaterialSynchronizeBuffer> buf_list;
  UniqueArray<Int32> data_sizes;
  bool use_generic_version = false;
  Int32 sync_version = 0;
  IMeshMaterialVariableSynchronizer* mat_synchronizer = nullptr;
  Int64 message_total_size = 0;
  UniqueArray<MeshMaterialVariable*> variables;
  MP::MessageTag message_tag;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariableSynchronizerList::Impl
{
 public:

  explicit Impl(IMeshMaterialMng* material_mng)
  : m_material_mng(material_mng)
  {
    // Pour utiliser l'ancien (avant la version accélérateur) mécanisme de synchronisation.
    // TEMPORAIRE: à supprimer fin 2023.
    if (auto v = Convert::Type<Int32>::tryParseFromEnvironment("ARCANE_MATERIAL_LEGACY_SYNCHRONIZE", true))
      m_use_generic_version = (v.value() == 0);
  }

 public:

  IMeshMaterialMng* m_material_mng;
  UniqueArray<MeshMaterialVariable*> m_mat_env_vars;
  UniqueArray<MeshMaterialVariable*> m_env_only_vars;
  Int64 m_total_size = 0;
  bool m_use_generic_version = true;
  eMemoryRessource m_buffer_memory_ressource = eMemoryRessource::Host;
  SyncInfo m_mat_env_sync_info;
  SyncInfo m_env_only_sync_info;
  bool m_is_in_sync = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableSynchronizerList::
MeshMaterialVariableSynchronizerList(IMeshMaterialMng* material_mng)
: m_p(new Impl(material_mng))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MeshMaterialVariableSynchronizerList::
~MeshMaterialVariableSynchronizerList()
{
  delete m_p;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

Int64 MeshMaterialVariableSynchronizerList::
totalMessageSize() const
{
  return m_p->m_total_size;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
apply()
{
  // Avec la version 8 des synchronisations, il faut faire obligatoirement
  // que chaque synchronisation soit bloquante car il n'y a qu'un seul
  // buffer partagé.
  Int32 v = m_p->m_material_mng->synchronizeVariableVersion();
  bool is_blocking = (v == 8);

  _beginSynchronize(is_blocking);
  _endSynchronize(is_blocking);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
beginSynchronize()
{
  Int32 v = m_p->m_material_mng->synchronizeVariableVersion();
  if (v != 7)
    ARCANE_FATAL("beginSynchronize() is only valid for synchronize version 7 (v={0})", v);
  _beginSynchronize(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
endSynchronize()
{
  _endSynchronize(false);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_beginSynchronize(bool is_blocking)
{
  if (m_p->m_is_in_sync)
    ARCANE_FATAL("Already synchronizing");
  m_p->m_is_in_sync = true;

  m_p->m_total_size = 0;
  // TODO: modifier le synchroniser pour faire cela en une passe
  // de send/recv/wait.
  IMeshMaterialMng* mm = m_p->m_material_mng;
  m_p->m_mat_env_sync_info = SyncInfo();
  m_p->m_env_only_sync_info = SyncInfo();

  {
    SyncInfo& sync_info = m_p->m_mat_env_sync_info;
    _fillSyncInfo(sync_info);
    sync_info.mat_synchronizer = mm->_internalApi()->allCellsMatEnvSynchronizer();
    sync_info.variables = m_p->m_mat_env_vars;
    sync_info.message_tag = MP::MessageTag(569);
    if (!sync_info.variables.empty()) {
      _beginSynchronizeMultiple(sync_info);
      if (is_blocking)
        _endSynchronizeMultiple2(sync_info);
    }
  }
  {
    SyncInfo& sync_info = m_p->m_env_only_sync_info;
    _fillSyncInfo(sync_info);
    sync_info.mat_synchronizer = mm->_internalApi()->allCellsEnvOnlySynchronizer();
    sync_info.variables = m_p->m_env_only_vars;
    sync_info.message_tag = MP::MessageTag(585);
    if (!sync_info.variables.empty()) {
      _beginSynchronizeMultiple(sync_info);
      if (is_blocking)
        _endSynchronizeMultiple2(sync_info);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_endSynchronize(bool is_blocking)
{
  if (!m_p->m_is_in_sync)
    ARCANE_FATAL("beginSynchronize() has to be called before endSynchronize()");

  {
    SyncInfo& sync_info = m_p->m_mat_env_sync_info;
    if (!sync_info.variables.empty() && !is_blocking)
      _endSynchronizeMultiple2(sync_info);
    m_p->m_total_size += sync_info.message_total_size;
  }
  {
    SyncInfo& sync_info = m_p->m_env_only_sync_info;
    if (!sync_info.variables.empty() && !is_blocking)
      _endSynchronizeMultiple2(sync_info);
    m_p->m_total_size += sync_info.message_total_size;
  }

  m_p->m_is_in_sync = false;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
add(MeshMaterialVariable* var)
{
  MatVarSpace mvs = var->space();
  if (mvs == MatVarSpace::MaterialAndEnvironment)
    m_p->m_mat_env_vars.add(var);
  else if (mvs == MatVarSpace::Environment)
    m_p->m_env_only_vars.add(var);
  else
    ARCANE_THROW(NotSupportedException, "Invalid space for variable name={0} space={1}",
                 var->name(), (int)mvs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_fillSyncInfo(SyncInfo& sync_info)
{
  sync_info.use_generic_version = m_p->m_use_generic_version;
  sync_info.sync_version = m_p->m_material_mng->synchronizeVariableVersion();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_beginSynchronizeMultiple(SyncInfo& sync_info)
{
  IMeshMaterialVariableSynchronizer* mmvs = sync_info.mat_synchronizer;
  eMemoryRessource mem = mmvs->bufferMemoryRessource();
  const Int32 sync_version = sync_info.sync_version;

  if (sync_version == 8) {
    // Version 8. Utilise le buffer commun pour éviter les multiples allocations
    sync_info.buf_list = mmvs->commonBuffer();
  }
  else if (sync_version == 7) {
    // Version 7. Utilise un buffer unique, mais réalloué à chaque fois.
    sync_info.buf_list = impl::makeOneBufferMeshMaterialSynchronizeBufferRef(mem);
  }
  else {
    // Version 6. Version historique avec plusieurs buffers recréés à chaque fois.
    sync_info.buf_list = impl::makeMultiBufferMeshMaterialSynchronizeBufferRef(mem);
  }
  if (sync_version < 8) {
    Int32ConstArrayView ranks = mmvs->variableSynchronizer()->communicatingRanks();
    Integer nb_rank = ranks.size();
    sync_info.buf_list->setNbRank(nb_rank);
  }

  _beginSynchronizeMultiple2(sync_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_beginSynchronizeMultiple2(SyncInfo& sync_info)
{
  ConstArrayView<MeshMaterialVariable*> vars = sync_info.variables;
  IMeshMaterialSynchronizeBuffer* buf_list = sync_info.buf_list.get();
  IMeshMaterialVariableSynchronizer* mmvs = sync_info.mat_synchronizer;
  // Version de la synchronisation qui envoie uniquement
  // les valeurs des matériaux et des milieux pour les mailles
  // partagées.
  // NOTE: Cette version nécessite que les matériaux soient correctement
  // synchronisés entre les sous-domaines.

  IVariableSynchronizer* var_syncer = mmvs->variableSynchronizer();
  IParallelMng* pm = var_syncer->parallelMng();
  IMessagePassingMng* mpm = pm->messagePassingMng();

  // TODO: gérer l'alignement des multiples buffer.

  // TODO: ajouter un tag pour garantir que les synchros sont sur les
  // memes variables.

  if (!pm->isParallel())
    return;
  const bool use_new_version = sync_info.use_generic_version;
  RunQueue queue = pm->_internalApi()->queue();

  mmvs->checkRecompute();

  ITraceMng* tm = pm->traceMng();
  Integer nb_var = vars.size();
  tm->info(4) << "MAT_SYNCHRONIZE version=" << sync_info.sync_version << " multiple n="
              << nb_var << " is_generic?=" << use_new_version;

  sync_info.data_sizes.resize(nb_var);
  Integer all_datatype_size = 0;
  for (Integer i = 0; i < nb_var; ++i) {
    sync_info.data_sizes[i] = vars[i]->dataTypeSize();
    all_datatype_size += sync_info.data_sizes[i];
    tm->info(4) << "MAT_SYNCHRONIZE name=" << vars[i]->name()
                << " size=" << sync_info.data_sizes[i];
  }

  Int32ConstArrayView ranks = var_syncer->communicatingRanks();
  Int32 nb_rank = ranks.size();

  // Calcul la taille des buffers et réalloue si nécessaire
  for (Integer i = 0; i < nb_rank; ++i) {
    ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
    Integer total_ghost = ghost_matcells.size();
    buf_list->setReceiveBufferSize(i, total_ghost * all_datatype_size);
    ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
    Integer total_shared = shared_matcells.size();
    buf_list->setSendBufferSize(i, total_shared * all_datatype_size);
  }
  buf_list->allocate();

  // Poste les receive.
  for (Integer i = 0; i < nb_rank; ++i) {
    Int32 rank = ranks[i];
    MP::PointToPointMessageInfo msg_info(MP::MessageRank(rank), sync_info.message_tag, MP::eBlockingType::NonBlocking);
    sync_info.requests.add(mpReceive(mpm, buf_list->receiveBuffer(i), msg_info));
  }

  // Copie les valeurs dans les buffers
  for (Integer i = 0; i < nb_rank; ++i) {
    ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
    Integer total_shared = shared_matcells.size();
    ByteArrayView values(buf_list->sendBuffer(i).smallView());
    Integer offset = 0;
    for (Integer z = 0; z < nb_var; ++z) {
      Integer my_data_size = sync_info.data_sizes[z];
      auto sub_view = values.subView(offset, total_shared * my_data_size);
      if (use_new_version) {
        auto* ptr = reinterpret_cast<std::byte*>(sub_view.data());
        vars[z]->_internalApi()->copyToBuffer(shared_matcells, { ptr, sub_view.size() }, &queue);
      }
      else
        vars[z]->copyToBuffer(shared_matcells, sub_view);
      offset += total_shared * my_data_size;
    }
  }

  // Attend que les copies soient terminées
  queue.barrier();

  // Poste les sends
  for (Integer i = 0; i < nb_rank; ++i) {
    Int32 rank = ranks[i];
    MP::PointToPointMessageInfo msg_info(MP::MessageRank(rank), sync_info.message_tag, MP::eBlockingType::NonBlocking);
    sync_info.requests.add(mpSend(mpm, buf_list->sendBuffer(i), msg_info));
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_endSynchronizeMultiple2(SyncInfo& sync_info)
{
  ConstArrayView<MeshMaterialVariable*> vars = sync_info.variables;
  IMeshMaterialVariableSynchronizer* mmvs = sync_info.mat_synchronizer;
  IVariableSynchronizer* var_syncer = mmvs->variableSynchronizer();
  IParallelMng* pm = var_syncer->parallelMng();

  if (!pm->isParallel())
    return;
  const bool use_new_version = sync_info.use_generic_version;
  RunQueue queue = pm->_internalApi()->queue();
  IMeshMaterialSynchronizeBuffer* buf_list = sync_info.buf_list.get();

  Int32ConstArrayView ranks = var_syncer->communicatingRanks();
  Int32 nb_rank = ranks.size();
  Integer nb_var = vars.size();

  pm->waitAllRequests(sync_info.requests);

  // Recopie les données recues dans les mailles fantomes.
  for (Integer i = 0; i < nb_rank; ++i) {
    ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
    Integer total_ghost = ghost_matcells.size();
    ByteConstArrayView values(buf_list->receiveBuffer(i).smallView());

    Integer offset = 0;
    for (Integer z = 0; z < nb_var; ++z) {
      Integer my_data_size = sync_info.data_sizes[z];
      auto sub_view = values.subView(offset, total_ghost * my_data_size);
      if (use_new_version) {
        auto* ptr = reinterpret_cast<const std::byte*>(sub_view.data());
        vars[z]->_internalApi()->copyFromBuffer(ghost_matcells, { ptr, sub_view.size() }, &queue);
      }
      else
        vars[z]->copyFromBuffer(ghost_matcells, sub_view);
      offset += total_ghost * my_data_size;
    }
  }
  sync_info.message_total_size += buf_list->totalSize();

  // Attend que les copies soient terminées
  queue.barrier();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
