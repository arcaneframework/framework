// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.cc                     (C) 2000-2023 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"
#include "arcane/utils/ValueConvert.h"

#include "arcane/core/IParallelMng.h"
#include "arcane/core/IVariableSynchronizer.h"
#include "arcane/core/materials/internal/IMeshMaterialVariableInternal.h"

#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MeshMaterialVariable.h"
#include "arcane/materials/IMeshMaterialSynchronizeBuffer.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

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
      m_use_generic_version = (v.value()==0);
  }

 public:

  IMeshMaterialMng* m_material_mng;
  UniqueArray<MeshMaterialVariable*> m_mat_env_vars;
  UniqueArray<MeshMaterialVariable*> m_env_only_vars;
  Int64 m_total_size = 0;
  bool m_use_generic_version = true;
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
  m_p->m_total_size = 0;
  // TODO: modifier le synchroniser pour faire cela en une passe
  // de send/recv/wait.
  IMeshMaterialMng* mm = m_p->m_material_mng;
  if (!m_p->m_mat_env_vars.empty())
    _synchronizeMultiple(m_p->m_mat_env_vars,mm->_allCellsMatEnvSynchronizer());
  if (!m_p->m_env_only_vars.empty())
    _synchronizeMultiple(m_p->m_env_only_vars,mm->_allCellsEnvOnlySynchronizer());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
add(MeshMaterialVariable* var)
{
  MatVarSpace mvs = var->space();
  if (mvs==MatVarSpace::MaterialAndEnvironment)
    m_p->m_mat_env_vars.add(var);
  else if (mvs==MatVarSpace::Environment)
    m_p->m_env_only_vars.add(var);
  else
    ARCANE_THROW(NotSupportedException,"Invalid space for variable name={0} space={1}",
                 var->name(),(int)mvs);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_synchronizeMultiple(ConstArrayView<MeshMaterialVariable*> vars,
                     IMeshMaterialVariableSynchronizer* mmvs)
{
  Int32 sync_version = m_p->m_material_mng->synchronizeVariableVersion();
  Ref<IMeshMaterialSynchronizeBuffer> buf_list;

  if (sync_version==8){
    // Version 8. Utilise le buffer commun pour éviter les multiples allocations
    buf_list = mmvs->commonBuffer();
  }
  else if (sync_version==7){
    // Version 7. Utilise un buffer unique mais réalloué à chaque fois.
    buf_list =  impl::makeOneBufferMeshMaterialSynchronizeBufferRef();
  }
  else{
    // Version 6. Version historique avec plusieurs buffers recréés à chaque fois.
    buf_list = impl::makeMultiBufferMeshMaterialSynchronizeBufferRef();
  }
  if (sync_version<8){
    Int32ConstArrayView ranks = mmvs->variableSynchronizer()->communicatingRanks();
    Integer nb_rank = ranks.size();
    buf_list->setNbRank(nb_rank);
  }
  _synchronizeMultiple2(vars,mmvs,buf_list.get());
  m_p->m_total_size += buf_list->totalSize();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_synchronizeMultiple2(ConstArrayView<MeshMaterialVariable*> vars,
                      IMeshMaterialVariableSynchronizer* mmvs,
                      IMeshMaterialSynchronizeBuffer* buf_list)
{
  // Version de la synchronisation qui envoie uniquement
  // les valeurs des matériaux et des milieux pour les mailles
  // partagées.
  // NOTE: Cette version nécessite que les matériaux soient correctement
  // synchronisés entre les sous-domaines.

  IVariableSynchronizer* var_syncer = mmvs->variableSynchronizer();
  IParallelMng* pm = var_syncer->parallelMng();
  Int32 sync_version = m_p->m_material_mng->synchronizeVariableVersion();

  // TODO: gérer l'alignement des multiples buffer.

  // TODO: ajouter un tag pour garantir que les synchros sont sur les
  // memes variables.

  if (!pm->isParallel())
    return;
  const bool use_new_version = m_p->m_use_generic_version;
  RunQueue* runqueue = nullptr;

  mmvs->checkRecompute();

  ITraceMng* tm = pm->traceMng();
  Integer nb_var = vars.size();
  tm->info(4) << "MAT_SYNCHRONIZE version=" << sync_version << " multiple n="
              << nb_var << " is_generic?=" << use_new_version;

  Int32UniqueArray data_sizes(nb_var);
  Integer all_datatype_size = 0;
  for( Integer i=0; i<nb_var; ++i ){
    data_sizes[i] = vars[i]->dataTypeSize();
    all_datatype_size += data_sizes[i];
    tm->info(4) << "MAT_SYNCHRONIZE name=" << vars[i]->name()
                << " size=" << data_sizes[i];
  }
  
  {
    Int32ConstArrayView ranks = var_syncer->communicatingRanks();
    Integer nb_rank = ranks.size();

    UniqueArray<Parallel::Request> requests;

    // Calcul la taille des buffers et réalloue si nécessaire
    for( Integer i=0; i<nb_rank; ++i ){
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total_ghost = ghost_matcells.size();
      buf_list->setReceiveBufferSize(i,total_ghost * all_datatype_size);
      ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
      Integer total_shared = shared_matcells.size();
      buf_list->setSendBufferSize(i,total_shared * all_datatype_size);
    }
    buf_list->allocate();

    // Poste les receive.
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      requests.add(pm->recv(buf_list->receiveBuffer(i).smallView(),rank,false));
    }

    // Poste les send
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
      Integer total_shared = shared_matcells.size();
      ByteArrayView values(buf_list->sendBuffer(i).smallView());
      Integer offset = 0;
      for( Integer z=0; z<nb_var; ++z ){
        Integer my_data_size = data_sizes[z];
        auto sub_view = values.subView(offset,total_shared * my_data_size);
        if (use_new_version){
          std::byte* ptr = reinterpret_cast<std::byte*>(sub_view.data());
          vars[z]->_internalApi()->copyToBuffer(shared_matcells,{ptr,sub_view.size()},runqueue);
        }
        else
          vars[z]->copyToBuffer(shared_matcells,sub_view);
        offset += total_shared * my_data_size;
      }
      requests.add(pm->send(values,rank,false));
    }

    pm->waitAllRequests(requests);

    // Recopie les données recues dans les mailles fantomes.
    for( Integer i=0; i<nb_rank; ++i ){
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total_ghost = ghost_matcells.size();
      ByteConstArrayView values(buf_list->receiveBuffer(i).smallView());

      Integer offset = 0;
      for( Integer z=0; z<nb_var; ++z ){
        Integer my_data_size = data_sizes[z];
        auto sub_view = values.subView(offset,total_ghost * my_data_size);
        if (use_new_version){
          const std::byte* ptr = reinterpret_cast<const std::byte*>(sub_view.data());
          vars[z]->_internalApi()->copyFromBuffer(ghost_matcells,{ptr,sub_view.size()},runqueue);
        }
        else
          vars[z]->copyFromBuffer(ghost_matcells,sub_view);
        offset += total_ghost * my_data_size;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
