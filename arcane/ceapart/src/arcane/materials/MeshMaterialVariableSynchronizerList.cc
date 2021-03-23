// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2021 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* MeshMaterialVariableSynchronizerList.cc                     (C) 2000-2016 */
/*                                                                           */
/* Synchroniseur de variables matériaux.                                     */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ITraceMng.h"
#include "arcane/utils/String.h"
#include "arcane/utils/PlatformUtils.h"
#include "arcane/utils/NotSupportedException.h"

#include "arcane/IParallelMng.h"
#include "arcane/IVariableSynchronizer.h"

#include "arcane/materials/MeshMaterialVariableSynchronizerList.h"
#include "arcane/materials/IMeshMaterialMng.h"
#include "arcane/materials/MeshMaterialVariable.h"

#include <vector>

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ARCANE_BEGIN_NAMESPACE
MATERIALS_BEGIN_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

class MeshMaterialVariableSynchronizerList::Impl
{
 public:
  Impl(IMeshMaterialMng* material_mng) : m_material_mng(material_mng){}
 public:
  IMeshMaterialMng* m_material_mng;
  UniqueArray<MeshMaterialVariable*> m_mat_env_vars;
  UniqueArray<MeshMaterialVariable*> m_env_only_vars;
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

void MeshMaterialVariableSynchronizerList::
apply()
{
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
    throw NotSupportedException(A_FUNCINFO,
                                String::format("Invalid space for variable name={0} space={1}",
                                               var->name(),(int)mvs));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void MeshMaterialVariableSynchronizerList::
_synchronizeMultiple(ConstArrayView<MeshMaterialVariable*> vars,
                     IMeshMaterialVariableSynchronizer* mmvs)
{
  // Version de la synchronisation qui envoie uniquement
  // les valeurs des matériaux et des milieux pour les mailles
  // partagées.
  // NOTE: Cette version nécessite que les matériaux soient correctement
  // synchronisés entre les sous-domaines.

  IVariableSynchronizer* var_syncer = mmvs->variableSynchronizer();
  IParallelMng* pm = var_syncer->parallelMng();

  // TODO: gérer l'alignement des multiples buffer.

  // TODO: ajouter un tag pour garantir que les synchros sont sur les
  // memes variables.

  if (!pm->isParallel())
    return;

  mmvs->checkRecompute();
  //mmvs->recompute();
  ITraceMng* tm = pm->traceMng();
  Integer nb_var = vars.size();
  tm->info(4) << "MAT_SYNCHRONIZE_V6 multiple n=" << nb_var;

  Int32UniqueArray data_sizes(nb_var);
  Integer all_datatype_size = 0;
  for( Integer i=0; i<nb_var; ++i ){
    data_sizes[i] = vars[i]->dataTypeSize();
    all_datatype_size += data_sizes[i];
    tm->info(4) << "MAT_SYNCHRONIZE_V6 name=" << vars[i]->name()
                << " size=" << data_sizes[i];
  }
  
  {
    Int32ConstArrayView ranks = var_syncer->communicatingRanks();
    Integer nb_rank = ranks.size();
    std::vector< UniqueArray<Byte> > shared_values(nb_rank);
    std::vector< UniqueArray<Byte> > ghost_values(nb_rank);

    UniqueArray<Parallel::Request> requests;

    // Poste les receive.
    Int32UniqueArray recv_ranks(nb_rank);
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total = ghost_matcells.size();
      ghost_values[i].resize(total * all_datatype_size);
      requests.add(pm->recv(ghost_values[i].view(),rank,false));
    }

    // Poste les send
    for( Integer i=0; i<nb_rank; ++i ){
      Int32 rank = ranks[i];
      ConstArrayView<MatVarIndex> shared_matcells(mmvs->sharedItems(i));
      Integer total_shared = shared_matcells.size();
      shared_values[i].resize(total_shared * all_datatype_size);      
      ByteArrayView values(shared_values[i]);
      Integer offset = 0;
      for( Integer z=0; z<nb_var; ++z ){
        Integer my_data_size = data_sizes[z];
        vars[z]->copyToBuffer(shared_matcells,values.subView(offset,total_shared * my_data_size));
        offset += total_shared * my_data_size;
      }
      requests.add(pm->send(values,rank,false));
    }

    pm->waitAllRequests(requests);

    // Recopie les données recues dans les mailles fantomes.
    for( Integer i=0; i<nb_rank; ++i ){
      ConstArrayView<MatVarIndex> ghost_matcells(mmvs->ghostItems(i));
      Integer total_ghost = ghost_matcells.size();
      ByteConstArrayView values(ghost_values[i].constView());

      Integer offset = 0;
      for( Integer z=0; z<nb_var; ++z ){
        Integer my_data_size = data_sizes[z];
        vars[z]->copyFromBuffer(ghost_matcells,values.subView(offset,total_ghost * my_data_size));
        offset += total_ghost * my_data_size;
      }
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MATERIALS_END_NAMESPACE
ARCANE_END_NAMESPACE

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
