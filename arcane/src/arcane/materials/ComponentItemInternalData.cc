// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternalData.cc                                (C) 2000-2024 */
/*                                                                           */
/* Gestion des listes de 'ComponentItemInternal'.                            */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/materials/internal/ComponentItemInternalData.h"

#include "arcane/utils/MemoryUtils.h"

#include "arcane/core/IItemFamily.h"
#include "arcane/core/internal/IItemFamilyInternal.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemInternalData::
ComponentItemInternalData(MeshMaterialMng* mmg)
: TraceAccessor(mmg->traceMng())
, m_material_mng(mmg)
, m_component_item_internal_storage(MemoryUtils::getAllocatorForMostlyReadOnlyData())
, m_shared_infos(MemoryUtils::getAllocatorForMostlyReadOnlyData())
{
  // Il y a une instance pour les MatCell, les EnvCell et les AllEnvCell
  // Il ne faut ensuite plus modifier ce tableau car on conserve des pointeurs
  // vers les éléments de ce tableau.
  m_shared_infos.resize(3);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
endCreate()
{
  const Int32 nb_env = m_material_mng->environments().size();
  m_mat_items_internal.clear();
  m_mat_items_internal.resize(nb_env);

  _initSharedInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Réinitialise les ComponentItemInternal.
 */
void ComponentItemInternalData::
_resetItemsInternal()
{
  ComponentItemSharedInfo* all_env_shared_info = allEnvSharedInfo();
  for (ComponentItemInternal& x : m_all_env_items_internal)
    x._reset(all_env_shared_info);

  ComponentItemSharedInfo* env_shared_info = envSharedInfo();
  for (ComponentItemInternal& x : m_env_items_internal)
    x._reset(env_shared_info);

  for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
    ArrayView<ComponentItemInternal> mat_items_internal = matItemsInternal(env->id());
    ComponentItemSharedInfo* mat_shared_info = matSharedInfo();
    for (ComponentItemInternal& x : mat_items_internal) {
      x._reset(mat_shared_info);
    }
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
resizeComponentItemInternals(Int32 max_local_id, Int32 total_env_cell)
{
  // Calcule le nombre total de ComponentItemInternal dont on a besoin
  // Les 'ComponentItemInternal' seront rangés dans 'm_component_item_internal_storage'
  // dans l'ordre suivant: AllEnvCell, EnvCell et chaque MatCell
  // TODO: Réserver de la place entre les vues pour ne pas être obligé de tout réallouer
  // dès qu'un des nombre d'élément change.
  Int32 total_nb_internal = 0;
  total_nb_internal += max_local_id; // Pour les AllEnvCell
  total_nb_internal += total_env_cell; // Pour les EnvCell
  for (const MeshEnvironment* env : m_material_mng->trueEnvironments())
    total_nb_internal += env->totalNbCellMat();

  // Redimensionne le conteneur. Il ne faut plus le modifier ensuite
  m_component_item_internal_storage.resize(total_nb_internal);

  // Maintenant récupère les vues sur chaque partie de 'm_component_item_internal_storage'
  {
    m_all_env_items_internal = m_component_item_internal_storage.subView(0, max_local_id);
    m_env_items_internal = m_component_item_internal_storage.subView(max_local_id, total_env_cell);
    Int32 index_in_container = max_local_id + total_env_cell;
    for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
      Int32 nb_cell_mat = env->totalNbCellMat();
      m_mat_items_internal[env->id()] = m_component_item_internal_storage.subView(index_in_container, nb_cell_mat);
      index_in_container += nb_cell_mat;
    }
  }

  _resetItemsInternal();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
_initSharedInfos()
{
  IItemFamily* family = m_material_mng->mesh()->cellFamily();
  ItemSharedInfo* item_shared_info = family->_internalApi()->commonItemSharedInfo();

  // NOTE : les champs 'm_components' sont des vues et donc il
  // ne faut pas que conteneurs associés de \a m_materials_mng soient modifiées.
  // Normalement ce n'est pas le cas, car la liste des constituants est fixe.
  ComponentItemSharedInfo* info_mat = sharedInfo(LEVEL_MATERIAL);
  info_mat->m_level = LEVEL_MATERIAL;
  info_mat->m_item_shared_info = item_shared_info;
  info_mat->m_components = m_material_mng->materialsAsComponents();

  ComponentItemSharedInfo* info_env = sharedInfo(LEVEL_ENVIRONMENT);
  info_env->m_level = LEVEL_ENVIRONMENT;
  info_env->m_item_shared_info = item_shared_info;
  info_env->m_components = m_material_mng->environmentsAsComponents();

  ComponentItemSharedInfo* info_all_env = sharedInfo(LEVEL_ALLENVIRONMENT);
  info_all_env->m_level = LEVEL_ALLENVIRONMENT;
  info_all_env->m_item_shared_info = item_shared_info;
  info_all_env->m_components = ConstArrayView<IMeshComponent*>();

  info() << "EndCreate ComponentItemInternalData nb_mat=" << info_mat->m_components.size()
         << " nb_env=" << info_env->m_components.size();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
