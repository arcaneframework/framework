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

#include "arcane/accelerator/RunCommandLoop.h"
#include "arcane/accelerator/SpanViews.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions ComponentItemInternalData::Storage::
_allocInfo(const MemoryAllocationOptions& alloc_info, const String& base_name, const String& name)
{
  MemoryAllocationOptions opts(alloc_info);
  opts.setArrayName(base_name + name);
  return opts;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemInternalData::Storage::
Storage(const MemoryAllocationOptions& alloc_info, const String& base_name)
: m_first_sub_constituent_item_id_list(_allocInfo(alloc_info, base_name, "FirtSubComponentIdList"))
, m_super_component_item_local_id_list(_allocInfo(alloc_info, base_name, "SuperComponentIdList"))
, m_component_id_list(_allocInfo(alloc_info, base_name, "ComponentIdList"))
, m_nb_sub_constituent_item_list(_allocInfo(alloc_info, base_name, "NbSubConstituentItemList"))
, m_global_item_local_id_list(_allocInfo(alloc_info, base_name, "GlobalItemLocalIdList"))
, m_var_index_list(_allocInfo(alloc_info, base_name, "VarIndexList"))
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::Storage::
resize(Int32 new_size, ComponentItemSharedInfo* shared_info, RunQueue& queue)
{
  // On dimensionne au nombre d'éléments + 1.
  // On décale de 1 la vue pour qu'elle puisse être indexée avec l'entité
  // nulle (d'indice (-1)).
  Int32 true_size = new_size + 1;
  m_size = new_size;

  const bool force_resize = false;
  MemoryUtils::checkResizeArrayWithCapacity(m_first_sub_constituent_item_id_list, true_size, force_resize);
  MemoryUtils::checkResizeArrayWithCapacity(m_super_component_item_local_id_list, true_size, force_resize);
  MemoryUtils::checkResizeArrayWithCapacity(m_component_id_list, true_size, force_resize);
  MemoryUtils::checkResizeArrayWithCapacity(m_nb_sub_constituent_item_list, true_size, force_resize);
  MemoryUtils::checkResizeArrayWithCapacity(m_global_item_local_id_list, true_size, force_resize);
  MemoryUtils::checkResizeArrayWithCapacity(m_var_index_list, true_size, force_resize);

  auto first_sub_constituent_item_id_list = m_first_sub_constituent_item_id_list.smallSpan();
  auto super_component_item_local_id_list = m_super_component_item_local_id_list.smallSpan();
  auto component_id_list = m_component_id_list.smallSpan();
  auto nb_sub_constituent_item_list = m_nb_sub_constituent_item_list.smallSpan();
  auto global_item_local_id_list = m_global_item_local_id_list.smallSpan();
  auto var_index_list = m_var_index_list.smallSpan();

  // Met à jour les pointeurs.
  // On le fait sur l'accélérateur pour éviter des recopies avec le CPU.
  {
    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, 1)
    {
      shared_info->m_storage_size = new_size;
      first_sub_constituent_item_id_list[0] = {};
      component_id_list[0] = -1;
      nb_sub_constituent_item_list[0] = 0;
      global_item_local_id_list[0] = NULL_ITEM_LOCAL_ID;
      var_index_list[0].reset();

      shared_info->m_first_sub_constituent_item_id_data = first_sub_constituent_item_id_list.data() + 1;
      shared_info->m_super_component_item_local_id_data = super_component_item_local_id_list.data() + 1;
      shared_info->m_component_id_data = component_id_list.data() + 1;

      shared_info->m_nb_sub_constituent_item_data = nb_sub_constituent_item_list.data() + 1;
      shared_info->m_global_item_local_id_data = global_item_local_id_list.data() + 1;
      shared_info->m_var_index_data = var_index_list.data() + 1;
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ComponentItemInternalData::
ComponentItemInternalData(MeshMaterialMng* mmg)
: TraceAccessor(mmg->traceMng())
, m_material_mng(mmg)
, m_shared_infos(MemoryUtils::getAllocatorForMostlyReadOnlyData())
, m_all_env_storage(_allocOptions(), "AllEnvStorage")
, m_env_storage(_allocOptions(), "EnvStorage")
, m_mat_storage(_allocOptions(), "MatStorage")
{
  // Initialise l'instance nulle. Ce n'est pas grave si on le fait plusieurs fois
  // car la valeur ne change pas.
  ComponentItemSharedInfo::_setNullInstance();

  // Il y a une instance pour les MatCell, les EnvCell et les AllEnvCell
  // Il ne faut ensuite plus modifier ce tableau car on conserve des pointeurs
  // vers les éléments de ce tableau.
  m_shared_infos.resize(3);
  m_shared_infos.setDebugName("ComponentItemInternalDataSharedInfo");
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

MemoryAllocationOptions ComponentItemInternalData::
_allocOptions()
{
  return MemoryAllocationOptions(platform::getDefaultDataAllocator());
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
endCreate()
{
  const Int32 nb_env = m_material_mng->environments().size();
  m_mat_items_internal_range.resize(nb_env);
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
  if (!arcaneIsCheck())
    return;

  RunQueue queue(m_material_mng->runQueue());

  {
    ComponentItemSharedInfo* all_env_shared_info = allEnvSharedInfo();
    ComponentItemSharedInfo* env_shared_info = envSharedInfo();
    ComponentItemSharedInfo* mat_shared_info = matSharedInfo();
    const Int32 all_env_size = m_all_env_storage.size();
    const Int32 env_size = m_env_storage.size();
    const Int32 mat_size = m_mat_storage.size();
    Int32 max_size = math::max(all_env_size, env_size, mat_size);

    auto command = makeCommand(queue);
    command << RUNCOMMAND_LOOP1(iter, max_size)
    {
      auto [i] = iter();
      ConstituentItemIndex cii(i);
      if (i < all_env_size)
        all_env_shared_info->_reset(cii);
      if (i < env_size)
        env_shared_info->_reset(cii);
      if (i < mat_size)
        mat_shared_info->_reset(cii);
    };
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
resizeComponentItemInternals(Int32 max_local_id, Int32 total_nb_env_cell)
{
  RunQueue& queue(m_material_mng->runQueue());

  auto environments = m_material_mng->trueEnvironments();
  const Int32 nb_env = environments.size();
  NumArray<ComponentItemInternalRange, MDDim1> host_mats_range(nb_env);

  // Calcule le nombre total de ComponentItemInternal dont on a besoin
  Int32 total_nb_mat_cell = 0;
  for (const MeshEnvironment* env : m_material_mng->trueEnvironments())
    total_nb_mat_cell += env->totalNbCellMat();

  // Maintenant récupère les vues sur chaque partie du conteneur
  {
    m_all_env_items_internal_range.setRange(0, max_local_id);
    m_env_items_internal_range.setRange(0, total_nb_env_cell);
    Int32 index_in_container = 0;
    for (const MeshEnvironment* env : m_material_mng->trueEnvironments()) {
      Int32 nb_cell_mat = env->totalNbCellMat();
      Int32 env_id = env->id();
      host_mats_range[env_id].setRange(index_in_container, nb_cell_mat);
      index_in_container += nb_cell_mat;
    }
  }

  info(4) << "ResizeStorage max_local_id=" << max_local_id
          << " total_nb_env_cell=" << total_nb_env_cell
          << " total_nb_mat_cell=" << total_nb_mat_cell;
  {
    RunQueue::ScopedAsync sc(&queue);
    m_mat_items_internal_range.copy(host_mats_range, &queue);
    m_all_env_storage.resize(max_local_id, allEnvSharedInfo(), queue);
    m_env_storage.resize(total_nb_env_cell, envSharedInfo(), queue);
    m_mat_storage.resize(total_nb_mat_cell, matSharedInfo(), queue);
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
  ComponentItemSharedInfo* info_env = sharedInfo(LEVEL_ENVIRONMENT);
  ComponentItemSharedInfo* info_all_env = sharedInfo(LEVEL_ALLENVIRONMENT);

  info_mat->m_level = LEVEL_MATERIAL;
  info_mat->m_item_shared_info = item_shared_info;
  info_mat->m_components = m_material_mng->materialsAsComponents();
  info_mat->m_super_component_item_shared_info = info_env;

  info_env->m_level = LEVEL_ENVIRONMENT;
  info_env->m_item_shared_info = item_shared_info;
  info_env->m_components = m_material_mng->environmentsAsComponents();
  info_env->m_super_component_item_shared_info = info_all_env;
  info_env->m_sub_component_item_shared_info = info_mat;

  info_all_env->m_level = LEVEL_ALLENVIRONMENT;
  info_all_env->m_item_shared_info = item_shared_info;
  info_all_env->m_components = ConstArrayView<IMeshComponent*>();
  info_all_env->m_sub_component_item_shared_info = info_env;
  info() << "EndCreate ComponentItemInternalData nb_mat=" << info_mat->m_components.size()
         << " nb_env=" << info_env->m_components.size();
  info() << "EndCreate ComponentItemInternalData mat_shared_info=" << info_mat
         << " env_shared_info=" << info_env;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
