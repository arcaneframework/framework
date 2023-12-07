// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2023 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ComponentItemInternalData.cc                                (C) 2000-2023 */
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
, m_all_env_items_internal(MemoryUtils::getAllocatorForMostlyReadOnlyData())
, m_env_items_internal(MemoryUtils::getAllocatorForMostlyReadOnlyData())
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
  m_mat_items_internal.reserve(nb_env);
  auto allocator = MemoryUtils::getAllocatorForMostlyReadOnlyData();
  for (Int32 i = 0; i < nb_env; ++i)
    m_mat_items_internal.add(UniqueArray<ComponentItemInternal>(allocator));

  _initSharedInfos();
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
resetEnvItemsInternal()
{
  ComponentItemSharedInfo* all_env_shared_info = allEnvSharedInfo();
  for (ComponentItemInternal& x : m_all_env_items_internal)
    x._reset(all_env_shared_info);

  ComponentItemSharedInfo* env_shared_info = envSharedInfo();
  for (ComponentItemInternal& x : m_env_items_internal)
    x._reset(env_shared_info);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
resizeAndResetMatCellForEnvironment(Int32 env_index, Int32 size)
{
  m_mat_items_internal[env_index].resize(size);

  ArrayView<ComponentItemInternal> mat_items_internal = matItemsInternal(env_index);
  ComponentItemSharedInfo* mat_shared_info = matSharedInfo();
  for (Integer i = 0; i < size; ++i) {
    ComponentItemInternal& ref_ii = mat_items_internal[i];
    ref_ii._reset(mat_shared_info);
  }
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ComponentItemInternalData::
_initSharedInfos()
{
  IItemFamily* family = m_material_mng->mesh()->cellFamily();
  ItemSharedInfo* item_shared_info = family->_internalApi()->commonItemSharedInfo();

  ComponentItemSharedInfo* info_mat = sharedInfo(LEVEL_MATERIAL);
  info_mat->m_level = LEVEL_MATERIAL;
  info_mat->m_item_shared_info = item_shared_info;

  ComponentItemSharedInfo* info_env = sharedInfo(LEVEL_ENVIRONMENT);
  info_env->m_level = LEVEL_ENVIRONMENT;
  info_env->m_item_shared_info = item_shared_info;

  ComponentItemSharedInfo* info_all_env = sharedInfo(LEVEL_ALLENVIRONMENT);
  info_all_env->m_level = LEVEL_ALLENVIRONMENT;
  info_all_env->m_item_shared_info = item_shared_info;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
