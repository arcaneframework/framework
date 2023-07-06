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
{
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
  for( Int32 i=0; i<nb_env; ++i )
    m_mat_items_internal.add(UniqueArray<ComponentItemInternal>(allocator));
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
