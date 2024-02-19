// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemLocalIdList.cc                               (C) 2000-2024 */
/*                                                                           */
/* Gestion des listes d'identifiants locaux de 'ComponentItemInternal'.      */
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/core/materials/internal/ConstituentItemLocalIdList.h"

#include "arcane/utils/MemoryUtils.h"

#include "arcane/materials/internal/MeshMaterialMng.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

ConstituentItemLocalIdList::
ConstituentItemLocalIdList(ComponentItemSharedInfo* shared_info)
: m_constituent_item_index_list(MemoryUtils::getAllocatorForMostlyReadOnlyData())
, m_shared_info(shared_info)
{
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

void ConstituentItemLocalIdList::
resize(Int32 new_size)
{
  m_constituent_item_index_list.resize(new_size);
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
