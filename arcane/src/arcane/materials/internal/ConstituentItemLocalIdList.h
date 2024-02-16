// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2024 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ConstituentItemLocalIdList.h                                (C) 2000-2024 */
/*                                                                           */
/* Gestion des listes d'identifiants locaux de 'ComponentItemInternal'.      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_MATERIALS_INTERNAL_CONSTITUENTITEMLOCALIDLIST_H
#define ARCANE_MATERIALS_INTERNAL_CONSTITUENTITEMLOCALIDLIST_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/TraceAccessor.h"
#include "arcane/utils/Array.h"

#include "arcane/materials/MaterialsGlobal.h"
#include "arcane/core/materials/ComponentItemInternal.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::Materials
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Liste d'indices locaux pour les 'ComponentItemInternal'.
 */
class ConstituentItemLocalIdList
{
 public:

  ConstituentItemLocalIdList(ComponentItemSharedInfo* shared_info);

 public:

  void resize(Int32 new_size);

 public:

  ConstArrayView<ComponentItemInternal*> itemsInternalView() const
  {
    return m_items_internal;
  }

  void setConstituentItem(Int32 index, ComponentItemInternalLocalId id)
  {
    m_item_internal_local_id_list[index] = id;
    m_items_internal[index] = m_shared_info->_itemInternal(id);
  }

  void copy(ConstArrayView<ComponentItemInternalLocalId> ids)
  {
    const Int32 size = ids.size();
    for (Int32 i = 0; i < size; ++i)
      setConstituentItem(i, ids[i]);
  }

  ConstArrayView<ComponentItemInternalLocalId> localIds() const
  {
    return m_item_internal_local_id_list;
  }

  ComponentItemInternal* itemInternal(Int32 index)
  {
    return m_shared_info->_itemInternal(localId(index));
  }

  ComponentItemInternalLocalId localId(Int32 index)
  {
    return m_item_internal_local_id_list[index];
  }

  ConstituentItemLocalIdListView view() const
  {
    return { m_shared_info, m_item_internal_local_id_list, m_items_internal };
  }

 private:

  //! Liste des ComponentItemInternal* pour ce constituant.
  UniqueArray<ComponentItemInternal*> m_items_internal;

  //! Liste des ComponentItemInternalLocalId pour ce constituant.
  UniqueArray<ComponentItemInternalLocalId> m_item_internal_local_id_list;

  ComponentItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
