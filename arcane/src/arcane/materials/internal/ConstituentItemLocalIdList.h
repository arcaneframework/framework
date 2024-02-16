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

  ConstituentItemLocalIdList();

 public:

  void resize(Int32 new_size);

 public:

  ConstArrayView<ComponentItemInternal*> itemsInternalView() const
  {
    return m_items_internal;
  }

  ArrayView<ComponentItemInternal*> itemsInternalView()
  {
    return m_items_internal;
  }

 private:

  //! Liste des ComponentItemInternal* pour ce constituant.
  UniqueArray<ComponentItemInternal*> m_items_internal;

  //! Liste des ComponentItemInternalLocalId pour ce constituant.
  UniqueArray<ComponentItemInternalLocalId> m_item_internal_local_id_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
