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
#ifndef ARCANE_CORE_MATERIALS_INTERNAL_CONSTITUENTITEMLOCALIDLIST_H
#define ARCANE_CORE_MATERIALS_INTERNAL_CONSTITUENTITEMLOCALIDLIST_H
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
class ARCANE_CORE_EXPORT ConstituentItemLocalIdList
{
 public:

  ConstituentItemLocalIdList(ComponentItemSharedInfo* shared_info, const String& debug_name);

 public:

  void resize(Int32 new_size);

 public:

  void setConstituentItem(Int32 index, ConstituentItemIndex id)
  {
    m_constituent_item_index_list[index] = id;
  }

  void copy(ConstArrayView<ConstituentItemIndex> ids)
  {
    const Int32 size = ids.size();
    resize(size);
    for (Int32 i = 0; i < size; ++i)
      setConstituentItem(i, ids[i]);
  }

  void copy(const ConstituentItemLocalIdListView& view)
  {
    m_shared_info = view.m_component_shared_info;
    const Int32 size = view.m_ids.size();
    resize(size);
    for (Int32 i = 0; i < size; ++i)
      setConstituentItem(i, view.m_ids[i]);
  }

  /*!
   * \brief Copie les constituents partitionnés en partie pure et partielle.
   */
  void copyPureAndPartial(ConstArrayView<ConstituentItemIndex> ids)
  {
    Int32 nb = ids.size();

    resize(nb);
    for (Int32 i = 0; i < nb; ++i)
      m_constituent_item_index_list[i] = ids[i];
  }

  ConstArrayView<ConstituentItemIndex> localIds() const
  {
    return m_constituent_item_index_list;
  }
  SmallSpan<ConstituentItemIndex> mutableLocalIds()
  {
    return m_constituent_item_index_list.view();
  }
  matimpl::ConstituentItemBase itemBase(Int32 index) const
  {
    return m_shared_info->_item(localId(index));
  }
  ConstituentItemIndex localId(Int32 index) const
  {
    return m_constituent_item_index_list[index];
  }

  MatVarIndex variableIndex(Int32 index) const
  {
    return m_shared_info->_varIndex(localId(index));
  }

  ConstituentItemLocalIdListView view() const
  {
    return { m_shared_info, m_constituent_item_index_list };
  }

 private:

  //! Liste des ConstituentItemIndex pour ce constituant.
  UniqueArray<ConstituentItemIndex> m_constituent_item_index_list;

  ComponentItemSharedInfo* m_shared_info = nullptr;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane::Materials

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
