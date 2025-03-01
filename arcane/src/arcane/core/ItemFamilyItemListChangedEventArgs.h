// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemFamilyItemListChangedEventArgs.h                        (C) 2000-2025 */
/*                                                                           */
/* Arguments de l'évènement pour l'ajout ou la supression d'entités.         */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMFAMILYITEMLISTCHANGEDARGS_H
#define ARCANE_CORE_ITEMFAMILYITEMLISTCHANGEDARGS_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"
#include "arcane/utils/UtilsTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \ingroup Mesh
 * \brief Arguments de l'évènement pour l'ajout ou la supression d'entités.
 */
class ARCANE_CORE_EXPORT ItemFamilyItemListChangedEventArgs
{
  friend mesh::DynamicMeshKindInfos;

 private:

  //! Constructeur. Uniquement pour DynamicMeshKindInfos.
  ItemFamilyItemListChangedEventArgs() = default;
  ItemFamilyItemListChangedEventArgs(IItemFamily* item_family, Int32 local_id, Int64 unique_id)
  : m_item_family(item_family)
  , m_local_id(local_id)
  , m_unique_id(unique_id)
  {}

 public:

  IItemFamily* itemFamily() const { return m_item_family; }
  Int32 localId() const { return m_local_id; }
  Int64 uniqueId() const { return m_unique_id; }
  bool isAdd() const { return m_is_add; }

 public:

  void setIsAdd(bool v) { m_is_add = v; }

 private:

  IItemFamily* m_item_family = nullptr;
  Int32 m_local_id = NULL_ITEM_LOCAL_ID;
  Int64 m_unique_id = NULL_ITEM_UNIQUE_ID;
  bool m_is_add = false;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
