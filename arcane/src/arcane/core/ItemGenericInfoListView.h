// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemGenericInfoListView.h                                   (C) 2000-2025 */
/*                                                                           */
/* Vue sur les informations génériques d'une famille d'entités.              */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMGENERICINFOLISTVIEW_H
#define ARCANE_CORE_ITEMGENERICINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemSharedInfo.h"
#include "arcane/core/ItemUniqueId.h"
#include "arcane/core/ItemLocalId.h"
#include "arcane/core/ItemFlags.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur les informations génériques d'une famille d'entités.
 *
 * Comme toutes les vues, les instances de cette classe sont temporaires
 * et ne doivent pas être conservées lorsque la famille associée est modifiée.
 */
class ARCANE_CORE_EXPORT ItemGenericInfoListView
{
  friend class ItemInfoListView;

 public:

  ItemGenericInfoListView() = default;
  /*!
   * \brief Construit une vue associée à la famille \a family.
   *
   * \a family peut valoir \a nullptr auquel cas l'instance n'est
   * pas utilisable pour récupérer des informations sur les entités
   */
  explicit ItemGenericInfoListView(IItemFamily* family);

  explicit ItemGenericInfoListView(const ItemInfoListView& info_list_view);

 public:

  //! Famille associée
  IItemFamily* itemFamily() const { return m_item_shared_info->itemFamily(); }

  //! Propriétaire de l'entité de numéro local \a local_id
  constexpr ARCCORE_HOST_DEVICE Int32 owner(Int32 local_id) const { return m_owners[local_id]; }

  //! Propriétaire de l'entité de numéro local \a local_id
  constexpr ARCCORE_HOST_DEVICE Int32 owner(ItemLocalId local_id) const { return m_owners[local_id.localId()]; }

  //! Type de l'entité de numéro local \a local_id
  constexpr ARCCORE_HOST_DEVICE Int16 typeId(Int32 local_id) const { return m_type_ids[local_id]; }

  //! Type de l'entité de numéro local \a local_id
  constexpr ARCCORE_HOST_DEVICE Int16 typeId(ItemLocalId local_id) const { return m_type_ids[local_id.localId()]; }

  //! uniqueId() de l'entité de numéro local \a local_id
  ARCCORE_HOST_DEVICE ItemUniqueId uniqueId(Int32 local_id) const
  {
    return ItemUniqueId{ m_unique_ids[local_id] };
  }

  //! uniqueId() de l'entité de numéro local \a local_id
  ARCCORE_HOST_DEVICE ItemUniqueId uniqueId(ItemLocalId local_id) const
  {
    return ItemUniqueId{ m_unique_ids[local_id.localId()] };
  }

  //! Indique si l'entité de numéro local \a local_id appartient au sous-domaine
  constexpr ARCCORE_HOST_DEVICE bool isOwn(Int32 local_id) const
  {
    return ItemFlags::isOwn(m_flags[local_id]);
  }

  //! Indique si l'entité de numéro local \a local_id appartient au sous-domaine
  constexpr ARCCORE_HOST_DEVICE bool isOwn(ItemLocalId local_id) const
  {
    return ItemFlags::isOwn(m_flags[local_id]);
  }

 private:

  // NOTE: Cette structure est utilisée dans le wrapping C#.
  // Si on modifie les champs, il faut mettre à jour la structure C# équivalente
  Int64ArrayView m_unique_ids;
  Int32ArrayView m_owners;
  Int32ArrayView m_flags;
  Int16ArrayView m_type_ids;
  ItemSharedInfo* m_item_shared_info = ItemSharedInfo::nullInstance();

 private:

  // Seule ItemFamily peut créer des instances via ce constructeur
  explicit ItemGenericInfoListView(ItemSharedInfo* shared_info);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
