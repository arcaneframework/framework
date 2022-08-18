// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemInfoListView.h                                          (C) 2000-2022 */
/*                                                                           */
/* Vue sur une liste pour obtenir des informations sur les entités.          */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMINFOLISTVIEW_H
#define ARCANE_ITEMINFOLISTVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arcane/utils/ArrayView.h"

#include "arcane/ItemTypes.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class ItemFamily;
}

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane
{
class ItemSharedInfo;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \brief Vue sur une liste pour obtenir des informations sur les entités.
 *
 * Comme toutes les vues, ces instances sont temporaires et ne doivent pas être
 * conservées entre deux modifications de la famille associée.
 *
 * Via cette classe, il est possible de récupérer une instance de Item à partir
 * d'un numéro local ItemLocalId.
 */
class ARCANE_CORE_EXPORT ItemInfoListView
{
  friend class mesh::ItemFamily;
  // A supprimer lorqu'on n'aura plus besoin de _itemsInternal()
  friend class ItemVectorView;

 public:

  ItemInfoListView() = default;

  /*!
   * \brief Construit une vue associée à la famille \a family.
   *
   * \a family peut valoir \a nullptr auquel cas l'instance n'est
   * pas utilisable pour récupérer des informations sur les entités
   */
  explicit ItemInfoListView(IItemFamily* family);

 public:

  //! Famille associée
  IItemFamily* itemFamily() const { return m_family; }

  //! Entité associé du numéro local \a local_id
  inline Item operator[](ItemLocalId local_id) const;

  //! Entité associé du numéro local \a local_id
  inline Item operator[](Int32 local_id) const;

 private:

  // Seule ItemFamily peut créer des instances via ce constructeur
  ItemInfoListView(IItemFamily* family, ItemSharedInfo* shared_info, ItemInternalArrayView items_internal)
  : m_family(family)
  , m_item_shared_info(shared_info)
  , m_item_internal_list(items_internal)
  {}

  ItemInternalArrayView _itemsInternal() const { return m_item_internal_list; }

 private:

  IItemFamily* m_family = nullptr;
  ItemSharedInfo* m_item_shared_info = nullptr;
  ItemInternalArrayView m_item_internal_list;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
