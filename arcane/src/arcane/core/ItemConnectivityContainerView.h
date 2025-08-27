// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2025 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityContainerView.h                             (C) 2000-2025 */
/*                                                                           */
/* Vues sur les conteneurs contenant les connectivités.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_CORE_ITEMCONNECTIVITYCONTAINERVIEW_H
#define ARCANE_CORE_ITEMCONNECTIVITYCONTAINERVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include "arcane/core/ArcaneTypes.h"
#include "arcane/core/ItemTypes.h"
#include "arcane/core/ItemLocalId.h"

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

namespace Arcane::mesh
{
class IncrementalItemConnectivityBase;
}

namespace Arcane
{
class ItemInternalConnectivityList;

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/
/*!
 * \internal
 * \brief Vues sur les conteneurs contenant les connectivités.
 * Cette classe permet de rendre opaque en dehors d'Arcane les conteneurs utilisés
 * pour faciliter leur changement éventuel.
 */
class ARCANE_CORE_EXPORT ItemConnectivityContainerView
{
  // Liste des classes qui ont le droit de créer ou de récupérer directement
  // des instances de cette classe
  friend ItemInternalConnectivityList;
  friend IndexedItemConnectivityViewBase;
  friend IndexedItemConnectivityViewBase2;
  friend mesh::IncrementalItemConnectivityBase;
  template <typename ItemType1, typename ItemType2>
  friend class IndexedItemConnectivityGenericViewT;

 private:

  ItemConnectivityContainerView() = default;
  ItemConnectivityContainerView(SmallSpan<const Int32> _list,
                                SmallSpan<const Int32> _indexes,
                                SmallSpan<const Int32> _nb_connected_item)
  : m_list_data(_list.data())
  , m_indexes(_indexes.data())
  , m_nb_connected_items(_nb_connected_item.data())
  , m_list_data_size(_list.size())
  , m_nb_item(_nb_connected_item.size())
  {
#ifdef ARCANE_CHECK
    _checkSize( _indexes.size(), _nb_connected_item.size());
#endif
  }

 public:

  /*!
   * \brief Vérifie que les deux instances \a this et \a rhs ont les mêmes valeurs.
   *
   * Lance un FatalErrorException si ce n'est pas le cas.
   */
  void checkSame(ItemConnectivityContainerView rhs) const;

 private:

  //! Liste des entités connectées à l'entité de localId() \a lid
  template <typename ItemType> constexpr ARCCORE_HOST_DEVICE
  ItemLocalIdListViewT<ItemType>
  itemsIds(ItemLocalId lid) const
  {
    ARCANE_CHECK_AT(lid.localId(), m_nb_item);
    Int32 x = m_indexes[lid];
    ARCANE_CHECK_AT(x, m_list_data_size);
    auto* p = &m_list_data[x];
    // TODO: LOCAL_ID_OFFSET
    return { p, m_nb_connected_items[lid], 0 };
  }

  //! \a index-ème entité connectée à l'entité de localId() \a lid
  template <typename ItemLocalIdType> constexpr ARCCORE_HOST_DEVICE
  ItemLocalIdType
  itemId(ItemLocalId lid, Int32 index) const
  {
    ARCANE_CHECK_AT(lid.localId(), m_nb_item);
    Int32 x = m_indexes[lid] + index;
    ARCANE_CHECK_AT(x, m_list_data_size);
    return ItemLocalIdType(m_list_data[x]);
  }

  //! Tableau des indices dans la table de connectivités
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const Int32>
  indexes() const { return { m_indexes, m_nb_item }; }

  //! Tableau du nombre d'entités connectées à une autre entité.
  constexpr ARCCORE_HOST_DEVICE SmallSpan<const Int32>
  nbConnectedItems() const { return { m_nb_connected_items,m_nb_item }; }

  //! Nombre d'entités
  constexpr ARCCORE_HOST_DEVICE Int32 nbItem() const { return m_nb_item; }

 private:

  const Int32* m_list_data = nullptr;
  const Int32* m_indexes = nullptr;
  const Int32* m_nb_connected_items = nullptr;
  Int32 m_list_data_size = 0;
  Int32 m_nb_item = 0;

 private:

  void _checkSize(Int32 indexes_size, Int32 nb_connected_item_size);
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
