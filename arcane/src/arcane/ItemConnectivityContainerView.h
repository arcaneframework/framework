// -*- tab-width: 2; indent-tabs-mode: nil; coding: utf-8-with-signature -*-
//-----------------------------------------------------------------------------
// Copyright 2000-2022 CEA (www.cea.fr) IFPEN (www.ifpenergiesnouvelles.com)
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: Apache-2.0
//-----------------------------------------------------------------------------
/*---------------------------------------------------------------------------*/
/* ItemConnectivityContainerView.h                             (C) 2000-2022 */
/*                                                                           */
/* Vues sur les conteneurs contenant les connectivités.                      */
/*---------------------------------------------------------------------------*/
#ifndef ARCANE_ITEMCONNECTIVITYCONTAINERVIEW_H
#define ARCANE_ITEMCONNECTIVITYCONTAINERVIEW_H
/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#include "arccore/base/Span.h"

#include "arcane/ArcaneTypes.h"
#include "arcane/ItemTypes.h"
#include "arcane/ItemLocalId.h"

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
  friend mesh::IncrementalItemConnectivityBase;
  template <typename ItemType1, typename ItemType2>
  friend class IndexedItemConnectivityGenericViewT;

 private:

  ItemConnectivityContainerView() = default;
  ItemConnectivityContainerView(SmallSpan<const ItemLocalId> _list,
                                SmallSpan<const Int32> _indexes,
                                SmallSpan<const Int32> _nb_item)
  : m_list_data(_list)
  , m_indexes(_indexes)
  , m_nb_item(_nb_item)
  {
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
  ItemLocalIdViewT<ItemType>
  itemsIds(ItemLocalId lid) const
  {
    using LocalIdType = typename ItemLocalIdViewT<ItemType>::LocalIdType;
    auto* p = static_cast<const LocalIdType*>(&m_list_data[m_indexes[lid]]);
    return { p, m_nb_item[lid] };
  }

  //! \a index-ème entité connectée à l'entité de localId() \a lid
  template <typename ItemLocalIdType> constexpr ARCCORE_HOST_DEVICE
  ItemLocalIdType
  itemId(ItemLocalId lid, Int32 index) const
  {
    return ItemLocalIdType(m_list_data[m_indexes[lid] + index]);
  }

 private:

  SmallSpan<const ItemLocalId> m_list_data;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_nb_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
