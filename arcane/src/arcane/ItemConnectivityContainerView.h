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

 private:

 ItemConnectivityContainerView(SmallSpan<const Int32> _list,
                               SmallSpan<const Int32> _indexes,
                               SmallSpan<const Int32> _nb_item)
  : m_list(_list), m_indexes(_indexes), m_nb_item(_nb_item)
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

  SmallSpan<const Int32> m_list;
  SmallSpan<const Int32> m_indexes;
  SmallSpan<const Int32> m_nb_item;
};

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

} // End namespace Arcane

/*---------------------------------------------------------------------------*/
/*---------------------------------------------------------------------------*/

#endif
